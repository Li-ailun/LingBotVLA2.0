#!/usr/bin/env python3
from __future__ import annotations

import argparse
from functools import partial
import os
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np


TOOL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TOOL_ROOT.parents[1]
for _path in (str(TOOL_ROOT), str(PROJECT_ROOT), str(PROJECT_ROOT / "lingbot-vla-v2")) :
    if _path not in sys.path:
        sys.path.insert(0, _path)

from simulation_rtc.protocol import PROTOCOL_VERSION


_RESERVED = {"protocol_version", "episode_id", "request_id", "rtc", "reset_policy", "robo_name"}


def normalize_physical_actions(feature_transform, actions):
    import torch
    physical = torch.from_numpy(np.asarray(actions, dtype=np.float32))
    item = {"action_is_pad": torch.zeros(physical.shape[0], dtype=torch.bool)}
    for target_key in feature_transform.actions:
        mapping = feature_transform.key_mapping[target_key]["origin_keys"]
        pieces = []
        for origin_key, bounds in mapping.items():
            if origin_key.split("*")[0] != "action":
                raise ValueError("RTC action mapping must originate from physical action")
            pieces.append(physical[..., int(bounds["start"]):int(bounds["end"])])
        item[target_key] = torch.cat(pieces, dim=-1)
    if feature_transform.normalizer is not None:
        item = feature_transform.normalizer.normalize(item)
    import torch.nn.functional as F
    padded_actions = []
    padded_masks = []
    for joint in feature_transform.feature_config.joints:
        action_key = f"action.{joint}"
        max_dim = int(feature_transform.feature_config.joints_max_dim[joint])
        if action_key in feature_transform.actions:
            value = item[action_key]
            pad_len = max_dim - value.shape[-1]
            if pad_len < 0:
                raise ValueError(f"negative RTC action padding for {action_key}")
            padded_actions.append(F.pad(value, (0, pad_len)))
            padded_masks.append(F.pad(torch.ones(value.shape[-1], dtype=torch.bool), (0, pad_len)))
        else:
            padded_actions.append(torch.zeros((physical.shape[0], max_dim), dtype=torch.float32))
            padded_masks.append(torch.zeros(max_dim, dtype=torch.bool))
    model_pad = int(feature_transform.model_config.max_action_dim) - sum(part.shape[-1] for part in padded_actions)
    if model_pad < 0:
        raise ValueError("RTC padded actions exceed model max_action_dim")
    if model_pad:
        padded_actions.append(torch.zeros((physical.shape[0], model_pad), dtype=torch.float32))
        padded_masks.append(torch.zeros(model_pad, dtype=torch.bool))
    target = torch.cat(padded_actions, dim=-1).to(dtype=torch.float32, device="cpu").numpy()
    mask = torch.cat(padded_masks, dim=-1).to(dtype=torch.bool, device="cpu").numpy()[None, None]
    return target, mask


class RTCServerCore:
    """Protocol/validation shell with injectable model-specific functions."""

    def __init__(self, ordinary_infer: Callable[[dict], np.ndarray],
                 rtc_infer: Callable[[dict, dict], np.ndarray],
                 normalize_target: Callable[[np.ndarray], Any], action_dim: int):
        self._ordinary_infer = ordinary_infer
        self._rtc_infer = rtc_infer
        self._normalize_target = normalize_target
        self._action_dim = int(action_dim)

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("RTC protocol version mismatch")
        episode_id, request_id = int(payload["episode_id"]), int(payload["request_id"])
        rtc = payload.get("rtc") or {"enabled": False}
        observation = {key: value for key, value in payload.items() if key not in _RESERVED}
        if rtc.get("enabled"):
            physical = np.asarray(rtc.get("previous_actions"), dtype=np.float32)
            if (physical.ndim != 2 or physical.shape[1] != self._action_dim
                    or not 1 <= physical.shape[0] <= 50 or not np.isfinite(physical).all()):
                raise ValueError("RTC previous_actions must be finite [1..50, action_dim]")
            prefix_length = int(rtc.get("prefix_length", 0))
            if prefix_length != physical.shape[0]:
                raise ValueError("RTC prefix_length must match previous_actions length")
            if physical.shape[0] < 50:
                padding = np.repeat(physical[-1:], 50 - physical.shape[0], axis=0)
                physical = np.concatenate((physical, padding), axis=0)
            normalized = self._normalize_target(physical)
            if isinstance(normalized, tuple):
                target, mask = normalized
            else:
                target = normalized
                mask = np.ones((1, 1, np.asarray(target).shape[-1]), dtype=bool)
            target = np.asarray(target, dtype=np.float32)
            if target.ndim == 2:
                target = target[None]
            mask = np.asarray(mask, dtype=bool)
            request = dict(rtc, target=target, valid_action_mask=mask)
            actions = self._rtc_infer(observation, request)
        else:
            actions = self._ordinary_infer(observation)
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (50, self._action_dim):
            raise ValueError(f"model action must have shape (50, {self._action_dim})")
        if not np.isfinite(actions).all():
            raise ValueError("model action must be finite")
        return {
            "protocol_version": PROTOCOL_VERSION,
            "episode_id": episode_id,
            "request_id": request_id,
            "action": actions,
            "rtc_ack": {
                "enabled": bool(rtc.get("enabled", False)),
                "inference_delay": int(rtc.get("inference_delay", 0)),
                "execution_horizon": int(rtc.get("execution_horizon", 0)),
                "prefix_attention_schedule": rtc.get("prefix_attention_schedule"),
            },
        }


def build_production_server_class():
    import torch
    from evaluation.lingbot_action_step2000.policy_server import ExportedCheckpointServer
    from deploy.lingbot_vla_v2_policy import WebsocketPolicyServer, str2bool
    from myscripts.inference_tools.lingbot_rtc.modeling_lingbot_vla_v2_rtc import install_rtc_sampler
    from myscripts.inference_tools.lingbot_rtc.rtc_request import LingBotRTCRequest

    schedule_ids = {"linear": 0, "exp": 1, "ones": 2, "zeros": 3}

    class ExportedCheckpointRTCServer(ExportedCheckpointServer):
        def load_vla(self, path_to_pi_model):
            policy = super().load_vla(path_to_pi_model)
            install_rtc_sampler(policy)
            return policy

        def reset(self, robo_name, path_to_pi_model=None):
            super().reset(robo_name, path_to_pi_model)
            self._rtc_core = RTCServerCore(
                self._ordinary_physical, self._rtc_physical,
                self._normalize_physical_chunk, action_dim=14,
            )

        def infer(self, payload, center_crop=True, return_normalized=False):
            if "reset" in payload:
                return super().infer(payload, center_crop=center_crop,
                                     return_normalized=return_normalized)
            if payload.get("reset_policy"):
                self.reset(payload.get("robo_name", "robotwin"))
            return self._rtc_core.infer(payload)

        def _ordinary_physical(self, observation):
            result = super().infer(observation)
            return np.asarray(result["action"], dtype=np.float32)

        def _normalize_physical_chunk(self, actions):
            return normalize_physical_actions(self.vla.feature_transform, actions)

        def _rtc_physical(self, observation, request):
            applied = self._prepare_model_input(observation)
            batch = {}
            for key, value in applied.items():
                batch[key] = value.unsqueeze(0) if isinstance(value, torch.Tensor) else [value]
            dtype = torch.bfloat16 if self.use_bf16 else torch.float32
            target = torch.from_numpy(request["target"]).to(device="cuda", dtype=dtype)
            mask = torch.from_numpy(request["valid_action_mask"]).to(device="cuda")
            rtc_request = LingBotRTCRequest(
                target=target,
                prefix_length=int(request["prefix_length"]),
                inference_delay=int(request["inference_delay"]),
                execution_horizon=int(request["execution_horizon"]),
                max_guidance_weight=float(request["max_guidance_weight"]),
                prefix_attention_schedule=schedule_ids[request["prefix_attention_schedule"]],
                valid_action_mask=mask,
            )
            sampler = partial(self.vla.model.sample_actions_rtc, rtc_request=rtc_request)
            normalized = self.vla.sample_actions_batch(
                batch, self.use_bf16, False, capture_time=False, sample_compile_fn=sampler,
            )
            physical = self._unapply_batched_actions([applied], normalized)
            return np.asarray(physical["action"][0], dtype=np.float32)

    return ExportedCheckpointRTCServer, WebsocketPolicyServer, str2bool


def main() -> None:
    Server, WebsocketPolicyServer, str2bool = build_production_server_class()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--use_length", type=int, default=50)
    parser.add_argument("--port", type=int, default=8006)
    parser.add_argument("--use_bf16", type=str2bool, default=True)
    parser.add_argument("--use_fp32", type=str2bool, default=False)
    parser.add_argument("--use_compile", type=str2bool, default=True)
    args = parser.parse_args()
    model = Server(args.model_path, use_length=args.use_length, chunk_ret=True,
                   use_bf16=args.use_bf16, use_fp32=args.use_fp32,
                   use_compile=args.use_compile)
    WebsocketPolicyServer(model, port=args.port).serve_forever()


if __name__ == "__main__":
    main()
