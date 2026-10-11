from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import RTCConfig, RobotActionLayout


PROTOCOL_VERSION = "lingbot-robotwin-rtc/1"


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class RTCRequest:
    observation: dict[str, Any]
    episode_id: int
    request_id: int
    rtc: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        return dict(self.observation, protocol_version=PROTOCOL_VERSION,
                    episode_id=self.episode_id, request_id=self.request_id,
                    rtc=self.rtc)


@dataclass(frozen=True)
class RTCResponse:
    episode_id: int
    request_id: int
    actions: np.ndarray
    server_timing: dict[str, Any]
    rtc_ack: dict[str, Any]


def build_bootstrap_request(observation: dict[str, Any], *, episode_id: int,
                            request_id: int) -> RTCRequest:
    return RTCRequest(dict(observation), int(episode_id), int(request_id), {"enabled": False})


def build_rtc_request(observation: dict[str, Any], previous_actions: np.ndarray, *,
                      episode_id: int, request_id: int, old_index: int,
                      config: RTCConfig) -> RTCRequest:
    actions = np.asarray(previous_actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] != config.chunk_horizon or not np.isfinite(actions).all():
        raise ProtocolError("previous_actions must be finite [50, D]")
    if old_index != config.request_index:
        raise ProtocolError("RTC request must be made at configured request_index")
    previous_tail = actions[int(old_index):].copy()
    rtc = {
        "enabled": True,
        "previous_actions": previous_tail,
        "old_index": int(old_index),
        "prefix_length": int(len(previous_tail)),
        "inference_delay": int(config.fixed_delay_steps),
        "execution_horizon": int(config.execution_horizon),
        "max_guidance_weight": float(config.max_guidance_weight),
        "prefix_attention_schedule": config.prefix_attention_schedule,
    }
    return RTCRequest(dict(observation), int(episode_id), int(request_id), rtc)


def validate_response(payload: dict[str, Any], expected_episode_id: int,
                      expected_request_id: int,
                      layout: RobotActionLayout) -> RTCResponse:
    if not isinstance(payload, dict):
        raise ProtocolError("response must be a mapping")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError("protocol version mismatch")
    if payload.get("episode_id") != expected_episode_id:
        raise ProtocolError("stale episode response")
    if payload.get("request_id") != expected_request_id:
        raise ProtocolError("stale request response")
    actions = np.asarray(payload.get("action"), dtype=np.float32)
    expected_shape = (50, layout.action_dim)
    if actions.shape != expected_shape:
        raise ProtocolError(f"action must have shape {expected_shape}, got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ProtocolError("action contains NaN or Inf")
    timing = payload.get("server_timing", {})
    ack = payload.get("rtc_ack", {})
    if not isinstance(timing, dict) or not isinstance(ack, dict):
        raise ProtocolError("server_timing and rtc_ack must be mappings")
    return RTCResponse(expected_episode_id, expected_request_id, actions, timing, ack)
