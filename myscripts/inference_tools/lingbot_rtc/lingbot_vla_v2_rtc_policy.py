"""Strict physical I/O boundary for the isolated LingBot RTC backend."""

from pathlib import Path
from time import monotonic

import numpy as np


CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


class LingBotVlaV2RTCPolicy:
    """Validate R1 Pro inputs; the injected backend owns LingBot transforms/model."""

    def __init__(self, model_path=None, qwen_path=None, robot_config_path=None,
                 norm_stats_path=None, *, use_bf16=True, use_compile=False, backend=None):
        self.model_path = None if model_path is None else Path(model_path)
        self.qwen_path = None if qwen_path is None else Path(qwen_path)
        self.robot_config_path = None if robot_config_path is None else Path(robot_config_path)
        self.norm_stats_path = None if norm_stats_path is None else Path(norm_stats_path)
        self.use_bf16, self.use_compile = use_bf16, use_compile
        self.backend = backend
        errors = []
        for label, path in (("robot config", self.robot_config_path), ("normalization stats", self.norm_stats_path)):
            if path is None or not path.is_file():
                errors.append(f"missing {label}: {path}")
        self.production_asset_errors = tuple(errors)
        self.production_assets_ready = not errors

    @staticmethod
    def _validate_observation(observation):
        if set(observation.get("images", {})) != set(CAMERAS):
            raise ValueError(f"images must contain exactly {CAMERAS}")
        state = np.asarray(observation.get("state"), dtype=np.float32)
        if state.shape != (20,) or not np.isfinite(state).all():
            raise ValueError("state must be finite shape (20,)")
        if not str(observation.get("prompt", "")).strip():
            raise ValueError("prompt must be non-empty")

    def physical_observation_to_model(self, observation):
        if not self.production_assets_ready:
            raise RuntimeError("production assets are not ready: " + "; ".join(self.production_asset_errors))
        return observation

    def physical_rtc_target_to_model(self, target):
        if not self.production_assets_ready:
            raise RuntimeError("production assets are not ready")
        return np.asarray(target, dtype=np.float32)

    def model_actions_to_physical(self, actions):
        if not self.production_assets_ready:
            raise RuntimeError("production assets are not ready")
        return np.asarray(actions, dtype=np.float32)

    def infer(self, observation):
        self._validate_observation(observation)
        model_observation = self.physical_observation_to_model(observation)
        rtc = observation.get("rtc") or {}
        start = monotonic()
        if self.backend is None:
            raise RuntimeError("LingBot backend has not been loaded")
        actions = self.backend.infer(model_observation, rtc_request=rtc if rtc.get("enabled") else None)
        actions = self.model_actions_to_physical(actions)
        if actions.shape != (50, 20) or not np.isfinite(actions).all():
            raise ValueError(f"backend must return finite actions [50, 20], got {actions.shape}")
        return {"actions": actions, "policy_timing": {
            "infer_ms": (monotonic() - start) * 1000.0,
            "rtc_enabled": bool(rtc.get("enabled", False)),
            "rtc_prefix_len": int(rtc.get("prev_chunk_left_over_length", 0)),
            "rtc_inference_delay": int(rtc.get("inference_delay", 0)),
            "rtc_execution_horizon": int(rtc.get("execution_horizon", 0)),
        }}
