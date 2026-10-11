from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RTCConfig:
    chunk_horizon: int
    request_index: int
    fixed_delay_steps: int
    execution_horizon: int
    crossfade_steps: int
    ema_alpha: float
    max_guidance_weight: float
    prefix_attention_schedule: str
    request_timeout_s: float

    def __post_init__(self) -> None:
        if self.chunk_horizon != 50:
            raise ValueError("LingBot RTC requires chunk_horizon=50")
        if not 0 <= self.request_index < self.chunk_horizon:
            raise ValueError("request_index is out of bounds")
        remaining = self.chunk_horizon - self.request_index
        if not 0 <= self.fixed_delay_steps <= self.execution_horizon <= remaining:
            raise ValueError("require 0 <= fixed_delay_steps <= execution_horizon <= remaining chunk")
        if self.crossfade_steps < 0:
            raise ValueError("crossfade_steps must be non-negative")
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if self.max_guidance_weight < 0:
            raise ValueError("max_guidance_weight must be non-negative")
        if self.prefix_attention_schedule not in {"linear", "exp", "ones", "zeros"}:
            raise ValueError("unsupported prefix_attention_schedule")
        if self.request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive")

    @classmethod
    def from_yaml(cls, path: Path | str) -> "RTCConfig":
        with Path(path).open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        def override(name: str, default, convert):
            value = os.environ.get(f"LINGBOT_RTC_{name}")
            return default if value is None else convert(value)

        return cls(
            chunk_horizon=int(raw["chunk_horizon"]),
            request_index=override("REQUEST_INDEX", raw["request_index"], int),
            fixed_delay_steps=override("FIXED_DELAY_STEPS", raw["fixed_delay_steps"], int),
            execution_horizon=override("EXECUTION_HORIZON", raw["execution_horizon"], int),
            crossfade_steps=override("CROSSFADE_STEPS", raw["smoothing"]["crossfade_steps"], int),
            ema_alpha=override("EMA_ALPHA", raw["smoothing"]["ema_alpha"], float),
            max_guidance_weight=float(raw["rtc"]["max_guidance_weight"]),
            prefix_attention_schedule=str(raw["rtc"]["prefix_attention_schedule"]),
            request_timeout_s=float(raw["fallback"]["request_timeout_s"]),
        )


@dataclass(frozen=True)
class RobotActionLayout:
    action_dim: int
    gripper_indices: tuple[int, ...]

    @classmethod
    def from_yaml(cls, path: Path | str) -> "RobotActionLayout":
        with Path(path).open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        occupied: set[int] = set()
        grippers: list[int] = []
        maximum = 0
        for feature_entry in raw.get("actions", []):
            if not isinstance(feature_entry, dict) or len(feature_entry) != 1:
                raise ValueError("each action feature must contain exactly one mapping")
            feature_name, feature = next(iter(feature_entry.items()))
            for source in feature.get("origin_keys", []):
                if not isinstance(source, dict) or len(source) != 1 or "action" not in source:
                    raise ValueError("all exported action sources must use origin key 'action'")
                bounds = source["action"]
                start, end = int(bounds["start"]), int(bounds["end"])
                if start < 0 or end <= start:
                    raise ValueError("invalid action slice")
                indices = set(range(start, end))
                if occupied & indices:
                    raise ValueError("action slices overlap")
                occupied.update(indices)
                maximum = max(maximum, end)
                if feature_name == "action.effector.position":
                    if end - start != 1:
                        raise ValueError("each effector action must be scalar")
                    grippers.append(start)
        if not occupied or occupied != set(range(maximum)):
            raise ValueError("action slices must cover a contiguous vector")
        if not grippers or len(set(grippers)) != len(grippers):
            raise ValueError("gripper indices are missing or ambiguous")
        return cls(action_dim=maximum, gripper_indices=tuple(grippers))
