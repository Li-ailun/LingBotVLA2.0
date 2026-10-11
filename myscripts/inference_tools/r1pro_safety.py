#!/usr/bin/env python3
# =============================================================================
# R1Pro OpenPI communication module: action safety layer.
# 功能说明：执行前对 action20 做安全处理，包括 NaN/inf 检查、absolute 范围限幅、
# 单步变化量限制、可选固定 torso target 等。该文件用于防止模型或测试动作突然
# 输出离谱数值，降低实机执行风险。
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from r1pro_action_utils import validate_action20, validate_state20
from r1pro_constants import ACTION_HIGH, ACTION_LOW, FIXED_TORSO_TARGET, IDX_TORSO, MAX_STEP_DELTA


@dataclass
class SafetyReport:
    accepted: bool = True
    reasons: list[str] = field(default_factory=list)
    clipped_abs_range: bool = False
    clipped_step_delta: bool = False
    forced_torso: bool = False

    def add(self, msg: str):
        self.reasons.append(msg)
        self.accepted = False

    def summary(self) -> str:
        flags = []
        if self.clipped_abs_range:
            flags.append("abs_clip")
        if self.clipped_step_delta:
            flags.append("step_clip")
        if self.forced_torso:
            flags.append("fixed_torso")
        if self.reasons:
            flags.extend(self.reasons)
        return ", ".join(flags) if flags else "ok"


class ActionSafetyLimiter:
    """Safety layer for absolute R1Pro action20 before publishing.

    The model/OpenPI server should already return absolute action20. This class
    does NOT do delta-to-absolute conversion. It only checks, clips, and limits
    jumps for safer real-robot execution.
    """

    def __init__(
        self,
        enable_abs_clip: bool = True,
        enable_step_limit: bool = True,
        force_fixed_torso_target: bool = False,
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
        max_step_delta: np.ndarray | None = None,
        fixed_torso_target: np.ndarray | None = None,
    ):
        self.enable_abs_clip = bool(enable_abs_clip)
        self.enable_step_limit = bool(enable_step_limit)
        self.force_fixed_torso_target = bool(force_fixed_torso_target)
        self.action_low = ACTION_LOW.copy() if action_low is None else validate_action20(action_low)
        self.action_high = ACTION_HIGH.copy() if action_high is None else validate_action20(action_high)
        self.max_step_delta = MAX_STEP_DELTA.copy() if max_step_delta is None else validate_action20(max_step_delta)
        self.fixed_torso_target = (
            FIXED_TORSO_TARGET.copy()
            if fixed_torso_target is None
            else np.asarray(fixed_torso_target, dtype=np.float32).reshape(4)
        )

    def sanitize(self, action20, state20) -> tuple[np.ndarray, SafetyReport]:
        report = SafetyReport()

        try:
            action = validate_action20(action20).copy()
            state = validate_state20(state20)
        except Exception as e:
            report.add(str(e))
            return np.zeros((20,), dtype=np.float32), report

        if not np.all(np.isfinite(action)):
            report.add("action has NaN/inf")
            return state.copy(), report

        if self.force_fixed_torso_target:
            action[IDX_TORSO] = self.fixed_torso_target
            report.forced_torso = True

        if self.enable_abs_clip:
            clipped = np.clip(action, self.action_low, self.action_high)
            if not np.allclose(clipped, action, atol=0.0, rtol=0.0):
                report.clipped_abs_range = True
                action = clipped

        if self.enable_step_limit:
            low = state - self.max_step_delta
            high = state + self.max_step_delta
            clipped = np.clip(action, low, high)

            # Do NOT apply per-step clipping to grippers.
            # 20D order:
            #   0:7   right_arm7
            #   7     right_gripper
            #   8:15  left_arm7
            #   15    left_gripper
            #   16:20 torso4
            #
            # Arm joints and torso are step-limited for safety.
            # Grippers are allowed to move directly to the model-predicted target.
            clipped[7] = action[7]
            clipped[15] = action[15]

            if not np.allclose(clipped, action, atol=0.0, rtol=0.0):
                report.clipped_step_delta = True
                action = clipped

        return action.astype(np.float32), report
