#!/usr/bin/env python3
# =============================================================================
# R1Pro OpenPI communication module: action/state utility functions.
# 功能说明：定义 R1Pro 20D state/action 的维度顺序、组装、拆分、校验，以及
# delta/absolute 动作转换的基础工具函数。该文件不直接连接 ROS 或 OpenPI server，
# 只负责纯 numpy 层面的数据格式处理。
# 20D 顺序：right_arm7 + right_gripper1 + left_arm7 + left_gripper1 + torso4。
# =============================================================================
from __future__ import annotations

import numpy as np

from r1pro_constants import (
    ACTION_DIM,
    ACTION_NAMES,
    DELTA_ACTION_MASK,
    IDX_LEFT_ARM,
    IDX_LEFT_GRIPPER,
    IDX_RIGHT_ARM,
    IDX_RIGHT_GRIPPER,
    IDX_TORSO,
    STATE_DIM,
)


def as_float32_1d(x, dim: int, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.shape[0] != dim:
        raise ValueError(f"{name} must have shape ({dim},), got {arr.shape}")
    return arr


def build_state20(
    right_arm7,
    right_gripper1,
    left_arm7,
    left_gripper1,
    torso4,
) -> np.ndarray:
    right_arm7 = as_float32_1d(right_arm7, 7, "right_arm7")
    right_gripper1 = as_float32_1d([right_gripper1], 1, "right_gripper1")
    left_arm7 = as_float32_1d(left_arm7, 7, "left_arm7")
    left_gripper1 = as_float32_1d([left_gripper1], 1, "left_gripper1")
    torso4 = as_float32_1d(torso4, 4, "torso4")
    return np.concatenate(
        [right_arm7, right_gripper1, left_arm7, left_gripper1, torso4],
        dtype=np.float32,
    )


def split_action20(action20) -> dict[str, np.ndarray | float]:
    action20 = as_float32_1d(action20, ACTION_DIM, "action20")
    return {
        "right_arm": action20[IDX_RIGHT_ARM],
        "right_gripper": float(action20[IDX_RIGHT_GRIPPER]),
        "left_arm": action20[IDX_LEFT_ARM],
        "left_gripper": float(action20[IDX_LEFT_GRIPPER]),
        "torso": action20[IDX_TORSO],
    }


def validate_state20(state20) -> np.ndarray:
    return as_float32_1d(state20, STATE_DIM, "state20")


def validate_action20(action20) -> np.ndarray:
    return as_float32_1d(action20, ACTION_DIM, "action20")


def validate_action_chunk(actions) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"actions must be 2D [horizon, 20], got {actions.shape}")
    if actions.shape[1] != ACTION_DIM:
        raise ValueError(f"actions last dim must be {ACTION_DIM}, got {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise ValueError("actions contain NaN or inf")
    return actions


def mock_action_chunk_from_state(state20, horizon: int = 50) -> np.ndarray:
    """Communication test only: hold current state as fake absolute targets."""
    state20 = validate_state20(state20)
    return np.repeat(state20[None, :], horizon, axis=0).astype(np.float32)


def absolute_to_delta_action(abs_action20, state20) -> np.ndarray:
    """Same convention as OpenPI DeltaActions for R1Pro 20D."""
    abs_action20 = validate_action20(abs_action20)
    state20 = validate_state20(state20)
    out = abs_action20.copy()
    out[DELTA_ACTION_MASK] = abs_action20[DELTA_ACTION_MASK] - state20[DELTA_ACTION_MASK]
    out[~DELTA_ACTION_MASK] = abs_action20[~DELTA_ACTION_MASK]
    return out


def delta_to_absolute_action(delta_or_abs_action20, state20) -> np.ndarray:
    """Same convention as OpenPI AbsoluteActions for R1Pro 20D."""
    delta_or_abs_action20 = validate_action20(delta_or_abs_action20)
    state20 = validate_state20(state20)
    out = delta_or_abs_action20.copy()
    out[DELTA_ACTION_MASK] = delta_or_abs_action20[DELTA_ACTION_MASK] + state20[DELTA_ACTION_MASK]
    out[~DELTA_ACTION_MASK] = delta_or_abs_action20[~DELTA_ACTION_MASK]
    return out


def compact_action_summary(action20) -> str:
    p = split_action20(action20)
    return (
        f"right_arm[0]={p['right_arm'][0]:.4f}, "
        f"right_gripper={p['right_gripper']:.4f}, "
        f"left_arm[0]={p['left_arm'][0]:.4f}, "
        f"left_gripper={p['left_gripper']:.4f}, "
        f"torso={np.array2string(p['torso'], precision=3)}"
    )


def action_table(action20, state20=None) -> str:
    action20 = validate_action20(action20)
    state20 = None if state20 is None else validate_state20(state20)
    lines = []
    if state20 is None:
        lines.append("index | name              | action")
        lines.append("-" * 52)
        for i, name in enumerate(ACTION_NAMES):
            lines.append(f"{i:>5} | {name:<17} | {action20[i]:>10.4f}")
    else:
        lines.append("index | name              | state      | action     | action-state")
        lines.append("-" * 76)
        for i, name in enumerate(ACTION_NAMES):
            lines.append(
                f"{i:>5} | {name:<17} | {state20[i]:>10.4f} | "
                f"{action20[i]:>10.4f} | {(action20[i] - state20[i]):>12.4f}"
            )
    return "\n".join(lines)
