"""Canonical R1Pro dual-arm 16D schema.

The ordering in this module is the only supported ordering for the 16D
toolchain: left arm, left gripper, right arm, right gripper.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

SCHEMA_NAME = "r1pro_dual_arm_16d_left_first_v1"
VECTOR_DIM = 16

STATE_NAMES = (
    "left_arm_j0", "left_arm_j1", "left_arm_j2", "left_arm_j3",
    "left_arm_j4", "left_arm_j5", "left_arm_j6", "left_gripper",
    "right_arm_j0", "right_arm_j1", "right_arm_j2", "right_arm_j3",
    "right_arm_j4", "right_arm_j5", "right_arm_j6", "right_gripper",
)
ACTION_NAMES = tuple(f"target_{name}" for name in STATE_NAMES)

GROUPS: Mapping[str, tuple[int, ...]] = {
    "left_arm": tuple(range(0, 7)),
    "left_gripper": (7,),
    "right_arm": tuple(range(8, 15)),
    "right_gripper": (15,),
}

TOPICS: Mapping[str, str] = {
    "cam_high": "/hdas/camera_head/left_raw/image_raw_color/compressed",
    "cam_left_wrist": "/hdas/camera_wrist_left/color/image_raw/compressed",
    "cam_right_wrist": "/hdas/camera_wrist_right/color/image_raw/compressed",
    "state_left_arm": "/hdas/feedback_arm_left",
    "state_left_gripper": "/hdas/feedback_gripper_left",
    "state_right_arm": "/hdas/feedback_arm_right",
    "state_right_gripper": "/hdas/feedback_gripper_right",
    "action_left_arm": "/motion_target/target_joint_state_arm_left",
    "action_left_gripper": "/motion_target/target_position_gripper_left",
    "action_right_arm": "/motion_target/target_joint_state_arm_right",
    "action_right_gripper": "/motion_target/target_position_gripper_right",
}

CAM_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
STATE_KEYS = ("state_left_arm", "state_left_gripper", "state_right_arm", "state_right_gripper")
ACTION_KEYS = ("action_left_arm", "action_left_gripper", "action_right_arm", "action_right_gripper")
CORE_TOPIC_KEYS = CAM_KEYS + STATE_KEYS + ACTION_KEYS


def _component(value: Sequence[float] | np.ndarray, *, name: str, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != size:
        raise ValueError(f"{name}: expected {size} values, got {array.size}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: contains non-finite values")
    return array


def _assemble(
    *,
    left_arm: Sequence[float] | np.ndarray,
    left_gripper: Sequence[float] | np.ndarray,
    right_arm: Sequence[float] | np.ndarray,
    right_gripper: Sequence[float] | np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        (
            _component(left_arm, name="left_arm", size=7),
            _component(left_gripper, name="left_gripper", size=1),
            _component(right_arm, name="right_arm", size=7),
            _component(right_gripper, name="right_gripper", size=1),
        )
    ).astype(np.float32, copy=False)


def assemble_state(**components: Sequence[float] | np.ndarray) -> np.ndarray:
    """Assemble one left-first 16D observation state."""
    return _assemble(**components)


def assemble_action(**components: Sequence[float] | np.ndarray) -> np.ndarray:
    """Assemble one left-first 16D target action."""
    return _assemble(**components)


def validate_vector_array(
    value: np.ndarray,
    *,
    name: str,
    episode: str | int | None = None,
) -> np.ndarray:
    """Validate an array whose final axis must use this exact 16D schema."""
    array = np.asarray(value)
    actual = array.shape[-1] if array.ndim else 0
    context = f" episode={episode}" if episode is not None else ""
    if array.ndim == 0 or actual != VECTOR_DIM:
        raise ValueError(f"{name}{context}: expected final dimension 16, got {actual}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name}{context}: contains non-finite values")
    return array
