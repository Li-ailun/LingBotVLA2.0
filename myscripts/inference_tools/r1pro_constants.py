#!/usr/bin/env python3
# =============================================================================
# R1Pro OpenPI communication module: centralized constants.
# 功能说明：集中管理 R1Pro 通讯端使用的默认 topic、joint name、20D 维度顺序、
# 默认 prompt、控制频率、requery_steps、安全范围和固定腰部目标等常量。
# 这样可以避免 topic 和维度顺序散落在多个脚本里，便于后续统一修改。
# =============================================================================
from __future__ import annotations

import numpy as np

# ============================================================
# R1Pro 20D convention used by the current OpenPI / pi0.5 setup
# ============================================================
# 0:7    right_arm7
# 7      right_gripper1
# 8:15   left_arm7
# 15     left_gripper1
# 16:20  torso4

STATE_DIM = 20
ACTION_DIM = 20
ACTION_HORIZON_DEFAULT = 50

IDX_RIGHT_ARM = slice(0, 7)
IDX_RIGHT_GRIPPER = 7
IDX_LEFT_ARM = slice(8, 15)
IDX_LEFT_GRIPPER = 15
IDX_TORSO = slice(16, 20)

ACTION_ORDER = (
    "right_arm7, right_gripper1, left_arm7, left_gripper1, torso4"
)

# True means this dimension is trained/predicted as delta action and converted
# back to absolute action by OpenPI's AbsoluteActions transform during inference.
# False means absolute action. Grippers remain absolute.
DELTA_ACTION_MASK = np.array(
    [True] * 7 + [False] + [True] * 7 + [False] + [True] * 4,
    dtype=bool,
)

ACTION_NAMES = [
    "right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3", "right_arm_4", "right_arm_5", "right_arm_6",
    "right_gripper",
    "left_arm_0", "left_arm_1", "left_arm_2", "left_arm_3", "left_arm_4", "left_arm_5", "left_arm_6",
    "left_gripper",
    "torso_0", "torso_1", "torso_2", "torso_3",
]

# -------------------------
# ROS topics
# -------------------------
TOPIC_CAM_HIGH = "/hdas/camera_head/left_raw/image_raw_color/compressed"
TOPIC_CAM_LEFT_WRIST = "/hdas/camera_wrist_left/color/image_raw/compressed"
TOPIC_CAM_RIGHT_WRIST = "/hdas/camera_wrist_right/color/image_raw/compressed"

TOPIC_FEEDBACK_ARM_RIGHT = "/hdas/feedback_arm_right"
TOPIC_FEEDBACK_ARM_LEFT = "/hdas/feedback_arm_left"
TOPIC_FEEDBACK_TORSO = "/hdas/feedback_torso"
TOPIC_FEEDBACK_GRIPPER_RIGHT = "/hdas/feedback_gripper_right"
TOPIC_FEEDBACK_GRIPPER_LEFT = "/hdas/feedback_gripper_left"

TOPIC_TARGET_ARM_RIGHT = "/motion_target/target_joint_state_arm_right"
TOPIC_TARGET_ARM_LEFT = "/motion_target/target_joint_state_arm_left"
TOPIC_TARGET_TORSO = "/motion_target/target_joint_state_torso"
TOPIC_TARGET_GRIPPER_RIGHT = "/motion_target/target_position_gripper_right"
TOPIC_TARGET_GRIPPER_LEFT = "/motion_target/target_position_gripper_left"

RIGHT_GRIPPER_TARGET_JOINT_NAME = "R1PRO_right_gripper_joint"
LEFT_GRIPPER_TARGET_JOINT_NAME = "R1PRO_left_gripper_joint"

RIGHT_ARM_FALLBACK_NAMES = [f"R1PRO_right_arm_joint_{i}" for i in range(7)]
LEFT_ARM_FALLBACK_NAMES = [f"R1PRO_left_arm_joint_{i}" for i in range(7)]
TORSO_FALLBACK_NAMES = [f"R1PRO_torso_joint_{i}" for i in range(4)]

# -------------------------
# Defaults
# -------------------------
DEFAULT_PROMPT = "hang the wire stripper on the wall hook"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_RATE_HZ = 15.0
DEFAULT_REQUERY_STEPS = 10

# -------------------------
# Safety defaults
# -------------------------
# Absolute sanity bounds. These are intentionally broad and should be tuned
# with real robot limits before --execute.
ACTION_LOW = np.array(
    [-6.30] * 7 + [0.0] + [-6.30] * 7 + [0.0] + [-3.20, -3.20, -3.20, -3.20],
    dtype=np.float32,
)
ACTION_HIGH = np.array(
    [6.30] * 7 + [100.0] + [6.30] * 7 + [100.0] + [3.20, 3.20, 3.20, 3.20],
    dtype=np.float32,
)

# Per-control-step jump limit around current feedback state. This is the most
# important safety layer for first real-robot tests.
MAX_STEP_DELTA = np.array(
    [0.030] * 7 + [5.0] + [0.030] * 7 + [5.0] + [0.020] * 4,
    dtype=np.float32,
)

# Optional fixed torso target used by the dataset. Not forced by default; use
# --force-fixed-torso-target only when you intentionally want to hold torso.
FIXED_TORSO_TARGET = np.array([0.47, -1.12, -1.24, 0.00], dtype=np.float32)
