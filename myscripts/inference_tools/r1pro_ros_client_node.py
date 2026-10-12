#!/usr/bin/env python3
# =============================================================================
# R1Pro OpenPI communication module: ROS2 client node.
# 功能说明：真实部署/回放测试的主通讯节点。负责从 ROS2 订阅三路图像、左右臂、
# 左右夹爪和腰部反馈，组装 OpenPI 所需 obs = images + state20 + prompt；
# 再通过 websocket 调用本地 OpenPI policy server 获取 actions [horizon, 20]。
# 收到 action20 后，本节点只做安全检查、限幅、拆分和发布；不要在这里手动 action + state，
# 因为 OpenPI server 的 AbsoluteActions 已经把 delta 输出恢复成 absolute action。
# 支持 hold-current/mock、dry-run、execute、show-images、record-debug 和安全限幅。
# =============================================================================
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Float64

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from r1pro_action_utils import (  # noqa: E402
    action_table,
    build_state20,
    compact_action_summary,
    mock_action_chunk_from_state,
    split_action20,
    validate_action_chunk,
)
from r1pro_constants import (  # noqa: E402
    ACTION_HORIZON_DEFAULT,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_PROMPT,
    DEFAULT_RATE_HZ,
    DEFAULT_REQUERY_STEPS,
    DELTA_ACTION_MASK,
    FIXED_TORSO_TARGET,
    LEFT_ARM_FALLBACK_NAMES,
    LEFT_GRIPPER_TARGET_JOINT_NAME,
    RIGHT_ARM_FALLBACK_NAMES,
    RIGHT_GRIPPER_TARGET_JOINT_NAME,
    TOPIC_CAM_HIGH,
    TOPIC_CAM_LEFT_WRIST,
    TOPIC_CAM_RIGHT_WRIST,
    TOPIC_FEEDBACK_ARM_LEFT,
    TOPIC_FEEDBACK_ARM_RIGHT,
    TOPIC_FEEDBACK_GRIPPER_LEFT,
    TOPIC_FEEDBACK_GRIPPER_RIGHT,
    TOPIC_FEEDBACK_TORSO,
    TOPIC_TARGET_ARM_LEFT,
    TOPIC_TARGET_ARM_RIGHT,
    TOPIC_TARGET_GRIPPER_LEFT,
    TOPIC_TARGET_GRIPPER_RIGHT,
    TOPIC_TARGET_TORSO,
    TORSO_FALLBACK_NAMES,
)
from r1pro_safety import ActionSafetyLimiter  # noqa: E402

try:
    from openpi_client import websocket_client_policy
except Exception as e:  # pragma: no cover - depends on local env
    websocket_client_policy = None
    OPENPI_CLIENT_IMPORT_ERROR = e
else:
    OPENPI_CLIENT_IMPORT_ERROR = None


class JsonlWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("a", encoding="utf-8")

    def write(self, obj: dict):
        self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


def _now_ns() -> int:
    return time.time_ns()


ANSI_BLUE = "\033[94m"
ANSI_YELLOW = "\033[93m"
ANSI_RESET = "\033[0m"


class R1ProRosOpenPIClient(Node):
    def __init__(self, args):
        super().__init__("r1pro_openpi_client")
        self.args = args
        self.execute = bool(args.execute)
        self.prompt = args.prompt
        self.policy_mode = "hold_current" if args.mock_policy else args.policy_mode

        self.policy = None
        if self.policy_mode == "server":
            if websocket_client_policy is None:
                raise RuntimeError(
                    "Failed to import the bundled websocket client. Use system Python for ROS, and set:\n"
                    f"export PYTHONPATH={THIS_DIR}:$PYTHONPATH"
                ) from OPENPI_CLIENT_IMPORT_ERROR
            self.get_logger().info(f"Connecting to OpenPI server ws://{args.host}:{args.port}")
            self.policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
            self.get_logger().info(f"Connected. Server metadata: {self.policy.get_server_metadata()}")
        elif self.policy_mode == "hold_current":
            self.get_logger().warn("Using hold-current/mock policy. No OpenPI server will be queried.")
        else:
            raise ValueError(f"Unsupported policy_mode: {self.policy_mode}")

        self.safety = ActionSafetyLimiter(
            enable_abs_clip=not args.disable_abs_clip,
            enable_step_limit=args.enable_step_limit and not args.disable_step_limit,
            force_fixed_torso_target=args.force_fixed_torso_target,
        )

        self.cam_high = None
        self.cam_left_wrist = None
        self.cam_right_wrist = None
        self.right_arm = None
        self.left_arm = None
        self.torso = None
        self.right_gripper = None
        self.left_gripper = None
        self.right_arm_names = []
        self.left_arm_names = []
        self.torso_names = []

        self.action_chunk_raw = None
        self.action_index = 0
        self.last_safe_action20 = None
        self.query_count = 0
        self.published_steps = 0
        self.start_time = time.monotonic()
        self.last_status_time = 0.0
        self.last_table_time = 0.0
        self.last_async_status_time = 0.0
        self.show_images_failed = False

        self.async_lock = threading.Lock()
        self.async_inflight = False
        self.async_next_chunk = None
        self.async_next_info = None
        self.async_last_error = None
        self.async_started_count = 0
        self.async_completed_count = 0
        self.async_switched_count = 0
        self.async_missed_requery_count = 0
        self.async_failure_count = 0
        self.async_next_retry_time = 0.0
        self.async_enabled = self.policy_mode == "server" and not args.sync_infer
        self.rtc_enabled = self.async_enabled and not args.disable_rtc
        self.rtc_latency_max_ms = (
            max(0, int(args.rtc_initial_inference_delay_steps)) / float(args.rate_hz) * 1000.0
        )
        self.record_lock = threading.Lock()
        # Do not disable legacy smoothing until the server proves it is running
        # RTC-capable policy code.  An old server will silently ignore obs["rtc"],
        # and disabling smoothing in that state makes real-robot motion worse.
        self.rtc_server_confirmed = False
        self.rtc_server_warning_printed = False

        # EXP6-A ------------------------------------------------------------
        #
        # VR release:
        #     await_initial
        #
        # fresh initial chunk promoted:
        #     skip_next_prefetch
        #
        # first post-VR NON-RTC replan promoted:
        #     await_first_rtc_prefetch
        #
        # first RTC replan promoted:
        #     idle
        #
        # The global rtc_enabled flag NEVER changes.  Only the RTC payload for
        # one post-VR prefetch request is suppressed.  This is important
        # because baseline queue-delay crop must remain active.
        self.exp6a_post_vr_rtc_state = "idle"
        self.exp6a_arm_count = 0
        self.exp6a_skip_request_count = 0
        self.exp6a_completed_count = 0

        # request_id -> whether that request actually carried obs["rtc"]
        self._exp6a_query_rtc_sent = {}
        # ------------------------------------------------------------------

        self.client_smoothing_allowed = True
        self.smoothing_enabled = (not args.disable_action_smoothing) and self.client_smoothing_allowed
        self.smooth_transition_steps = max(0, int(args.smooth_transition_steps))
        self.smooth_mask = DELTA_ACTION_MASK.copy()
        self.publish_smoothing_enabled = (not args.disable_publish_smoothing) and self.client_smoothing_allowed
        self.publish_smooth_mask = DELTA_ACTION_MASK.copy()
        self.publish_smooth_alpha = float(np.clip(args.publish_smooth_alpha, 0.0, 1.0))

        # GRIPPER-DEBOUNCE7 ---------------------------------------------------
        # Gripper stabilization is deliberately independent from arm/torso
        # SmoothStep + EMA.  Default OFF preserves the sealed formal baseline.
        self.gripper_stabilizer_enabled = bool(args.enable_gripper_stabilizer)
        self.gripper_debounce_steps = max(1, int(args.gripper_debounce_steps))
        self.gripper_reversal_lock_steps = max(
            0,
            int(args.gripper_reversal_lock_steps),
        )
        self.gripper_low_threshold = float(args.gripper_low_threshold)
        self.gripper_high_threshold = float(args.gripper_high_threshold)

        if self.gripper_low_threshold >= self.gripper_high_threshold:
            raise ValueError(
                "gripper-low-threshold must be smaller than "
                "gripper-high-threshold"
            )

        # 0 = LOW, 1 = HIGH, None = not initialized yet.
        # Right/left grippers are completely independent.
        self._gripper_latched_state = {
            "right": None,
            "left": None,
        }
        self._gripper_pending_state = {
            "right": None,
            "left": None,
        }
        self._gripper_pending_count = {
            "right": 0,
            "left": 0,
        }

        # Number of future POLICY frames during which an opposite
        # gripper state is forbidden from accumulating debounce count.
        self._gripper_lock_remaining = {
            "right": 0,
            "left": 0,
        }

        self._gripper_hold_value = {
            "right": None,
            "left": None,
        }

        # Used to detect episode/intervention pauses. Normal policy execution
        # is 15 Hz, so a >1 s gap is an unambiguous discontinuity.
        self._gripper_last_policy_publish_time = None

        # When VR hands control back to POLICY, the stabilizer is explicitly
        # re-anchored from physical gripper feedback. The first policy frame
        # must not immediately treat that re-anchor as a generic long-gap
        # reset and erase the freshly installed reversal lock.
        self._gripper_skip_gap_reset_once = False
        # --------------------------------------------------------------------

        self.record_writer = None
        self.record_dir = None
        if args.record_debug:
            self.record_dir = Path(args.record_dir).expanduser().resolve()
            self.record_dir.mkdir(parents=True, exist_ok=True)
            (self.record_dir / "images").mkdir(exist_ok=True)
            self.record_writer = JsonlWriter(self.record_dir / "records.jsonl")
            self.get_logger().info(f"Debug recording enabled: {self.record_dir}")

        self._create_ros_io()

        if self.gripper_stabilizer_enabled:
            self.get_logger().info(
                "[gripper-stabilizer] ENABLED: "
                f"debounce_steps={self.gripper_debounce_steps}, "
                f"reversal_lock_steps={self.gripper_reversal_lock_steps}, "
                f"low_threshold={self.gripper_low_threshold:.3f}, "
                f"high_threshold={self.gripper_high_threshold:.3f}, "
                "dims=right_gripper(7)+left_gripper(15)"
            )

        self.timer = self.create_timer(1.0 / float(args.rate_hz), self._control_timer)

        self.get_logger().info("R1Pro ROS OpenPI client initialized.")
        self.get_logger().info(
            f"policy_mode={self.policy_mode}, execute={self.execute}, "
            f"rate_hz={args.rate_hz}, requery_steps={args.requery_steps}, "
            f"infer_mode={'async' if self.async_enabled else 'sync'}, "
            f"async_prefetch_steps={args.async_prefetch_steps}, "
            f"rtc_enabled={self.rtc_enabled}, "
            f"rtc_execution_horizon={args.rtc_execution_horizon}, "
            f"rtc_max_guidance_weight={args.rtc_max_guidance_weight:.3f}, "
            f"rtc_schedule={args.rtc_prefix_attention_schedule}, "
            f"rtc_initial_delay_steps={args.rtc_initial_inference_delay_steps}, "
            f"rtc_max_delay_steps={args.rtc_max_inference_delay_steps}, "
            f"action_smoothing={self.smoothing_enabled}, "
            f"smooth_transition_steps={self.smooth_transition_steps}, "
            f"publish_smoothing={self.publish_smoothing_enabled}, "
            f"publish_smoothing_dims=arms+torso, "
            f"publish_smooth_alpha={self.publish_smooth_alpha:.3f}, "
            f"step_limit={self.safety.enable_step_limit}"
        )
        if not self.execute:
            self.get_logger().warn("Dry-run mode: actions will be printed but NOT published. Add --execute to publish.")
        else:
            self.get_logger().error("EXECUTE MODE ENABLED: robot target commands will be published.")

    def _create_ros_io(self):
        self.create_subscription(CompressedImage, self.args.topic_cam_high, lambda m: self._compressed_image_cb(m, "cam_high"), qos_profile_sensor_data)
        self.create_subscription(CompressedImage, self.args.topic_cam_left_wrist, lambda m: self._compressed_image_cb(m, "cam_left_wrist"), qos_profile_sensor_data)
        self.create_subscription(CompressedImage, self.args.topic_cam_right_wrist, lambda m: self._compressed_image_cb(m, "cam_right_wrist"), qos_profile_sensor_data)

        self.create_subscription(JointState, self.args.topic_feedback_arm_right, self._right_arm_cb, qos_profile_sensor_data)
        self.create_subscription(JointState, self.args.topic_feedback_arm_left, self._left_arm_cb, qos_profile_sensor_data)
        self.create_subscription(JointState, self.args.topic_feedback_torso, self._torso_cb, qos_profile_sensor_data)

        if self.args.gripper_feedback_type == "joint_state":
            self.create_subscription(JointState, self.args.topic_feedback_gripper_right, lambda m: self._set_gripper_from_joint_state("right", m), qos_profile_sensor_data)
            self.create_subscription(JointState, self.args.topic_feedback_gripper_left, lambda m: self._set_gripper_from_joint_state("left", m), qos_profile_sensor_data)
        elif self.args.gripper_feedback_type == "float64":
            self.create_subscription(Float64, self.args.topic_feedback_gripper_right, lambda m: self._set_gripper("right", float(m.data)), qos_profile_sensor_data)
            self.create_subscription(Float64, self.args.topic_feedback_gripper_left, lambda m: self._set_gripper("left", float(m.data)), qos_profile_sensor_data)
        else:
            raise ValueError(f"Unsupported gripper_feedback_type: {self.args.gripper_feedback_type}")

        self.pub_right_arm = self.create_publisher(JointState, self.args.topic_target_arm_right, 10)
        self.pub_left_arm = self.create_publisher(JointState, self.args.topic_target_arm_left, 10)
        self.pub_torso = self.create_publisher(JointState, self.args.topic_target_torso, 10)

        if self.args.gripper_target_type == "joint_state":
            self.pub_right_gripper = self.create_publisher(JointState, self.args.topic_target_gripper_right, 10)
            self.pub_left_gripper = self.create_publisher(JointState, self.args.topic_target_gripper_left, 10)
        elif self.args.gripper_target_type == "float64":
            self.pub_right_gripper = self.create_publisher(Float64, self.args.topic_target_gripper_right, 10)
            self.pub_left_gripper = self.create_publisher(Float64, self.args.topic_target_gripper_left, 10)
        else:
            raise ValueError(f"Unsupported gripper_target_type: {self.args.gripper_target_type}")

    def _compressed_image_cb(self, msg: CompressedImage, key: str):
        np_buf = np.frombuffer(msg.data, dtype=np.uint8)
        bgr = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
        if bgr is None:
            self.get_logger().warn(f"Failed to decode compressed image: {key}")
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if key == "cam_high":
            self.cam_high = rgb
        elif key == "cam_left_wrist":
            self.cam_left_wrist = rgb
        elif key == "cam_right_wrist":
            self.cam_right_wrist = rgb
        else:
            raise ValueError(key)

    def _right_arm_cb(self, msg: JointState):
        arr = np.asarray(msg.position, dtype=np.float32).reshape(-1)
        if arr.shape[0] >= 7:
            self.right_arm = arr[:7]
            self.right_arm_names = list(msg.name[:7]) if len(msg.name) >= 7 else []

    def _left_arm_cb(self, msg: JointState):
        arr = np.asarray(msg.position, dtype=np.float32).reshape(-1)
        if arr.shape[0] >= 7:
            self.left_arm = arr[:7]
            self.left_arm_names = list(msg.name[:7]) if len(msg.name) >= 7 else []

    def _torso_cb(self, msg: JointState):
        arr = np.asarray(msg.position, dtype=np.float32).reshape(-1)
        if arr.shape[0] >= 4:
            self.torso = arr[:4]
            self.torso_names = list(msg.name[:4]) if len(msg.name) >= 4 else []

    def _set_gripper(self, side: str, value: float):
        if side == "right":
            self.right_gripper = float(value)
        elif side == "left":
            self.left_gripper = float(value)
        else:
            raise ValueError(side)

    def _set_gripper_from_joint_state(self, side: str, msg: JointState):
        if len(msg.position) < 1:
            return
        self._set_gripper(side, float(msg.position[0]))

    def _ready(self) -> bool:
        return all(x is not None for x in [
            self.cam_high, self.cam_left_wrist, self.cam_right_wrist,
            self.right_arm, self.right_gripper, self.left_arm, self.left_gripper, self.torso,
        ])

    def _build_state20(self) -> np.ndarray:
        return build_state20(self.right_arm, self.right_gripper, self.left_arm, self.left_gripper, self.torso)

    def _build_obs(self) -> dict:
        return {
            "images": {
                "cam_high": self.cam_high,
                "cam_left_wrist": self.cam_left_wrist,
                "cam_right_wrist": self.cam_right_wrist,
            },
            "state": self._build_state20().astype(np.float32),
            "prompt": self.prompt,
        }

    def _copy_obs(self, obs: dict) -> dict:
        copied = {
            "images": {k: np.asarray(v).copy() for k, v in obs["images"].items()},
            "state": np.asarray(obs["state"], dtype=np.float32).copy(),
            "prompt": str(obs["prompt"]),
        }
        if "rtc" in obs:
            rtc = dict(obs["rtc"])
            prev = rtc.get("prev_chunk_left_over", None)
            if prev is not None:
                rtc["prev_chunk_left_over"] = np.asarray(prev, dtype=np.float32).copy()
            copied["rtc"] = rtc
        return copied

    def _rtc_schedule_id(self) -> int:
        # Matches openpi.models.pi0 RTC_SCHEDULE_* constants:
        # linear=0, exp=1, ones=2, zeros=3.
        return {
            "linear": 0,
            "exp": 1,
            "ones": 2,
            "zeros": 3,
        }[self.args.rtc_prefix_attention_schedule]

    def _estimate_rtc_inference_delay_steps(
        self,
        minimum_delay_steps: int = 0,
    ) -> int:
        step_ms = 1000.0 / max(float(self.args.rate_hz), 1e-6)

        # RTC must account for both:
        #   1. actions elapsed while inference is running, and
        #   2. actions that will elapse before the client is allowed to
        #      promote the new chunk at the planned requery boundary.
        latency_delay = int(
            np.ceil(max(0.0, self.rtc_latency_max_ms) / step_ms)
        )
        delay = max(latency_delay, max(0, int(minimum_delay_steps)))

        max_delay = max(0, int(self.args.rtc_max_inference_delay_steps))
        if max_delay > 0:
            delay = min(delay, max_delay)

        return max(0, min(delay, ACTION_HORIZON_DEFAULT))

    def _obs_with_rtc(self, obs: dict, reason: str) -> dict:
        obs_copy = self._copy_obs(obs)

        # EXP6-A:
        # After a real VR release and a fresh INITIAL chunk, suppress RTC
        # guidance for the FIRST prefetch only.
        #
        # The query still happens at async_prefetch_steps=6.  Only obs["rtc"]
        # is omitted, so the server performs an ordinary fresh inference.
        if (
            self.exp6a_post_vr_rtc_state == "skip_next_prefetch"
            and reason.startswith("prefetch_at_")
        ):
            self.exp6a_skip_request_count += 1

            self.get_logger().info(
                "[EXP6-A] SKIP RTC for first post-VR prefetch: "
                f"skip_request_count={self.exp6a_skip_request_count}, "
                f"reason={reason}, "
                f"action_index={self.action_index}; "
                "request remains async and baseline queue-delay crop remains enabled"
            )

            return obs_copy

        if not self.rtc_enabled:
            return obs_copy

        prev_left_over = None
        prefix_len = 0
        if self.action_chunk_raw is not None:
            chunk = validate_action_chunk(self.action_chunk_raw)
            start = int(np.clip(self.action_index, 0, len(chunk)))
            prev_left_over = chunk[start:].astype(np.float32, copy=True)
            prefix_len = int(prev_left_over.shape[0])
            if prefix_len == 0:
                prev_left_over = None

        planned_handoff_delay = 0
        if (
            reason.startswith("prefetch_at_")
            and self.action_chunk_raw is not None
        ):
            # Async chunks are not promoted immediately when inference
            # finishes. They are promoted at requery_steps, so RTC must align
            # to that effective handoff point rather than latency alone.
            planned_handoff_delay = max(
                0,
                int(self.args.requery_steps) - int(self.action_index),
            )

        inference_delay = self._estimate_rtc_inference_delay_steps(
            minimum_delay_steps=planned_handoff_delay,
        )
        obs_copy["rtc"] = {
            "enabled": True,
            "reason": reason,
            "prev_chunk_left_over": prev_left_over,
            "prev_chunk_left_over_length": prefix_len,
            "inference_delay": inference_delay,
            "execution_horizon": int(self.args.rtc_execution_horizon),
            "max_guidance_weight": float(self.args.rtc_max_guidance_weight),
            "prefix_attention_schedule": self._rtc_schedule_id(),
        }
        return obs_copy

    def _need_new_chunk(self) -> bool:
        if self.action_chunk_raw is None:
            return True
        if self.action_index >= len(self.action_chunk_raw):
            return True
        if self.action_index >= self.args.requery_steps:
            return True
        return False

    def _query_policy(self, obs: dict) -> np.ndarray:
        self.query_count += 1
        if self.policy_mode == "hold_current":
            actions = mock_action_chunk_from_state(obs["state"], horizon=ACTION_HORIZON_DEFAULT)
            self.get_logger().info("Hold-current/mock action chunk generated from current state.")
            result = {"actions": actions}
        else:
            t0 = time.monotonic()
            result = self.policy.infer(obs)
            dt_ms = (time.monotonic() - t0) * 1000.0
            actions = validate_action_chunk(result["actions"])
            timing = {"client_roundtrip_ms": dt_ms, "policy_timing": result.get("policy_timing", {}), "server_timing": result.get("server_timing", {})}
            self._update_rtc_server_status(obs, result)
            self.get_logger().info(f"Received action chunk: shape={actions.shape}, timing={timing}")

        actions = validate_action_chunk(actions)
        if self.record_writer is not None:
            self._record_query(obs, result, actions)
        return actions

    def _update_rtc_server_status(self, obs: dict, result: dict):
        if not self.rtc_enabled or "rtc" not in obs:
            return

        policy_timing = result.get("policy_timing", {}) or {}
        confirmed = bool(policy_timing.get("rtc_enabled", False))
        if confirmed:
            if not self.rtc_server_confirmed:
                self.rtc_server_confirmed = True
                if not self.args.enable_client_smoothing:
                    self.smoothing_enabled = False
                    self.publish_smoothing_enabled = False
                self.get_logger().info(
                    f"[rtc] server confirmed: "
                    f"prefix_len={policy_timing.get('rtc_prefix_len')}, "
                    f"inference_delay={policy_timing.get('rtc_inference_delay')}, "
                    f"execution_horizon={policy_timing.get('rtc_execution_horizon')}; "
                    f"client_smoothing={self.smoothing_enabled}, "
                    f"publish_smoothing={self.publish_smoothing_enabled}"
                )
            return

        if not self.rtc_server_warning_printed:
            self.rtc_server_warning_printed = True
            self.get_logger().warn(
                f"{ANSI_YELLOW}[rtc] client is sending RTC fields, but the server response "
                f"does not contain policy_timing.rtc_enabled=True. "
                f"The policy server is probably still running old code or was not restarted. "
                f"Keeping client smoothing enabled for safety.{ANSI_RESET}"
            )

    def _record_query(self, obs: dict, result: dict, actions: np.ndarray):
        qid = self.query_count
        npz_path = self.record_dir / f"query_{qid:06d}.npz"
        with self.record_lock:
            np.savez_compressed(npz_path, state=obs["state"], actions=actions)
            img_paths = {}
            if self.args.record_images:
                for k, rgb in obs["images"].items():
                    img_path = self.record_dir / "images" / f"query_{qid:06d}_{k}.jpg"
                    cv2.imwrite(str(img_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    img_paths[k] = str(img_path)
            self.record_writer.write({
                "type": "policy_query",
                "time_ns": _now_ns(),
                "query_id": qid,
                "prompt": self.prompt,
                "rtc": self._json_safe_rtc(obs.get("rtc")),
                "state20": obs["state"].astype(float).tolist(),
                "actions_shape": list(actions.shape),
                "actions_npz": str(npz_path),
                "image_paths": img_paths,
                "policy_timing": result.get("policy_timing", {}),
                "server_timing": result.get("server_timing", {}),
            })

    def _record_step(self, raw_action, safe_action, safety_summary: str):
        if self.record_writer is None:
            return
        with self.record_lock:
            self.record_writer.write({
                "type": "control_step",
                "time_ns": _now_ns(),
                "step": self.published_steps,
                "execute": self.execute,
                "action_index": self.action_index,
                "raw_action20": np.asarray(raw_action, dtype=float).tolist(),
                "safe_action20": np.asarray(safe_action, dtype=float).tolist(),
                "safety": safety_summary,
            })

    def _maybe_start_async_query(self, obs: dict, reason: str) -> bool:
        if not self.async_enabled:
            return False
        now = time.monotonic()
        with self.async_lock:
            if now < self.async_next_retry_time:
                return False
            if self.async_inflight or self.async_next_chunk is not None:
                return False
            self.async_inflight = True
            self.async_started_count += 1
            request_id = self.async_started_count
            action_index_at_request = int(self.action_index)
        obs_copy = self._obs_with_rtc(obs, reason)

        rtc_request_sent = bool(
            "rtc" in obs_copy
            and isinstance(obs_copy.get("rtc"), dict)
            and bool(obs_copy["rtc"].get("enabled", False))
        )

        self._exp6a_query_rtc_sent[int(request_id)] = rtc_request_sent

        if (
            self.exp6a_post_vr_rtc_state == "await_first_rtc_prefetch"
            and reason.startswith("prefetch_at_")
            and rtc_request_sent
        ):
            self.get_logger().info(
                "[EXP6-A] FIRST RTC query after skipped non-RTC cycle: "
                f"reason={reason}, "
                f"action_index_at_request={action_index_at_request}"
            )

        worker = threading.Thread(
            target=self._async_query_worker,
            args=(request_id, obs_copy, reason, action_index_at_request),
            daemon=True,
        )
        worker.start()
        self.get_logger().info(
            f"[async] started policy query #{request_id}: "
            f"reason={reason}, action_index_at_request={action_index_at_request}"
        )
        return True

    def _async_query_worker(self, request_id: int, obs: dict, reason: str, action_index_at_request: int):
        t0 = time.monotonic()
        try:
            actions = self._query_policy(obs)
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            info = {
                "request_id": request_id,
                "reason": reason,
                "elapsed_ms": elapsed_ms,
                "ready_time": time.monotonic(),
                "action_index_at_request": action_index_at_request,
            }
            with self.async_lock:
                self.async_next_chunk = actions
                self.async_next_info = info
                self.async_inflight = False
                self.async_completed_count += 1
                self.async_failure_count = 0
                self.async_next_retry_time = 0.0
                self.async_last_error = None
                # The first request after a model/server code change often pays
                # a large JIT compile cost.  RTC inference_delay should reflect
                # steady-state network+inference latency, not that compile
                # outlier.  Non-initial successful calls update the max tracker.
                if reason != "initial":
                    self.rtc_latency_max_ms = max(self.rtc_latency_max_ms, elapsed_ms)
            self.get_logger().info(
                f"[async] policy query #{request_id} ready: "
                f"shape={actions.shape}, elapsed_ms={elapsed_ms:.1f}, reason={reason}"
            )
        except Exception as e:
            with self.async_lock:
                self.async_inflight = False
                self.async_last_error = repr(e)
                self.async_failure_count += 1
                backoff_s = min(2.0, 0.25 * (2 ** min(self.async_failure_count - 1, 3)))
                self.async_next_retry_time = time.monotonic() + backoff_s
            self.get_logger().error(
                f"[async] policy query #{request_id} failed: {e} "
                f"(retry_backoff={backoff_s:.2f}s, consecutive_failures={self.async_failure_count})"
            )

    def _promote_async_chunk_if_ready(self, reason: str) -> bool:
        old_action_index = self.action_index
        old_chunk = self.action_chunk_raw
        with self.async_lock:
            if self.async_next_chunk is None:
                return False
            chunk = self.async_next_chunk
            info = self.async_next_info or {}
            self.async_next_chunk = None
            self.async_next_info = None
            self.async_switched_count += 1
            switch_count = self.async_switched_count
        new_chunk = self._apply_rtc_queue_delay(
            new_chunk=validate_action_chunk(chunk),
            old_action_index=old_action_index,
            info=info,
            reason=reason,
        )
        self.action_chunk_raw = self._smooth_chunk_transition(
            old_chunk=old_chunk,
            old_action_index=old_action_index,
            new_chunk=new_chunk,
            reason=reason,
        )
        self.action_index = 0

        self._exp6a_note_promoted_chunk(
            reason=reason,
            info=info,
        )

        msg = (
            f"[async] switched to ready chunk #{switch_count}: "
            f"reason={reason}, source_query={info.get('request_id')}, "
            f"old_action_index={old_action_index}, "
            f"action_index_at_request={info.get('action_index_at_request')}, "
            f"query_elapsed_ms={info.get('elapsed_ms', -1):.1f}"
        )
        if reason == "requery_steps" and old_action_index <= self.args.requery_steps:
            msg = f"{ANSI_BLUE}{msg}{ANSI_RESET}"
        self.get_logger().info(msg)
        return True

    def _apply_rtc_queue_delay(
        self,
        *,
        new_chunk: np.ndarray,
        old_action_index: int,
        info: dict,
        reason: str,
    ) -> np.ndarray:
        """Match LeRobot RTC ActionQueue.merge(...): discard actions elapsed during inference."""

        if not self.rtc_enabled or old_action_index <= 0 or reason == "initial":
            return new_chunk

        action_index_at_request = info.get("action_index_at_request")
        if action_index_at_request is None:
            return new_chunk

        real_delay = max(0, int(old_action_index) - int(action_index_at_request))
        clamped_delay = max(0, min(real_delay, len(new_chunk) - 1))
        if clamped_delay <= 0:
            return new_chunk

        cropped = new_chunk[clamped_delay:].copy()
        self.get_logger().info(
            f"[rtc] action queue delay crop: real_delay={real_delay}, "
            f"clamped_delay={clamped_delay}, action_index_at_request={action_index_at_request}, "
            f"old_action_index={old_action_index}, new_len={len(cropped)}"
        )
        return cropped

    # ------------------------------------------------------------------
    # EXP6-A: skip first post-VR RTC request
    # ------------------------------------------------------------------

    def _exp6a_arm_post_vr_skip_first_rtc(self) -> None:
        """
        Arm EXP6-A after a real VR -> POLICY handback.

        The fresh INITIAL policy chunk is still generated normally.
        Its first prefetch will be ordinary non-RTC inference.
        """
        self.exp6a_post_vr_rtc_state = "await_initial"
        self.exp6a_arm_count += 1
        self._exp6a_query_rtc_sent.clear()

        # GRIPPER-DEBOUNCE7:
        # VR commands bypass this policy-side stabilizer completely.
        # On VR -> POLICY handback, discard any pre-intervention candidate
        # history and re-anchor the latch to current physical feedback.
        self._reset_gripper_stabilizer_from_feedback(
            reason="post_vr_handback"
        )

        self.get_logger().info(
            "[EXP6-A] armed after VR release: "
            "waiting for fresh initial chunk; "
            f"arm_count={self.exp6a_arm_count}"
        )

        if not self.rtc_server_confirmed:
            self.get_logger().warning(
                "[EXP6-A] armed before RTC server confirmation. "
                "For the cleanest experiment, VR intervention should occur "
                "after normal baseline RTC has already been confirmed."
            )

    def _exp6a_disarm_post_vr_skip_first_rtc(
        self,
        reason: str,
    ) -> None:
        previous = self.exp6a_post_vr_rtc_state
        self.exp6a_post_vr_rtc_state = "idle"
        self._exp6a_query_rtc_sent.clear()

        if previous != "idle":
            self.get_logger().info(
                "[EXP6-A] disarmed: "
                f"previous_state={previous}, reason={reason}"
            )

    def _exp6a_note_promoted_chunk(
        self,
        *,
        reason: str,
        info: dict,
    ) -> None:
        """
        Advance EXP6-A only when the corresponding query is actually promoted.

        This deliberately avoids consuming the one-shot state merely because a
        query was started.  If a query fails and is retried, the first successful
        post-VR prefetch is still the one that is executed without RTC.
        """
        request_id = info.get("request_id")

        rtc_sent = None

        if request_id is not None:
            rtc_sent = self._exp6a_query_rtc_sent.pop(
                int(request_id),
                None,
            )

        state = self.exp6a_post_vr_rtc_state

        # Fresh INITIAL chunk after VR handback.
        if reason == "initial":
            if state == "await_initial":
                self.exp6a_post_vr_rtc_state = "skip_next_prefetch"

                self.get_logger().info(
                    "[EXP6-A] fresh initial chunk promoted; "
                    "next post-VR prefetch will be NON-RTC"
                )

            elif state in (
                "skip_next_prefetch",
                "await_first_rtc_prefetch",
            ):
                self.get_logger().warning(
                    "[EXP6-A] unexpected second INITIAL promotion while "
                    f"state={state}; abandoning pending EXP6-A sequence"
                )
                self.exp6a_post_vr_rtc_state = "idle"

            return

        if reason != "requery_steps":
            return

        # First post-VR prefetch: RTC must have been suppressed.
        if state == "skip_next_prefetch":
            if rtc_sent is False:
                self.exp6a_post_vr_rtc_state = "await_first_rtc_prefetch"

                self.get_logger().info(
                    "[EXP6-A] NON-RTC replan promoted at first post-VR "
                    "handoff; next prefetch will be the FIRST RTC query. "
                    f"source_query={request_id}, "
                    f"action_index_at_request="
                    f"{info.get('action_index_at_request')}"
                )
            else:
                self.get_logger().warning(
                    "[EXP6-A] expected first post-VR requery to be NON-RTC, "
                    f"but rtc_sent={rtc_sent}; abandoning EXP6-A sequence"
                )
                self.exp6a_post_vr_rtc_state = "idle"

            return

        # Next requery is the first RTC-guided chunk after the skipped cycle.
        if state == "await_first_rtc_prefetch":
            if rtc_sent is True:
                self.exp6a_post_vr_rtc_state = "idle"
                self.exp6a_completed_count += 1

                self.get_logger().info(
                    "[EXP6-A] FIRST RTC replan promoted after one NON-RTC "
                    "cycle; returning to untouched steady RTC baseline. "
                    f"completed_count={self.exp6a_completed_count}, "
                    f"source_query={request_id}, "
                    f"action_index_at_request="
                    f"{info.get('action_index_at_request')}"
                )
            else:
                self.get_logger().warning(
                    "[EXP6-A] expected first RTC requery after skipped cycle, "
                    f"but rtc_sent={rtc_sent}; abandoning EXP6-A sequence"
                )
                self.exp6a_post_vr_rtc_state = "idle"

    def _json_safe_rtc(self, rtc: dict | None) -> dict | None:
        if not rtc:
            return None
        prev = rtc.get("prev_chunk_left_over", None)
        return {
            "enabled": bool(rtc.get("enabled", False)),
            "reason": rtc.get("reason"),
            "prev_chunk_left_over_shape": list(np.asarray(prev).shape) if prev is not None else None,
            "prev_chunk_left_over_length": int(rtc.get("prev_chunk_left_over_length", 0)),
            "inference_delay": int(rtc.get("inference_delay", 0)),
            "execution_horizon": int(rtc.get("execution_horizon", 0)),
            "max_guidance_weight": float(rtc.get("max_guidance_weight", 0.0)),
            "prefix_attention_schedule": int(rtc.get("prefix_attention_schedule", 0)),
        }

    def _smooth_chunk_transition(
        self,
        old_chunk: np.ndarray | None,
        old_action_index: int,
        new_chunk: np.ndarray,
        reason: str,
    ) -> np.ndarray:
        if (
            not self.smoothing_enabled
            or self.smooth_transition_steps <= 0
            or old_chunk is None
            or reason != "requery_steps"
        ):
            return new_chunk

        old_chunk = validate_action_chunk(old_chunk)
        new_chunk = validate_action_chunk(new_chunk).copy()
        remaining_old = len(old_chunk) - old_action_index
        blend_steps = min(self.smooth_transition_steps, remaining_old, len(new_chunk))
        if blend_steps <= 0:
            return new_chunk

        mask = self.smooth_mask
        for i in range(blend_steps):
            # Smoothstep cross-fade: 0 -> 1 with zero-ish slope at both ends.
            t = float(i + 1) / float(blend_steps + 1)
            alpha = t * t * (3.0 - 2.0 * t)
            old_action = old_chunk[old_action_index + i]
            new_action = new_chunk[i].copy()
            new_action[mask] = (1.0 - alpha) * old_action[mask] + alpha * new_action[mask]
            new_chunk[i] = new_action

        self.get_logger().info(
            f"[smooth] blended transition: steps={blend_steps}, "
            f"old_start_index={old_action_index}, dims=arms+torso"
        )
        return new_chunk

    # ------------------------------------------------------------------
    # GRIPPER-DEBOUNCE7
    # ------------------------------------------------------------------

    def _classify_gripper_intent(self, value: float):
        """Return 0=LOW, 1=HIGH, None=inside hysteresis band."""
        value = float(value)

        if not np.isfinite(value):
            return None

        if value <= self.gripper_low_threshold:
            return 0

        if value >= self.gripper_high_threshold:
            return 1

        return None

    def _feedback_gripper_state(self, value):
        """
        Convert current physical feedback into a latch state.

        Clear endpoint feedback uses the normal hysteresis thresholds.
        If feedback is currently between thresholds (e.g. gripper is moving),
        use the threshold midpoint so the state machine still has one
        well-defined side to hold.
        """
        if value is None:
            return None

        value = float(value)

        if not np.isfinite(value):
            return None

        state = self._classify_gripper_intent(value)

        if state is not None:
            return state

        midpoint = 0.5 * (
            self.gripper_low_threshold + self.gripper_high_threshold
        )
        return 0 if value < midpoint else 1

    def _reset_gripper_stabilizer_from_feedback(self, reason: str) -> None:
        """
        Clear all pending transitions and anchor each gripper latch to
        the latest physical feedback.

        This is used after VR handback. The same state is also refreshed
        automatically after a long policy-output gap (episode/reset pause).
        """
        if not self.gripper_stabilizer_enabled:
            return

        for side, feedback in (
            ("right", self.right_gripper),
            ("left", self.left_gripper),
        ):
            state = self._feedback_gripper_state(feedback)

            self._gripper_latched_state[side] = state
            self._gripper_pending_state[side] = None
            self._gripper_pending_count[side] = 0

            # A real VR handback represents a freshly established human
            # gripper state. Protect it from an immediate policy reversal.
            if reason == "post_vr_handback":
                self._gripper_lock_remaining[side] = (
                    self.gripper_reversal_lock_steps
                )
            else:
                self._gripper_lock_remaining[side] = 0

            if feedback is not None and np.isfinite(float(feedback)):
                self._gripper_hold_value[side] = float(feedback)
            else:
                self._gripper_hold_value[side] = None

        self._gripper_last_policy_publish_time = None
        self._gripper_skip_gap_reset_once = (
            reason == "post_vr_handback"
        )

        self.get_logger().info(
            "[gripper-stabilizer] reset: "
            f"reason={reason}, "
            f"right_latch={self._gripper_latched_state['right']}, "
            f"left_latch={self._gripper_latched_state['left']}"
        )

    def _apply_gripper_stabilizer(
        self,
        action20: np.ndarray,
        obs: dict,
        now: float,
    ) -> tuple[np.ndarray, bool]:
        """
        Stateful gripper-only debounce/hysteresis.

        Arm and torso dimensions are never touched.

        For each gripper independently:
          - raw <= LOW threshold  -> LOW intent
          - raw >= HIGH threshold -> HIGH intent
          - otherwise             -> ambiguous; keep current latch

        A transition opposite to the current latch must be requested for
        N consecutive policy frames before it is committed.

        Example, N=7:
            H H H | L L L L L L L
                    1 2 3 4 5 6 7 -> commit LOW

        Any HIGH before count reaches 7 cancels the pending LOW transition.
        """
        if not self.gripper_stabilizer_enabled:
            return action20, False

        action = validate_action_chunk(
            np.asarray([action20], dtype=np.float32)
        )[0]
        out = action.copy()

        # A normal control interval is ~0.067 s at 15 Hz.
        # A >1 s policy-output gap means episode pause/reset/intervention.
        gap_reset_needed = (
            self._gripper_last_policy_publish_time is None
            or (
                now - self._gripper_last_policy_publish_time
            ) > 1.0
        )

        if gap_reset_needed:
            if self._gripper_skip_gap_reset_once:
                # post-VR handback has already initialized latch/hold/LOCK
                # from physical feedback. Preserve that state for this first
                # POLICY frame instead of immediately wiping the LOCK.
                self._gripper_skip_gap_reset_once = False
            else:
                for side, idx in (("right", 7), ("left", 15)):
                    state_value = None

                    try:
                        state_value = float(obs["state"][idx])
                    except Exception:
                        pass

                    state = self._feedback_gripper_state(state_value)

                    self._gripper_latched_state[side] = state
                    self._gripper_pending_state[side] = None
                    self._gripper_pending_count[side] = 0
                    self._gripper_lock_remaining[side] = 0

                    if state_value is not None and np.isfinite(state_value):
                        self._gripper_hold_value[side] = state_value
                    else:
                        self._gripper_hold_value[side] = None
        else:
            self._gripper_skip_gap_reset_once = False

        self._gripper_last_policy_publish_time = float(now)

        changed = False

        for side, idx in (
            ("right", 7),
            ("left", 15),
        ):
            raw_value = float(action[idx])
            candidate = self._classify_gripper_intent(raw_value)
            latched = self._gripper_latched_state[side]

            # If initialization feedback was unavailable, initialize from
            # the first clear policy intent. This first initialization is
            # not treated as a state transition.
            if latched is None:
                if candidate is not None:
                    self._gripper_latched_state[side] = candidate
                    self._gripper_hold_value[side] = raw_value
                    self._gripper_pending_state[side] = None
                    self._gripper_pending_count[side] = 0
                continue

            # ----------------------------------------------------------
            # REVERSAL LOCK
            #
            # After a committed transition, forbid the opposite state for
            # N complete POLICY frames.
            #
            # Opposite intents observed during the lock are intentionally
            # discarded. They are NOT allowed to pre-fill the debounce
            # counter. Therefore, after the lock expires, confirmation
            # must restart from candidate #1.
            # ----------------------------------------------------------
            if self._gripper_lock_remaining[side] > 0:
                self._gripper_pending_state[side] = None
                self._gripper_pending_count[side] = 0

                self._gripper_lock_remaining[side] -= 1

                if candidate == latched:
                    # Same committed mode is always allowed. Preserve the
                    # model's numeric target within that HIGH/LOW mode.
                    self._gripper_hold_value[side] = raw_value
                else:
                    # Opposite or ambiguous intent: hold committed output.
                    hold = self._gripper_hold_value[side]

                    if hold is not None and np.isfinite(float(hold)):
                        out[idx] = float(hold)
                    else:
                        try:
                            out[idx] = float(obs["state"][idx])
                        except Exception:
                            out[idx] = raw_value

                    if abs(float(out[idx]) - raw_value) > 1e-6:
                        changed = True

                continue

            # Middle hysteresis band is not a valid confirmation frame.
            # Hold the previously committed command and clear pending.
            if candidate is None:
                self._gripper_pending_state[side] = None
                self._gripper_pending_count[side] = 0

                hold = self._gripper_hold_value[side]
                if hold is not None and np.isfinite(float(hold)):
                    out[idx] = float(hold)

                continue

            # Policy agrees with the currently committed state.
            # Cancel any opposite candidate sequence.
            if candidate == latched:
                self._gripper_pending_state[side] = None
                self._gripper_pending_count[side] = 0

                # Preserve the model's value inside the already committed
                # HIGH/LOW state; do NOT quantize to fixed 0/80.
                self._gripper_hold_value[side] = raw_value
                continue

            # Opposite clear intent.
            if self._gripper_pending_state[side] == candidate:
                self._gripper_pending_count[side] += 1
            else:
                self._gripper_pending_state[side] = candidate
                self._gripper_pending_count[side] = 1

            count = self._gripper_pending_count[side]

            if count >= self.gripper_debounce_steps:
                old_state = latched

                self._gripper_latched_state[side] = candidate
                self._gripper_pending_state[side] = None
                self._gripper_pending_count[side] = 0
                self._gripper_hold_value[side] = raw_value

                # Start a fresh minimum-dwell / reversal-lock window.
                self._gripper_lock_remaining[side] = (
                    self.gripper_reversal_lock_steps
                )

                self.get_logger().info(
                    "[gripper-stabilizer] COMMIT: "
                    f"side={side}, "
                    f"{old_state}->{candidate}, "
                    f"confirmed_frames={self.gripper_debounce_steps}, "
                    f"reversal_lock_frames={self.gripper_reversal_lock_steps}, "
                    f"raw={raw_value:.3f}"
                )

                # 7th confirmation frame is the first frame that actually
                # publishes the new gripper state.
                continue

            # Not confirmed yet: keep publishing the previously committed
            # gripper command.
            hold = self._gripper_hold_value[side]

            if hold is not None and np.isfinite(float(hold)):
                out[idx] = float(hold)
            else:
                # Conservative fallback: hold the current measured state.
                try:
                    out[idx] = float(obs["state"][idx])
                except Exception:
                    out[idx] = raw_value

            if abs(float(out[idx]) - raw_value) > 1e-6:
                changed = True

        return out, changed

    def _publish_one_action(self, raw_action20, obs: dict, now: float):
        safe_action20, report = self.safety.sanitize(raw_action20, obs["state"])
        safe_action20, smoothed = self._smooth_published_action(safe_action20)
        if smoothed:
            report.reasons.append("publish_smooth")

        safe_action20, gripper_held = self._apply_gripper_stabilizer(
            safe_action20,
            obs,
            now,
        )
        if gripper_held:
            report.reasons.append("gripper_debounce_hold")

        self.last_safe_action20 = safe_action20.copy()
        self.published_steps += 1

        if now - self.last_status_time > self.args.preview_period_sec:
            self.last_status_time = now
            prefix = "[execute]" if self.execute else "[dry-run]"
            self.get_logger().info(f"{prefix} action20 preview: {compact_action_summary(safe_action20)} | safety={report.summary()}")

        if self.args.print_action_table and now - self.last_table_time > self.args.table_period_sec:
            self.last_table_time = now
            self.get_logger().info("\n" + action_table(safe_action20, obs["state"]))

        self._record_step(raw_action20, safe_action20, report.summary())

        if self.execute:
            self._publish_action20(safe_action20)

    def _smooth_published_action(self, action20: np.ndarray) -> tuple[np.ndarray, bool]:
        if (
            not self.publish_smoothing_enabled
            or self.last_safe_action20 is None
            or not np.all(np.isfinite(self.last_safe_action20))
        ):
            return action20, False

        prev = validate_action_chunk(np.asarray([self.last_safe_action20], dtype=np.float32))[0]
        action = validate_action_chunk(np.asarray([action20], dtype=np.float32))[0]
        smoothed = action.copy()
        alpha = self.publish_smooth_alpha
        smoothed[self.publish_smooth_mask] = (
            (1.0 - alpha) * prev[self.publish_smooth_mask]
            + alpha * action[self.publish_smooth_mask]
        )
        changed = not np.allclose(smoothed, action, atol=1e-6, rtol=0.0)
        return smoothed, changed

    def _control_timer_sync(self, obs: dict, now: float):
        if self._need_new_chunk():
            try:
                self.action_chunk_raw = self._query_policy(obs)
            except Exception as e:
                self.get_logger().error(f"Policy inference failed: {e}")
                self.action_chunk_raw = None
                self.action_index = 0
                return
            self.action_index = 0

        raw_action20 = self.action_chunk_raw[self.action_index]
        self.action_index += 1
        self._publish_one_action(raw_action20, obs, now)

    def _control_timer_async(self, obs: dict, now: float):
        if self.action_chunk_raw is None:
            if not self._promote_async_chunk_if_ready("initial"):
                self._maybe_start_async_query(obs, "initial")
                if now - self.last_async_status_time > 1.0:
                    self.last_async_status_time = now
                    with self.async_lock:
                        inflight = self.async_inflight
                        error = self.async_last_error
                    self.get_logger().info(
                        f"[async] waiting for initial action chunk: "
                        f"inflight={inflight}, last_error={error}"
                    )
                return

        if self.action_index >= self.args.requery_steps:
            if not self._promote_async_chunk_if_ready("requery_steps"):
                self.async_missed_requery_count += 1
                if now - self.last_async_status_time > 1.0:
                    self.last_async_status_time = now
                    with self.async_lock:
                        inflight = self.async_inflight
                    self.get_logger().warn(
                        f"{ANSI_YELLOW}[async] next chunk not ready at requery boundary; "
                        f"continuing current chunk at index={self.action_index}, "
                        f"inflight={inflight}, misses={self.async_missed_requery_count}"
                        f"{ANSI_RESET}"
                    )

        if self.action_index >= self.args.async_prefetch_steps:
            self._maybe_start_async_query(obs, f"prefetch_at_{self.action_index}")

        if self.action_index >= len(self.action_chunk_raw):
            if self._promote_async_chunk_if_ready("horizon_end"):
                pass
            elif self.last_safe_action20 is not None:
                if now - self.last_async_status_time > 1.0:
                    self.last_async_status_time = now
                    self.get_logger().warn("[async] action horizon exhausted; holding last safe action")
                self._publish_one_action(self.last_safe_action20, obs, now)
                return
            else:
                self._maybe_start_async_query(obs, "horizon_end_no_action")
                return

        raw_action20 = self.action_chunk_raw[self.action_index]
        self.action_index += 1
        self._publish_one_action(raw_action20, obs, now)

    def _control_timer(self):
        now = time.monotonic()
        if self.args.duration_sec > 0 and (now - self.start_time) >= self.args.duration_sec:
            self.get_logger().info("duration-sec reached, stopping.")
            self._request_shutdown()
            return
        if self.args.max_steps > 0 and self.published_steps >= self.args.max_steps:
            self.get_logger().info("max-steps reached, stopping.")
            self._request_shutdown()
            return

        if not self._ready():
            if now - self.last_status_time > 2.0:
                self.last_status_time = now
                self.get_logger().info(
                    "Waiting for all inputs: "
                    f"cam_high={self.cam_high is not None}, "
                    f"cam_left_wrist={self.cam_left_wrist is not None}, "
                    f"cam_right_wrist={self.cam_right_wrist is not None}, "
                    f"right_arm={self.right_arm is not None}, "
                    f"right_gripper={self.right_gripper is not None}, "
                    f"left_arm={self.left_arm is not None}, "
                    f"left_gripper={self.left_gripper is not None}, "
                    f"torso={self.torso is not None}"
                )
            return

        if self.args.show_images:
            self._show_images()

        obs = self._build_obs()
        if self.async_enabled:
            self._control_timer_async(obs, now)
        else:
            self._control_timer_sync(obs, now)

    def _show_images(self):
        if self.show_images_failed:
            return
        try:
            imgs = []
            for title, rgb in [("high", self.cam_high), ("left", self.cam_left_wrist), ("right", self.cam_right_wrist)]:
                small = cv2.resize(rgb, (320, 180))
                bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
                cv2.putText(bgr, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                imgs.append(bgr)
            canvas = np.concatenate(imgs, axis=1)
            cv2.putText(canvas, "EXECUTE" if self.execute else "DRY-RUN", (8, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("R1Pro OpenPI cameras", canvas)
            cv2.waitKey(1)
        except Exception as e:
            self.show_images_failed = True
            self.get_logger().warn(f"show-images disabled after error: {e}")

    def _make_joint_state(self, names, positions, fallback_names) -> JointState:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        positions = np.asarray(positions, dtype=np.float32).reshape(-1)
        if names and len(names) == len(positions):
            msg.name = list(names)
        elif fallback_names and len(fallback_names) == len(positions):
            msg.name = list(fallback_names)
        else:
            msg.name = [f"joint_{i}" for i in range(len(positions))]
        msg.position = [float(x) for x in positions]
        return msg

    def _make_gripper_joint_state(self, joint_name: str, position: float) -> JointState:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [joint_name]
        msg.position = [float(position)]
        return msg

    def _publish_action20(self, action20):
        parts = split_action20(action20)
        self.pub_right_arm.publish(self._make_joint_state(self.right_arm_names, parts["right_arm"], RIGHT_ARM_FALLBACK_NAMES))
        self.pub_left_arm.publish(self._make_joint_state(self.left_arm_names, parts["left_arm"], LEFT_ARM_FALLBACK_NAMES))
        self.pub_torso.publish(self._make_joint_state(self.torso_names, parts["torso"], TORSO_FALLBACK_NAMES))

        if self.args.gripper_target_type == "joint_state":
            self.pub_right_gripper.publish(self._make_gripper_joint_state(self.args.right_gripper_joint_name, parts["right_gripper"]))
            self.pub_left_gripper.publish(self._make_gripper_joint_state(self.args.left_gripper_joint_name, parts["left_gripper"]))
        else:
            r = Float64(); r.data = float(parts["right_gripper"])
            l = Float64(); l.data = float(parts["left_gripper"])
            self.pub_right_gripper.publish(r)
            self.pub_left_gripper.publish(l)

    def _request_shutdown(self):
        try:
            if self.record_writer is not None:
                with self.record_lock:
                    self.record_writer.close()
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser(description="R1Pro ROS2 <-> OpenPI websocket client")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--policy-mode", choices=["server", "hold_current"], default="server")
    parser.add_argument("--mock-policy", action="store_true", help="Alias for --policy-mode hold_current")
    parser.add_argument("--execute", action="store_true", help="Actually publish robot target commands.")

    parser.add_argument("--rate-hz", type=float, default=DEFAULT_RATE_HZ)
    parser.add_argument("--requery-steps", type=int, default=DEFAULT_REQUERY_STEPS)
    parser.add_argument("--sync-infer", action="store_true", help="Disable async inference and use the old blocking policy query path.")
    parser.add_argument("--async-prefetch-steps", type=int, default=6, help="Start the next background policy query after this many action steps from the current chunk.")
    parser.add_argument("--disable-rtc", action="store_true", help="Disable Real-Time Chunking fields sent to the OpenPI server.")
    parser.add_argument("--rtc-execution-horizon", type=int, default=10, help="RTC overlap horizon. Official LeRobot typical values are 8-12.")
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=10.0, help="RTC max guidance weight. LeRobot recommends 10.0 for 10-step flow matching policies.")
    parser.add_argument("--rtc-prefix-attention-schedule", choices=["linear", "exp", "ones", "zeros"], default="exp", help="RTC prefix attention schedule. LeRobot recommends exp for getting started.")
    parser.add_argument("--rtc-initial-inference-delay-steps", type=int, default=3, help="Initial RTC inference delay estimate before latency samples are available.")
    parser.add_argument("--rtc-max-inference-delay-steps", type=int, default=4, help="Cap RTC inference-delay estimate in action steps so one-time JIT compile latency does not poison RTC prefix alignment. 0 disables the cap.")
    parser.add_argument("--enable-client-smoothing", action="store_true", help="Keep legacy client-side smoothstep/EMA smoothing even when RTC is enabled.")
    parser.add_argument("--smooth-transition-steps", type=int, default=4, help="Blend this many steps from the old chunk tail into the new chunk head at each requery boundary.")
    parser.add_argument("--disable-action-smoothing", action="store_true", help="Disable chunk-boundary action smoothing.")
    parser.add_argument("--disable-publish-smoothing", action="store_true", help="Disable final EMA low-pass smoothing against the last published action.")
    parser.add_argument("--publish-smooth-alpha", type=float, default=0.35, help="EMA low-pass factor for final joint publish smoothing. 1.0 means no smoothing; lower is smoother but slower.")

    # GRIPPER-DEBOUNCE7: default OFF so the sealed formal baseline is unchanged.
    parser.add_argument(
        "--enable-gripper-stabilizer",
        action="store_true",
        help="Enable stateful gripper-only hysteresis/debounce/latch.",
    )
    parser.add_argument(
        "--gripper-debounce-steps",
        type=int,
        default=7,
        help="Require this many consecutive opposite gripper intents before committing.",
    )
    parser.add_argument(
        "--gripper-reversal-lock-steps",
        type=int,
        default=0,
        help=(
            "After a committed gripper transition, ignore opposite "
            "state intents for this many policy frames before allowing "
            "a fresh debounce sequence."
        ),
    )
    parser.add_argument(
        "--gripper-low-threshold",
        type=float,
        default=20.0,
        help="Gripper command <= this value is a clear LOW intent.",
    )
    parser.add_argument(
        "--gripper-high-threshold",
        type=float,
        default=60.0,
        help="Gripper command >= this value is a clear HIGH intent.",
    )

    parser.add_argument("--max-steps", type=int, default=0, help="0 means no limit")
    parser.add_argument("--duration-sec", type=float, default=0.0, help="0 means no limit")
    parser.add_argument("--preview-period-sec", type=float, default=1.0)
    parser.add_argument("--print-action-table", action="store_true")
    parser.add_argument("--table-period-sec", type=float, default=5.0)

    parser.add_argument("--disable-abs-clip", action="store_true")
    parser.add_argument("--disable-step-limit", action="store_true")
    parser.add_argument("--enable-step-limit", action="store_true", help="Re-enable the old hard per-step joint clipping safety layer.")
    parser.add_argument("--force-fixed-torso-target", action="store_true", help=f"Force torso target to {FIXED_TORSO_TARGET.tolist()} before publishing")

    parser.add_argument("--show-images", action="store_true")
    parser.add_argument("--record-debug", action="store_true")
    parser.add_argument(
        "--record-dir",
        default=str(THIS_DIR / "records" / "r1pro_policy_records" / "latest"),
    )
    parser.add_argument("--record-images", action="store_true")

    parser.add_argument("--topic-cam-high", default=TOPIC_CAM_HIGH)
    parser.add_argument("--topic-cam-left-wrist", default=TOPIC_CAM_LEFT_WRIST)
    parser.add_argument("--topic-cam-right-wrist", default=TOPIC_CAM_RIGHT_WRIST)

    parser.add_argument("--topic-feedback-arm-right", default=TOPIC_FEEDBACK_ARM_RIGHT)
    parser.add_argument("--topic-feedback-arm-left", default=TOPIC_FEEDBACK_ARM_LEFT)
    parser.add_argument("--topic-feedback-torso", default=TOPIC_FEEDBACK_TORSO)
    parser.add_argument("--topic-feedback-gripper-right", default=TOPIC_FEEDBACK_GRIPPER_RIGHT)
    parser.add_argument("--topic-feedback-gripper-left", default=TOPIC_FEEDBACK_GRIPPER_LEFT)
    parser.add_argument("--gripper-feedback-type", choices=["joint_state", "float64"], default="joint_state")

    parser.add_argument("--topic-target-arm-right", default=TOPIC_TARGET_ARM_RIGHT)
    parser.add_argument("--topic-target-arm-left", default=TOPIC_TARGET_ARM_LEFT)
    parser.add_argument("--topic-target-torso", default=TOPIC_TARGET_TORSO)
    parser.add_argument("--topic-target-gripper-right", default=TOPIC_TARGET_GRIPPER_RIGHT)
    parser.add_argument("--topic-target-gripper-left", default=TOPIC_TARGET_GRIPPER_LEFT)
    parser.add_argument("--gripper-target-type", choices=["joint_state", "float64"], default="joint_state")
    parser.add_argument("--right-gripper-joint-name", default=RIGHT_GRIPPER_TARGET_JOINT_NAME)
    parser.add_argument("--left-gripper-joint-name", default=LEFT_GRIPPER_TARGET_JOINT_NAME)

    args, ros_args = parser.parse_known_args()
    return args, ros_args


def main():
    args, ros_args = parse_args()
    rclpy.init(args=ros_args)
    node = R1ProRosOpenPIClient(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
