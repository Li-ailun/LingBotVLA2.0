#!/usr/bin/env python3
"""Keyboard-controlled R1Pro ROS 2 MCAP episode recorder.

Records exactly three RGB streams plus dual-arm joint and dual-gripper state/action
topics. The operator starts and stops each episode, then chooses whether to keep
or discard it. Torso topics are never recorded.

Example:
  source /opt/ros/humble/setup.bash
  /usr/bin/python3 myscripts/16d/record_tools/record_r1pro_mcap_episode.py \
    --root datasets/r1pro_mcap_raw --task pick_up_object
"""
import argparse
import json
import os
import pty
import shutil
import signal
import subprocess
import threading
import time
import sys
import termios
from pathlib import Path
from collections import deque
from typing import NamedTuple

# ======================================================================================
# Terminal colors: only affects printed text, not recording logic.
# ======================================================================================
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_RESET = "\033[0m"


def color_text(text, color):
    return f"{color}{text}{COLOR_RESET}"


import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    HistoryPolicy,
    ReliabilityPolicy,
    DurabilityPolicy,
)

from sensor_msgs.msg import CompressedImage, JointState, Imu
from geometry_msgs.msg import PoseStamped, TwistStamped


# ======================================================================================
# ROS topics
# ======================================================================================

IMAGE_TOPICS = {
    "head_rgb": "/hdas/camera_head/left_raw/image_raw_color/compressed",
    "head_right_rgb": "/hdas/camera_head/right_raw/image_raw_color/compressed",
    "left_wrist_rgb": "/hdas/camera_wrist_left/color/image_raw/compressed",
    "right_wrist_rgb": "/hdas/camera_wrist_right/color/image_raw/compressed",
}

STATE_TOPICS = {
    "feedback_arm_left": "/hdas/feedback_arm_left",
    "feedback_arm_right": "/hdas/feedback_arm_right",
    "feedback_gripper_left": "/hdas/feedback_gripper_left",
    "feedback_gripper_right": "/hdas/feedback_gripper_right",
    "feedback_chassis": "/hdas/feedback_chassis",
    "imu_chassis": "/hdas/imu_chassis",
    "pose_ee_arm_left": "/motion_control/pose_ee_arm_left",
    "pose_ee_arm_right": "/motion_control/pose_ee_arm_right",
}

ACTION_TOPICS = {
    "target_pose_arm_left": "/motion_target/target_pose_arm_left",
    "target_pose_arm_right": "/motion_target/target_pose_arm_right",
    "target_position_gripper_left": "/motion_target/target_position_gripper_left",
    "target_position_gripper_right": "/motion_target/target_position_gripper_right",
    "target_speed_chassis": "/motion_target/target_speed_chassis",
    "target_joint_state_arm_left": "/motion_target/target_joint_state_arm_left",
    "target_joint_state_arm_right": "/motion_target/target_joint_state_arm_right",
}

# R1Pro 16D 训练真正需要的11个话题：3路图像 + 4路state + 4路action。
R1PRO_16D_CORE_TOPIC_KEYS = [
    "head_rgb",
    "left_wrist_rgb",
    "right_wrist_rgb",
    "feedback_arm_left",
    "feedback_arm_right",
    "feedback_gripper_left",
    "feedback_gripper_right",
    "target_joint_state_arm_left",
    "target_joint_state_arm_right",
    "target_position_gripper_left",
    "target_position_gripper_right",
]

# 轻量完整录制集合：在 11 个 R1Pro 16D 必要话题基础上，
# 增加双臂末端位姿反馈和双臂末端位姿指令，共 15 个话题。
#
# 组成：
#   3 路图像
#   5 路关节/夹爪/腰部反馈
#   5 路关节/夹爪/腰部指令
#   2 路末端位姿反馈
#   2 路末端位姿指令
MINIMAL_WITH_EE_TOPIC_KEYS = [
    *R1PRO_16D_CORE_TOPIC_KEYS,
    "pose_ee_arm_left",
    "pose_ee_arm_right",
    "target_pose_arm_left",
    "target_pose_arm_right",
]

def all_dataset_topics():
    topics = []
    topics.extend(IMAGE_TOPICS.values())
    topics.extend(STATE_TOPICS.values())
    topics.extend(ACTION_TOPICS.values())
    return sorted(set(topics))


def r1pro_16d_core_topics():
    merged = {}
    merged.update(IMAGE_TOPICS)
    merged.update(STATE_TOPICS)
    merged.update(ACTION_TOPICS)
    return sorted(set(merged[k] for k in R1PRO_16D_CORE_TOPIC_KEYS))


def minimal_with_ee_topics():
    """Return the 15-topic lightweight set: R1Pro 16D core + dual-arm EE pose feedback/targets."""
    merged = {}
    merged.update(IMAGE_TOPICS)
    merged.update(STATE_TOPICS)
    merged.update(ACTION_TOPICS)
    return sorted(set(merged[k] for k in MINIMAL_WITH_EE_TOPIC_KEYS))


def str2bool(x):
    if isinstance(x, bool):
        return x
    x = str(x).strip().lower()
    if x in ("1", "true", "yes", "y", "on"):
        return True
    if x in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {x}")


def apply_recording_mode(args):
    """Resolve the direct-recording shortcut onto the legacy monitor flag."""
    if getattr(args, "direct_recording", False):
        args.enable_timestamp_monitor = False
    return args


def safe_task_dir_name(task: str) -> str:
    text = str(task).strip()
    safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in text)
    while "__" in safe:
        safe = safe.replace("__", "_")
    safe = safe.strip("._-")
    return safe[:120] if safe else "untitled_task"


class InstructionSpec(NamedTuple):
    task_id: str
    instruction_id: str
    text: str
    language: str = "en"


PLATE_CUP_INSTRUCTIONS = {
    "stack_plates_cups_01": InstructionSpec(
        "stack_plates_and_cups",
        "stack_plates_cups_01",
        "Arrange the blue plate at the bottom with the pink plate on top. "
        "Arrange the cups from bottom to top as blue, red, and yellow, "
        "with the cup stack on the plate stack.",
    ),
    "stack_plates_cups_02": InstructionSpec(
        "stack_plates_and_cups",
        "stack_plates_cups_02",
        "Arrange the blue plate at the bottom with the pink plate on top. "
        "Arrange the cups from bottom to top as red, yellow, and blue, "
        "with the cup stack on the plate stack.",
    ),
    "stack_plates_cups_03": InstructionSpec(
        "stack_plates_and_cups",
        "stack_plates_cups_03",
        "Arrange the pink plate at the bottom with the blue plate on top. "
        "Arrange the cups from bottom to top as yellow, blue, and red, "
        "with the cup stack on the plate stack.",
    ),
}
INSTRUCTION_ALIASES = {
    "1": "stack_plates_cups_01",
    "2": "stack_plates_cups_02",
    "3": "stack_plates_cups_03",
    **{key: key for key in PLATE_CUP_INSTRUCTIONS},
}

OPERATOR_IDS = ["li", "liu", "yu", "xing"]
OPERATOR_ORDER = {operator_id: index for index, operator_id in enumerate(OPERATOR_IDS)}


def resolve_operator(value: str) -> str:
    operator_id = str(value).strip().lower()
    if operator_id not in OPERATOR_ORDER:
        raise ValueError(
            f"Unknown operator {value!r}; choose one of: {', '.join(OPERATOR_IDS)}"
        )
    return operator_id


def resolve_instruction(value: str) -> InstructionSpec:
    key = INSTRUCTION_ALIASES.get(str(value).strip().lower())
    if key is None:
        choices = ", ".join(sorted(INSTRUCTION_ALIASES))
        raise ValueError(f"Unknown instruction {value!r}; choose one of: {choices}")
    return PLATE_CUP_INSTRUCTIONS[key]


def instruction_path_parts(spec: InstructionSpec) -> list[str]:
    return [safe_task_dir_name(spec.task_id), safe_task_dir_name(spec.instruction_id)]


def apply_instruction(args, value: str) -> InstructionSpec:
    spec = resolve_instruction(value)
    args.task = spec.text
    args.task_id = spec.task_id
    args.instruction_id = spec.instruction_id
    args.instruction_text = spec.text
    args.language = spec.language
    return spec


def task_directory(root: Path, args) -> Path:
    operator_id = resolve_operator(args.operator_id)
    if getattr(args, "instruction_id", None):
        spec = resolve_instruction(args.instruction_id)
        task_id, instruction_id = instruction_path_parts(spec)
        return root / task_id / operator_id / instruction_id
    return root / safe_task_dir_name(args.task) / operator_id


def now_stamp_ms() -> str:
    t = time.time()
    return time.strftime("%Y%m%d%H%M%S", time.localtime(t)) + f"{int((t % 1) * 1000):03d}"


def json_safe(x):
    if isinstance(x, Path):
        return str(x.expanduser())
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    return str(x)



def flush_stdin_buffer():
    """Clear already-buffered ENTER keys before starting the stop prompt."""
    try:
        if sys.stdin is not None and sys.stdin.isatty():
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass


def msg_header_stamp_sec(msg):
    """Return ROS header stamp seconds, or None if msg has no valid header stamp."""
    if not hasattr(msg, "header"):
        return None
    stamp = getattr(msg.header, "stamp", None)
    if stamp is None:
        return None
    sec = int(stamp.sec)
    nsec = int(stamp.nanosec)
    if sec == 0 and nsec == 0:
        return None
    return float(sec) + float(nsec) * 1e-9


def get_next_episode_index(task_dir: Path) -> int:
    if not task_dir.exists():
        return 0
    eps = []
    for p in task_dir.glob("episode_*"):
        if not p.is_dir():
            continue
        name = p.name
        # episode_000000_20260615010101999
        parts = name.split("_")
        if len(parts) >= 2 and parts[0] == "episode":
            try:
                eps.append(int(parts[1]))
            except Exception:
                pass
    return max(eps) + 1 if eps else 0


def episode_directories_for_index(task_dir: Path, episode_index: int) -> list[str]:
    """Return existing episode directories that use the requested MCAP index."""
    index = int(episode_index)
    if index < 0:
        raise ValueError("episode index must be a non-negative integer")
    prefix = f"episode_{index:06d}"
    return [
        str(path)
        for path in sorted(task_dir.glob(f"{prefix}*"))
        if path.is_dir() and (path.name == prefix or path.name.startswith(prefix + "_"))
    ]


def prompt_start_episode_index(task_dir: Path, input_fn=input, print_fn=print) -> int:
    """Prompt until a non-negative, unused MCAP episode index is selected."""
    recommended = get_next_episode_index(task_dir)
    while True:
        raw = input_fn(f"Start episode index [recommended: {recommended}]: ").strip()
        if raw == "":
            candidate = recommended
        else:
            try:
                candidate = int(raw)
            except ValueError:
                print_fn("[EPISODE INDEX] Enter a non-negative integer.")
                continue
        if candidate < 0:
            print_fn("[EPISODE INDEX] Enter a non-negative integer.")
            continue

        existing = episode_directories_for_index(task_dir, candidate)
        if existing:
            print_fn(
                f"[EPISODE INDEX] episode_{candidate:06d} already exists: "
                + ", ".join(existing)
            )
            continue
        return candidate


def require_available_episode_index(task_dir: Path, episode_index: int) -> int:
    """Validate an explicitly supplied non-negative MCAP episode index."""
    index = int(episode_index)
    existing = episode_directories_for_index(task_dir, index)
    if existing:
        raise ValueError(
            f"episode_{index:06d} already exists: " + ", ".join(existing)
        )
    return index


class TimestampMonitorNode(Node):
    """Subscribe to selected topics only for health/timestamp monitoring."""

    def __init__(self, args):
        super().__init__("r1pro_mcap_timestamp_monitor")
        self.args = args

        self.latest = {}
        self.counts = {}
        self.history = deque(maxlen=10000)
        self.recording_started_monotonic = None
        self.last_check_monotonic = 0.0
        self.last_report = None

        self.topic_key_to_topic = {}
        self.topic_key_to_topic.update(IMAGE_TOPICS)
        self.topic_key_to_topic.update(STATE_TOPICS)
        self.topic_key_to_topic.update(ACTION_TOPICS)

        self.monitor_keys = list(R1PRO_16D_CORE_TOPIC_KEYS)

        # 监控只关心最新消息，不需要缓存 50 条，统一 KEEP_LAST depth=1。
        self.monitor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self._monitor_subscriptions = []
        for key in self.monitor_keys:
            topic = self.topic_key_to_topic[key]
            msg_type = self._msg_type_for_key(key)
            sub = self.create_subscription(
                msg_type,
                topic,
                lambda msg, key=key, topic=topic: self._cb(key, topic, msg),
                self.monitor_qos,
            )
            self._monitor_subscriptions.append(sub)

    def _msg_type_for_key(self, key):
        if key in IMAGE_TOPICS:
            return CompressedImage
        if key == "imu_chassis":
            return Imu
        if key.startswith("pose_ee") or key.startswith("target_pose"):
            return PoseStamped
        if key == "target_speed_chassis":
            return TwistStamped
        return JointState

    def _cb(self, key, topic, msg):
        arrival = time.monotonic()
        header_stamp = msg_header_stamp_sec(msg)

        self.latest[key] = {
            "topic": topic,
            "arrival_monotonic": arrival,
            "header_stamp": header_stamp,
            "has_header_stamp": header_stamp is not None,
        }
        self.counts[key] = self.counts.get(key, 0) + 1
        self.history.append({
            "key": key,
            "topic": topic,
            "arrival_monotonic": arrival,
            "header_stamp": header_stamp,
            "has_header_stamp": header_stamp is not None,
        })

    def reset_episode_state(self):
        """Forget all data from the previous episode/preparation phase."""
        self.latest.clear()
        self.counts.clear()
        self.history.clear()
        self.recording_started_monotonic = None
        self.last_check_monotonic = 0.0
        self.last_report = None

    def ready_missing_keys(self, min_messages=1):
        return [
            key for key in self.monitor_keys
            if key not in self.latest or self.counts.get(key, 0) < int(min_messages)
        ]

    def _evaluate_current(
        self,
        *,
        max_age,
        max_span,
        min_messages=1,
        force_arrival=False,
        elapsed=None,
    ):
        now = time.monotonic()
        missing = self.ready_missing_keys(min_messages=min_messages)

        stale = []
        for key in self.monitor_keys:
            item = self.latest.get(key)
            if item is None:
                continue
            age = now - float(item["arrival_monotonic"])
            if age > float(max_age):
                stale.append({
                    "key": key,
                    "topic": item["topic"],
                    "age": age,
                    "max_age": float(max_age),
                })

        valid_header_items = [
            (key, self.latest[key])
            for key in self.monitor_keys
            if key in self.latest and self.latest[key]["has_header_stamp"]
        ]
        all_have_header = len(valid_header_items) == len(self.monitor_keys)

        if not force_arrival and all_have_header:
            source = "header_stamp"
            values = [(key, float(item["header_stamp"])) for key, item in valid_header_items]
        else:
            if (
                not force_arrival
                and not self.args.timestamp_use_arrival_fallback
                and not all_have_header
            ):
                no_header = [
                    key for key in self.monitor_keys
                    if key in self.latest and not self.latest[key]["has_header_stamp"]
                ]
                return {
                    "ok": False,
                    "reasons": ["missing_header_stamp"],
                    "source": "header_stamp",
                    "elapsed": elapsed,
                    "timestamp_span": None,
                    "timestamp_max_span": float(max_span),
                    "oldest_key": None,
                    "oldest_time": None,
                    "newest_key": None,
                    "newest_time": None,
                    "missing": missing,
                    "stale": stale,
                    "no_valid_header_stamp": no_header,
                    "counts": dict(self.counts),
                    "monitor_keys": list(self.monitor_keys),
                }

            source = "arrival_time_fallback"
            values = [
                (key, float(self.latest[key]["arrival_monotonic"]))
                for key in self.monitor_keys
                if key in self.latest
            ]

        if values:
            min_key, min_t = min(values, key=lambda x: x[1])
            max_key, max_t = max(values, key=lambda x: x[1])
            span = max_t - min_t
        else:
            min_key = max_key = None
            min_t = max_t = span = None

        reasons = []
        if missing:
            reasons.append("missing_topics")
        if stale:
            reasons.append("stale_topics")
        if span is None:
            reasons.append("no_timestamps")
        elif span > float(max_span):
            reasons.append("timestamp_span_too_large")

        return {
            "ok": not reasons,
            "reasons": reasons,
            "source": source,
            "elapsed": elapsed,
            "timestamp_span": span,
            "timestamp_max_span": float(max_span),
            "oldest_key": min_key,
            "oldest_time": min_t,
            "newest_key": max_key,
            "newest_time": max_t,
            "missing": missing,
            "stale": stale,
            "counts": dict(self.counts),
            "monitor_keys": list(self.monitor_keys),
        }

    def wait_until_stable(
        self,
        *,
        stable_sec,
        timeout_sec,
        min_messages,
        stage_name,
    ):
        """Require all monitored topics to remain fresh/synchronized continuously."""
        deadline = time.monotonic() + float(timeout_sec)
        stable_since = None
        next_log = 0.0
        last_report = {}

        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
            now = time.monotonic()
            report = self._evaluate_current(
                max_age=float(self.args.timestamp_max_age),
                max_span=float(self.args.timestamp_max_span),
                min_messages=int(min_messages),
                force_arrival=True,
                elapsed=None,
            )
            last_report = report

            if report["ok"]:
                if stable_since is None:
                    stable_since = now
                held = now - stable_since
                if now >= next_log:
                    print(
                        f"[{stage_name}] healthy; stable {held:.1f}/{float(stable_sec):.1f}s "
                        f"span={float(report['timestamp_span']):.4f}s",
                        flush=True,
                    )
                    next_log = now + 1.0
                if held >= float(stable_sec):
                    report["stable_sec_reached"] = held
                    return True, report
            else:
                stable_since = None
                if now >= next_log:
                    print(
                        f"[{stage_name}] waiting: reasons={report['reasons']} "
                        f"missing={report['missing']} stale={[x['key'] for x in report['stale']]}",
                        flush=True,
                    )
                    next_log = now + 1.0

        last_report = dict(last_report or {})
        last_report["stable_timeout_sec"] = float(timeout_sec)
        last_report["stable_sec_required"] = float(stable_sec)
        return False, last_report

    def start_recording_monitor(self):
        self.recording_started_monotonic = time.monotonic()
        self.last_check_monotonic = 0.0
        self.last_report = None

    def check_timestamp_sync(self, force=False):
        if self.recording_started_monotonic is None:
            return True, "not_recording", {}

        now = time.monotonic()
        if not force and now - self.last_check_monotonic < float(self.args.timestamp_check_every):
            return True, "skip", self.last_report or {}

        self.last_check_monotonic = now
        elapsed = now - self.recording_started_monotonic
        if elapsed < float(self.args.timestamp_grace):
            return True, "grace", {"elapsed": elapsed}

        report = self._evaluate_current(
            max_age=float(self.args.timestamp_max_age),
            max_span=float(self.args.timestamp_max_span),
            min_messages=1,
            force_arrival=False,
            elapsed=elapsed,
        )
        self.last_report = report

        if report["ok"]:
            return True, "ok", report
        return False, ",".join(report["reasons"]), report

    def print_timestamp_report(self):
        ok, status, report = self.check_timestamp_sync(force=True)
        if status in {"skip", "grace", "not_recording"}:
            print(f"[TS] {status}: {report}")
            return ok

        span = report.get("timestamp_span")
        span_text = "None" if span is None else f"{span:.4f}s"
        print(
            f"[TS] ok={report.get('ok')} source={report.get('source')} "
            f"span={span_text} max={report.get('timestamp_max_span')} "
            f"oldest={report.get('oldest_key')} newest={report.get('newest_key')}"
        )
        if report.get("missing"):
            print("[TS][missing]", report["missing"])
        if report.get("stale"):
            print("[TS][stale]", report["stale"])
        if report.get("reasons"):
            print("[TS][reasons]", report["reasons"])
        return ok


class McapRecorder:
    def __init__(self, args, bag_dir: Path, topics: list[str]):
        self.args = args
        self.bag_dir = Path(bag_dir).expanduser().resolve()
        self.topics = topics
        self.proc = None
        self.pty_master_fd = None
        self.paused = False
        self.start_mode = None
        self.recording_started_wall_time = None
        self.stderr_lines = []
        self.stderr_thread = None

    def _drain_stderr(self, stream):
        try:
            for line in iter(stream.readline, ""):
                self.stderr_lines.append(line)
        except Exception:
            pass
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def _build_cmd(self, *, start_paused: bool):
        ros2 = shutil.which("ros2")
        if ros2 is None:
            raise RuntimeError("ros2 command not found. Please source ROS2 environment first.")

        cmd = [
            ros2,
            "bag",
            "record",
            "-s",
            str(self.args.storage_id),
            "-o",
            str(self.bag_dir),
        ]
        if start_paused:
            cmd.append("--start-paused")

        cmd.extend(self.topics)
        return cmd

    def _launch(self, cmd, *, use_pty: bool):
        if self.bag_dir.exists():
            raise FileExistsError(f"Bag directory already exists: {self.bag_dir}")

        self.bag_dir.parent.mkdir(parents=True, exist_ok=True)

        if use_pty:
            master_fd, slave_fd = pty.openpty()
            self.pty_master_fd = master_fd
            stdin_arg = slave_fd
        else:
            slave_fd = None
            self.pty_master_fd = None
            stdin_arg = subprocess.DEVNULL

        try:
            self.proc = subprocess.Popen(
                cmd,
                stdin=stdin_arg,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid,
                bufsize=1,
            )
            launch_wall_time = time.time()
        finally:
            if slave_fd is not None:
                os.close(slave_fd)

        if self.proc.stderr is not None:
            self.stderr_thread = threading.Thread(
                target=self._drain_stderr,
                args=(self.proc.stderr,),
                daemon=True,
            )
            self.stderr_thread.start()

        time.sleep(float(self.args.bag_startup_wait))
        if self.proc.poll() is not None:
            raise RuntimeError(
                f"ros2 bag record exited early with code {self.proc.returncode}. stderr:\n"
                + "".join(self.stderr_lines[-200:])
            )
        return launch_wall_time

    def start_paused(self):
        cmd = self._build_cmd(start_paused=True)
        print("[MCAP] Starting PAUSED:")
        print(" ", " ".join(cmd))
        self._launch(cmd, use_pty=True)
        self.paused = True
        self.start_mode = "prepared_paused"

    def start_direct(self):
        """Start rosbag immediately without any timestamp-monitor preparation."""
        cmd = self._build_cmd(start_paused=False)
        print("[MCAP] Starting DIRECT recording (timestamp monitor disabled):")
        print(" ", " ".join(cmd))
        launch_wall_time = self._launch(cmd, use_pty=False)
        self.paused = False
        self.start_mode = "direct"
        self.recording_started_wall_time = launch_wall_time
        print("[MCAP] Direct recorder is ACTIVE; messages are being written.", flush=True)

    def is_alive(self):
        return self.proc is not None and self.proc.poll() is None

    def _send_space(self):
        if not self.is_alive():
            raise RuntimeError("ros2 bag record is not running")
        if self.pty_master_fd is None:
            raise RuntimeError("rosbag PTY is not initialized")
        os.write(self.pty_master_fd, b" ")
        time.sleep(float(self.args.bag_toggle_wait))

    def resume(self):
        if self.paused:
            self._send_space()
            self.paused = False
            self.recording_started_wall_time = time.time()
            print("[MCAP] Recorder RESUMED; messages are now being written.", flush=True)

    def pause(self):
        if self.is_alive() and not self.paused:
            if self.pty_master_fd is None:
                raise RuntimeError("Direct recorder cannot be paused because it has no PTY")
            self._send_space()
            self.paused = True
            print("[MCAP] Recorder PAUSED.", flush=True)

    def stop(self):
        if self.proc is None:
            return "".join(self.stderr_lines)

        print("[MCAP] Stopping ros2 bag record...")
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGINT)
        except ProcessLookupError:
            pass

        try:
            self.proc.wait(timeout=float(self.args.bag_stop_timeout))
        except subprocess.TimeoutExpired:
            print("[MCAP] ros2 bag did not stop after SIGINT; sending SIGTERM...")
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                print("[MCAP] ros2 bag still alive; sending SIGKILL...")
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.proc.wait(timeout=3.0)

        if self.stderr_thread is not None:
            self.stderr_thread.join(timeout=1.0)

        if self.pty_master_fd is not None:
            try:
                os.close(self.pty_master_fd)
            except OSError:
                pass
            self.pty_master_fd = None

        self.proc = None
        return "".join(self.stderr_lines)

def verify_start_paused_support():
    ros2 = shutil.which("ros2")
    if ros2 is None:
        raise RuntimeError("ros2 command not found. Please source /opt/ros/humble/setup.bash")
    result = subprocess.run(
        [ros2, "bag", "record", "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=10,
        check=False,
    )
    if "--start-paused" not in result.stdout:
        raise RuntimeError(
            "This ros2 bag record does not expose --start-paused. "
            "Run: ros2 bag record --help | grep start-paused"
        )
    print("[CHECK] ros2 bag record supports --start-paused")


def subscriber_baseline(node: Node, topics: list[str]):
    return {topic: int(node.count_subscribers(topic)) for topic in topics}


def wait_for_bag_subscribers(
    node: TimestampMonitorNode,
    recorder: McapRecorder,
    topics: list[str],
    baseline: dict[str, int],
    timeout_sec: float,
):
    if not topics:
        print("[SUBSCRIBERS] record-all-topics mode: skip exact per-topic subscriber matching")
        return True, {"skipped": True, "reason": "record_all_topics"}

    deadline = time.monotonic() + float(timeout_sec)
    next_log = 0.0
    last_counts = {}
    while rclpy.ok() and time.monotonic() < deadline:
        if not recorder.is_alive():
            return False, {
                "ok": False,
                "reason": "rosbag_exited_early",
                "stderr_tail": "".join(recorder.stderr_lines[-100:]),
            }

        rclpy.spin_once(node, timeout_sec=0.02)
        counts = {topic: int(node.count_subscribers(topic)) for topic in topics}
        last_counts = counts
        missing = [
            topic for topic in topics
            if counts.get(topic, 0) < int(baseline.get(topic, 0)) + 1
        ]
        now = time.monotonic()
        if not missing:
            print(f"[SUBSCRIBERS] All {len(topics)} rosbag subscriptions matched.", flush=True)
            return True, {
                "ok": True,
                "baseline": baseline,
                "current": counts,
                "topics": topics,
            }

        if now >= next_log:
            print(
                f"[SUBSCRIBERS] waiting: {len(missing)}/{len(topics)} not yet matched; "
                f"examples={missing[:5]}",
                flush=True,
            )
            next_log = now + 1.0

    missing = [
        topic for topic in topics
        if last_counts.get(topic, 0) < int(baseline.get(topic, 0)) + 1
    ]
    return False, {
        "ok": False,
        "reason": "subscriber_match_timeout",
        "timeout_sec": float(timeout_sec),
        "baseline": baseline,
        "current": last_counts,
        "missing_topics": missing,
    }


def write_episode_config(
    episode_dir: Path,
    args,
    episode_index: int,
    topics: list[str],
    episode_created_wall_time: float,
    recording_started_wall_time: float | None,
    stopped_wall_time: float | None,
    timestamp_report: dict | None,
    preparation_report: dict | None,
    aborted: bool,
):
    episode_dir.mkdir(parents=True, exist_ok=True)

    duration_sec = None
    if recording_started_wall_time is not None and stopped_wall_time is not None:
        duration_sec = float(stopped_wall_time - recording_started_wall_time)

    monitor_enabled = bool(args.enable_timestamp_monitor)
    monitor_keys = list(R1PRO_16D_CORE_TOPIC_KEYS) if monitor_enabled else []

    config = {
        "task": args.task,
        "task_id": args.task_id,
        "instruction_id": args.instruction_id,
        "instruction_text": args.instruction_text,
        "language": args.language,
        "operator_id": args.operator_id,
        "operator_order": OPERATOR_ORDER[args.operator_id],
        "schema": "r1pro_dual_arm_16d_left_first_v1",
        "safe_task_dir_name": safe_task_dir_name(args.task),
        "episode_index": episode_index,
        "aborted": aborted,
        "episode_created_at": time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(episode_created_wall_time)
        ),
        "recording_started_at": None if recording_started_wall_time is None else time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(recording_started_wall_time)
        ),
        "stopped_at": None if stopped_wall_time is None else time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(stopped_wall_time)
        ),
        "duration_sec": duration_sec,
        "root": str(Path(args.root).expanduser().resolve()),
        "episode_dir": str(Path(episode_dir).expanduser().resolve()),
        "bag_dir": str((episode_dir / "bag").expanduser().resolve()),
        "storage_id": args.storage_id,
        "topics_recorded": topics,
        "image_topics": IMAGE_TOPICS,
        "state_topics": STATE_TOPICS,
        "action_topics": ACTION_TOPICS,
        "r1pro_16d_core_topic_keys": R1PRO_16D_CORE_TOPIC_KEYS,
        "recording_mode": (
            "prepared_start_with_timestamp_monitor"
            if monitor_enabled
            else "direct_without_timestamp_monitor"
        ),
        "prepared_start": {
            "enabled": monitor_enabled,
            "start_paused": monitor_enabled,
            "monitor_qos_depth": 1 if monitor_enabled else None,
            "precheck_stable_sec": float(args.precheck_stable_sec) if monitor_enabled else None,
            "precheck_timeout": float(args.precheck_timeout) if monitor_enabled else None,
            "precheck_min_messages": int(args.precheck_min_messages) if monitor_enabled else None,
            "bag_subscriber_timeout": float(args.bag_subscriber_timeout) if monitor_enabled else None,
            "report": preparation_report,
        },
        "timestamp_monitor": {
            "enabled": monitor_enabled,
            "monitor_core_only": True if monitor_enabled else None,
            "monitor_keys": monitor_keys,
            "timestamp_max_span": float(args.timestamp_max_span) if monitor_enabled else None,
            "timestamp_max_age": float(args.timestamp_max_age) if monitor_enabled else None,
            "timestamp_check_every": float(args.timestamp_check_every) if monitor_enabled else None,
            "timestamp_grace": float(args.timestamp_grace) if monitor_enabled else None,
            "timestamp_use_arrival_fallback": (
                bool(args.timestamp_use_arrival_fallback) if monitor_enabled else None
            ),
            "abort_on_timestamp_error": (
                bool(args.abort_on_timestamp_error) if monitor_enabled else False
            ),
            "final_report": timestamp_report,
        },
        "args": {k: json_safe(v) for k, v in vars(args).items()},
    }

    (episode_dir / "record_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False)
    )
    (episode_dir / "task.txt").write_text(str(args.task) + "\n")

    if timestamp_report is not None:
        (episode_dir / "timestamp_report.json").write_text(
            json.dumps(timestamp_report, indent=2, ensure_ascii=False)
        )
    if preparation_report is not None:
        (episode_dir / "preparation_report.json").write_text(
            json.dumps(preparation_report, indent=2, ensure_ascii=False)
        )

def record_one_episode(
    node: TimestampMonitorNode | None,
    args,
    task_dir: Path,
    episode_index: int,
    topics: list[str],
):
    stamp = now_stamp_ms()
    episode_dir = task_dir / f"episode_{episode_index:06d}_{stamp}"
    bag_dir = episode_dir / "bag"

    print("\n" + "=" * 100)
    print(f"[EPISODE] {episode_index:06d}")
    print("task:", args.task)
    print("episode_dir:", episode_dir)
    print("=" * 100)

    monitor_enabled = bool(args.enable_timestamp_monitor)
    stop_requested = {"value": False, "cancel": False, "command": ""}

    def wait_for_stop_enter():
        print(
            color_text(
                "\n[MANUAL] Recording is READY and ACTIVE. You can start operating the robot now.\n"
                "  ENTER      STOP and SAVE this episode\n"
                "  q + ENTER  CANCEL and DELETE this episode\n",
                COLOR_BLUE,
            ),
            flush=True,
        )
        try:
            with open("/dev/tty", "r") as tty_in:
                try:
                    termios.tcflush(tty_in.fileno(), termios.TCIFLUSH)
                except Exception:
                    pass
                line = tty_in.readline()
                if line == "":
                    print("[MANUAL][WARN] /dev/tty returned EOF; ignore stop request.", flush=True)
                    return
                cmd = line.strip().lower()
        except Exception as exc:
            print(f"[MANUAL][WARN] Failed to read /dev/tty for stop input: {exc}", flush=True)
            return

        stop_requested["command"] = cmd
        if cmd in {"q", "quit", "exit", "c", "cancel"}:
            print(
                color_text(
                    "[MANUAL][CANCEL] User requested cancel. This episode will be deleted.",
                    COLOR_RED,
                ),
                flush=True,
            )
            stop_requested["cancel"] = True
            stop_requested["value"] = True
            return

        stop_requested["cancel"] = False
        stop_requested["value"] = True

    recorder = McapRecorder(args, bag_dir, topics)
    episode_created_wall_time = time.time()
    recording_started_wall_time = None
    stopped_wall_time = None
    final_report = None
    preparation_report = {
        "ok": False,
        "mode": (
            "prepared_start_with_timestamp_monitor"
            if monitor_enabled
            else "direct_without_timestamp_monitor"
        ),
        "monitor_keys": list(node.monitor_keys) if node is not None else [],
        "stages": {},
    }
    aborted = False
    stderr_text = ""
    formal_recording_started = False

    try:
        if monitor_enabled:
            if node is None:
                raise RuntimeError("Timestamp monitor is enabled but monitor node is not available")

            # Stage 1: 只接受本次准备阶段的新消息。
            print(f"[PRECHECK] Reset monitor state; waiting for {len(node.monitor_keys)} required topics...")
            node.reset_episode_state()
            pre_ok, pre_report = node.wait_until_stable(
                stable_sec=0.0,
                timeout_sec=float(args.precheck_timeout),
                min_messages=1,
                stage_name="PRECHECK",
            )
            preparation_report["stages"]["required_topics_present_before_bag"] = pre_report
            if not pre_ok:
                raise RuntimeError("Required monitored topics were not ready before starting rosbag")

            # Stage 2: 在 rosbag 启动前记录订阅者基线。
            baseline = subscriber_baseline(node, topics)
            preparation_report["stages"]["subscriber_baseline"] = baseline

            # Stage 3: rosbag 暂停启动。
            recorder.start_paused()

            # Stage 4: 等待 rosbag 订阅匹配。
            matched, subscriber_report = wait_for_bag_subscribers(
                node=node,
                recorder=recorder,
                topics=topics,
                baseline=baseline,
                timeout_sec=float(args.bag_subscriber_timeout),
            )
            preparation_report["stages"]["bag_subscribers_matched"] = subscriber_report
            if not matched:
                raise RuntimeError("rosbag subscriptions did not finish matching")

            # Stage 5: 在 rosbag 负载下连续稳定。
            node.reset_episode_state()
            print(
                f"[STABILIZE] rosbag is PAUSED. Requiring {len(node.monitor_keys)} topics "
                f"to remain healthy for {float(args.precheck_stable_sec):.1f}s..."
            )
            stable_ok, stable_report = node.wait_until_stable(
                stable_sec=float(args.precheck_stable_sec),
                timeout_sec=float(args.precheck_timeout),
                min_messages=int(args.precheck_min_messages),
                stage_name="STABILIZE",
            )
            preparation_report["stages"]["stable_under_rosbag_load"] = stable_report
            if not stable_ok:
                raise RuntimeError("Topics did not remain stable under rosbag load")

            # Stage 6: 恢复 rosbag，开始正式录制。
            recorder.resume()
            node.reset_episode_state()
            node.start_recording_monitor()
            recording_started_wall_time = (
                recorder.recording_started_wall_time or time.time()
            )
            preparation_report["stages"]["recording_started"] = {
                "start_mode": recorder.start_mode,
                "wall_time": recording_started_wall_time,
            }

        else:
            # 完全不创建时间戳监控订阅，也不做任何 precheck。
            print(
                color_text(
                    "[DIRECT] Timestamp monitor is DISABLED. "
                    "Starting rosbag directly without precheck/start-paused.",
                    COLOR_YELLOW,
                ),
                flush=True,
            )
            recorder.start_direct()
            recording_started_wall_time = (
                recorder.recording_started_wall_time or time.time()
            )
            preparation_report["stages"]["direct_start"] = {
                "start_mode": recorder.start_mode,
                "wall_time": recording_started_wall_time,
                "timestamp_monitor_enabled": False,
                "precheck_skipped": True,
                "subscriber_match_wait_skipped": True,
                "stabilization_skipped": True,
            }

        formal_recording_started = True
        preparation_report["ok"] = True
        preparation_report["recording_started_wall_time"] = recording_started_wall_time

        flush_stdin_buffer()
        stop_thread = threading.Thread(target=wait_for_stop_enter, daemon=True)
        stop_thread.start()

        if monitor_enabled:
            next_report_time = time.monotonic() + float(args.timestamp_report_every)
            while rclpy.ok() and not stop_requested["value"]:
                now = time.monotonic()
                rclpy.spin_once(node, timeout_sec=0.01)
                ok, status, report = node.check_timestamp_sync()
                final_report = report

                if status not in {"skip", "grace", "not_recording"} and now >= next_report_time:
                    node.print_timestamp_report()
                    next_report_time = now + float(args.timestamp_report_every)

                if not ok and args.abort_on_timestamp_error:
                    print(
                        color_text(
                            "[TS][ABORT] Timestamp monitor failed. "
                            "Aborting and deleting this episode.",
                            COLOR_RED,
                        )
                    )
                    print(
                        color_text(
                            json.dumps(report, indent=2, ensure_ascii=False),
                            COLOR_RED,
                        )
                    )
                    aborted = True
                    break
        else:
            # Direct mode: only watch whether rosbag process is alive.
            while not stop_requested["value"]:
                if not recorder.is_alive():
                    print(
                        color_text(
                            "[MCAP][ABORT] ros2 bag record exited unexpectedly.",
                            COLOR_RED,
                        ),
                        flush=True,
                    )
                    aborted = True
                    break
                time.sleep(0.02)

        if stop_requested.get("cancel", False):
            print(
                color_text(
                    "[MANUAL][CANCEL] Cancel current episode. "
                    "It will be deleted and episode index will not advance.",
                    COLOR_RED,
                )
            )
            aborted = True

    except Exception as exc:
        aborted = True
        preparation_report["ok"] = False
        preparation_report["failure"] = str(exc)
        print(color_text(f"[PREP][ABORT] {exc}", COLOR_RED), flush=True)

    finally:
        # Prepared-start 模式可先 pause；direct 模式没有 PTY，直接 SIGINT 收尾。
        if (
            monitor_enabled
            and recorder.is_alive()
            and formal_recording_started
        ):
            try:
                recorder.pause()
            except Exception as exc:
                print(f"[MCAP][WARN] Failed to pause before stop: {exc}")

        stderr_text = recorder.stop()
        stopped_wall_time = time.time()

        if monitor_enabled and formal_recording_started and node is not None:
            _, final_status, final_report = node.check_timestamp_sync(force=True)
            print("[TS][FINAL]", final_status)
            if final_report:
                print(json.dumps(final_report, indent=2, ensure_ascii=False))
        elif not monitor_enabled and formal_recording_started:
            final_status = "disabled"
            final_report = None
            print("[TS][FINAL] disabled (direct recording mode)")
        else:
            final_status = "preparation_failed"
            final_report = {
                "ok": False,
                "reasons": ["preparation_failed"],
                "preparation": preparation_report,
            }
            print("[TS][FINAL] preparation_failed")

        if final_report is not None and stop_requested.get("cancel", False):
            final_report["manual_cancel"] = True
            final_report["cancel_command"] = stop_requested.get("command", "")

        write_episode_config(
            episode_dir=episode_dir,
            args=args,
            episode_index=episode_index,
            topics=topics,
            episode_created_wall_time=episode_created_wall_time,
            recording_started_wall_time=recording_started_wall_time,
            stopped_wall_time=stopped_wall_time,
            timestamp_report=final_report,
            preparation_report=preparation_report,
            aborted=aborted,
        )

        if stderr_text:
            (episode_dir / "ros2_bag_stderr.txt").write_text(stderr_text)

        if aborted and args.delete_aborted:
            try:
                shutil.rmtree(episode_dir)
                print(color_text(f"[ABORT] Deleted: {episode_dir}", COLOR_RED))
            except FileNotFoundError:
                pass
            return False

        mcap_files = sorted(bag_dir.glob("*.mcap"))
        metadata_file = bag_dir / "metadata.yaml"
        if not metadata_file.exists():
            print("[WARN] metadata.yaml not found:", metadata_file)
        if not mcap_files:
            print("[WARN] no .mcap files found in:", bag_dir)
        else:
            print(color_text("[MCAP] Saved files:", COLOR_GREEN))
            for file_path in mcap_files:
                print(color_text(f"  {file_path}", COLOR_GREEN))

    return not aborted

def run_manual_loop(args):
    root = Path(args.root).expanduser().resolve()
    task_dir = task_directory(root, args)
    task_dir.mkdir(parents=True, exist_ok=True)

    topics = r1pro_16d_core_topics()

    monitor_enabled = bool(args.enable_timestamp_monitor)
    node = None
    rclpy_initialized = False

    if monitor_enabled:
        verify_start_paused_support()
        rclpy.init()
        rclpy_initialized = True
        node = TimestampMonitorNode(args)
        recorder_title = "R1Pro MCAP Prepared-start Episode Recorder"
    else:
        recorder_title = "R1Pro MCAP Direct Episode Recorder (No Timestamp Monitor)"

    print("=" * 100)
    print(recorder_title)
    print("=" * 100)
    print("root:", root)
    print("task:", args.task)
    print("operator_id:", args.operator_id)
    print("task_dir:", task_dir)
    print("storage_id:", args.storage_id)
    print("recording_profile: r1pro_dual_arm_16d_left_first_v1")
    print("topics_count:", len(topics))
    print("enable_timestamp_monitor:", monitor_enabled)

    if monitor_enabled:
        print("monitor_topics_count:", len(node.monitor_keys))
        print("monitor_qos_depth: 1")
        print("precheck_stable_sec:", args.precheck_stable_sec)
        print("timestamp_max_span:", args.timestamp_max_span)
        print("timestamp_max_age:", args.timestamp_max_age)
        print("abort_on_timestamp_error:", args.abort_on_timestamp_error)
        print("start_mode: --start-paused -> subscriber match -> stabilize -> resume")
    else:
        print("monitor_topics_count: 0")
        print("abort_on_timestamp_error: ignored")
        print("precheck/subscriber-match/stabilize: disabled")
        print("start_mode: direct ros2 bag record")
    print("=" * 100)

    print("[TOPICS] Will record:")
    for topic in topics:
        print(" ", topic)

    if monitor_enabled:
        print("[MONITOR] Will monitor:")
        for key in node.monitor_keys:
            print(f"  {key}: {node.topic_key_to_topic[key]}")
    else:
        print("[MONITOR] DISABLED: no extra timestamp-monitor subscriptions will be created.")

    try:
        if args.episode_index == "auto":
            episode_index = prompt_start_episode_index(task_dir)
        else:
            try:
                episode_index = require_available_episode_index(
                    task_dir, int(args.episode_index)
                )
            except (TypeError, ValueError) as exc:
                print(color_text(f"[EPISODE INDEX] {exc}", COLOR_RED))
                episode_index = prompt_start_episode_index(task_dir)

        while True:
            if monitor_enabled and not rclpy.ok():
                break

            existing = episode_directories_for_index(task_dir, episode_index)
            if existing:
                print(color_text(
                    f"[EPISODE INDEX] episode_{episode_index:06d} already exists: "
                    + ", ".join(existing),
                    COLOR_RED,
                ))
                episode_index = prompt_start_episode_index(task_dir)

            action_word = "PREPARE" if monitor_enabled else "START DIRECT RECORDING"
            cmd = input(
                color_text(
                    f"\n[MANUAL] Ready. Press ENTER to {action_word} "
                    f"episode_{episode_index:06d}; type 1/2/3 to switch instruction; "
                    f"or q to quit: ",
                    COLOR_YELLOW,
                )
            ).strip().lower()

            if cmd in {"q", "quit", "exit"}:
                print("[MANUAL] Quit requested.")
                break

            if cmd in INSTRUCTION_ALIASES:
                spec = apply_instruction(args, cmd)
                task_dir = task_directory(root, args)
                task_dir.mkdir(parents=True, exist_ok=True)
                episode_index = prompt_start_episode_index(task_dir)
                print(color_text(
                    f"[MANUAL] Switched to {spec.instruction_id}: {spec.text}",
                    COLOR_GREEN,
                ))
                continue

            success = record_one_episode(
                node=node,
                args=args,
                task_dir=task_dir,
                episode_index=episode_index,
                topics=topics,
            )

            if success:
                episode_index += 1
                print(color_text("[MANUAL] Episode saved. Ready for next episode.", COLOR_GREEN))
            else:
                print(color_text("[MANUAL] Episode aborted. Episode index not advanced; retry allowed.", COLOR_RED))

    finally:
        if node is not None:
            node.destroy_node()
        if rclpy_initialized and rclpy.ok():
            rclpy.shutdown()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "datasets/r1pro_mcap_raw",
        help="Root directory. Bags are saved under <root>/<safe_task>/episode_xxxxxx_timestamp/bag",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Generic task instruction. Mutually exclusive with --instruction-id.",
    )
    parser.add_argument(
        "--instruction-id",
        help="Plate/cup instruction: 1, 2, 3, or stack_plates_cups_01..03.",
    )
    parser.add_argument(
        "--operator",
        dest="operator_id",
        help="Recording operator: li, liu, yu, or xing. Prompted when omitted.",
    )
    parser.add_argument("--episode-index", type=str, default="auto")

    parser.add_argument("--storage-id", type=str, default="mcap")
    parser.add_argument(
        "--bag-startup-wait",
        type=float,
        default=0.5,
        help=("Process launch wait. In direct mode, rosbag is already writing during this wait; "
              "in monitored mode, readiness is still determined by subscriber matching."),
    )
    parser.add_argument("--bag-toggle-wait", type=float, default=0.25)
    parser.add_argument("--bag-stop-timeout", type=float, default=10.0)
    parser.add_argument("--bag-subscriber-timeout", type=float, default=15.0)

    parser.add_argument(
        "--enable-timestamp-monitor",
        type=str2bool,
        default=True,
        help=(
            "True: enable the 11-topic timestamp monitor and prepared-start logic. "
            "False: do not create monitor subscriptions; start rosbag directly."
        ),
    )
    parser.add_argument(
        "--direct-recording",
        action="store_true",
        help=(
            "Start rosbag directly without topic readiness, subscriber matching, "
            "or timestamp monitoring. Equivalent to --enable-timestamp-monitor false."
        ),
    )

    parser.add_argument(
        "--timestamp-max-span",
        type=float,
        default=0.35,
        help="Max allowed span in seconds across latest monitored topic timestamps.",
    )
    parser.add_argument(
        "--timestamp-max-age",
        type=float,
        default=0.75,
        help="Max allowed age in seconds since latest arrival for every monitored topic.",
    )
    parser.add_argument("--timestamp-check-every", type=float, default=0.2)
    parser.add_argument("--timestamp-report-every", type=float, default=2.0)
    parser.add_argument("--timestamp-grace", type=float, default=2.0)
    parser.add_argument(
        "--timestamp-use-arrival-fallback",
        type=str2bool,
        default=True,
        help="If some topics lack valid header.stamp, compare arrival times instead.",
    )
    parser.add_argument("--abort-on-timestamp-error", type=str2bool, default=True)
    parser.add_argument("--delete-aborted", type=str2bool, default=True)

    parser.add_argument(
        "--precheck-stable-sec",
        type=float,
        default=2.0,
        help="Continuous healthy duration required after rosbag subscriptions are matched.",
    )
    parser.add_argument(
        "--precheck-timeout",
        type=float,
        default=15.0,
        help="Timeout for topic-presence and post-rosbag stability checks.",
    )
    parser.add_argument(
        "--precheck-min-messages",
        type=int,
        default=3,
        help="Minimum fresh messages required from every monitored topic during stabilization.",
    )

    # Kept only for compatibility with older launch commands.
    parser.add_argument("--warmup-timeout", type=float, default=10.0)

    args = parser.parse_args()
    apply_recording_mode(args)
    if args.operator_id:
        try:
            args.operator_id = resolve_operator(args.operator_id)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        if not sys.stdin.isatty():
            parser.error("provide --operator with one of: li, liu, yu, xing")
        print("Select recording operator: li, liu, yu, or xing")
        while True:
            try:
                args.operator_id = resolve_operator(input("operator> "))
                break
            except ValueError as exc:
                print(color_text(f"[MANUAL] {exc}", COLOR_RED))

    if args.task and args.instruction_id:
        parser.error("--task and --instruction-id are mutually exclusive")
    if args.instruction_id:
        try:
            apply_instruction(args, args.instruction_id)
        except ValueError as exc:
            parser.error(str(exc))
    elif args.task:
        args.task_id = safe_task_dir_name(args.task)
        args.instruction_id = None
        args.instruction_text = args.task
        args.language = "unspecified"
    else:
        if not sys.stdin.isatty():
            parser.error("provide --task or --instruction-id")
        print("Select plate/cup instruction: 1, 2, or 3")
        try:
            apply_instruction(args, input("instruction> ").strip())
        except ValueError as exc:
            parser.error(str(exc))
    if args.enable_timestamp_monitor:
        if args.precheck_stable_sec < 0:
            parser.error("--precheck-stable-sec must be >= 0")
        if args.precheck_timeout <= 0:
            parser.error("--precheck-timeout must be > 0")
        if args.precheck_min_messages < 1:
            parser.error("--precheck-min-messages must be >= 1")

    run_manual_loop(args)


if __name__ == "__main__":
    main()
