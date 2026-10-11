#!/usr/bin/env python3
"""
###############
把mcap转换成中间数据：
（1）fps=15
cd /home/lixiang/PROJECT/LingBotVLA2.0
source /opt/ros/humble/setup.bash

/usr/bin/python3 /home/lixiang/PROJECT/LingBotVLA2.0/myscripts/16d/convert_tools/export_r1pro_mcap_to_intermediate.py \
  --input /home/lixiang/PROJECT/LingBotVLA2.0/datasets/r1pro_mcap_raw/pick_place \
  --output /home/lixiang/PROJECT/LingBotVLA2.0/datasets/r1pro_intermediate_16d/pick_place \
  --fps 15 \
  --max-image-dt 0.12 \
  --max-vec-dt 0.20 \
  --overwrite

0.12和0.20参数是否合适：
转换后的intermediate显示：mcap帧和intermediate帧相比，后者损失了例如：
MCAP cam_high 总帧数：40110
intermediate 保留帧数：40040
丢弃帧数：70
整体丢帧率：0.1745%

（2）fps=30
cd /home/lixiang/PROJECT/LingBotVLA2.0
source /opt/ros/humble/setup.bash

/usr/bin/python3 /home/lixiang/PROJECT/LingBotVLA2.0/myscripts/16d/convert_tools/export_r1pro_mcap_to_intermediate.py \
  --input /home/lixiang/PROJECT/LingBotVLA2.0/datasets/r1pro_mcap_raw/hang_screwdriver_on_tool_rack_30hz \
  --output /home/lixiang/PROJECT/LingBotVLA2.0/datasets/r1pro_intermediate_16d/hang_screwdriver_on_tool_rack_30hz \
  --fps 30 \
  --max-image-dt 0.05 \
  --max-vec-dt 0.06 \
  --overwrite


0.05和0.06参数是否合适：
转换后的intermediate显示：mcap帧和intermediate帧相比，后者损失了例如：
MCAP cam_high 总帧数：40110
intermediate 保留帧数：40040
丢弃帧数：70
整体丢帧率：0.1745%

结论：
所以指令参数可以一直使用。

#################


Export R1Pro rosbag2/MCAP episodes to an intermediate dataset.

This script MUST run with ROS system Python:

    source /opt/ros/humble/setup.bash
    /usr/bin/python3 export_r1pro_mcap_to_intermediate.py ...

Input:
    r1pro_mcap_raw/<task_dir>/episode_xxx/
        bag/
            metadata.yaml
            bag_0.mcap
        task.txt
        record_config.json
        timestamp_report.json

Output:
    r1pro_intermediate/<task_dir>/episode_xxx/
        task.txt
        episode_meta.json
        timestamps.npy
        state.npy          shape [N, 16]
        action.npy         shape [N, 16]
        images/
            cam_high/
                000000.jpg
                ...
            cam_left_wrist/
            cam_right_wrist/

State/action order:
    left_arm7, left_gripper1, right_arm7, right_gripper1

Image mapping:
    cam_high        <- /hdas/camera_head/left_raw/image_raw_color/compressed
    cam_left_wrist  <- /hdas/camera_wrist_left/color/image_raw/compressed
    cam_right_wrist <- /hdas/camera_wrist_right/color/image_raw/compressed
"""

import argparse
import bisect
import importlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_schema = importlib.import_module("myscripts.16d.common.schema")
ACTION_KEYS = _schema.ACTION_KEYS
ACTION_NAMES = _schema.ACTION_NAMES
CAM_KEYS = _schema.CAM_KEYS
SCHEMA_NAME = _schema.SCHEMA_NAME
STATE_KEYS = _schema.STATE_KEYS
STATE_NAMES = _schema.STATE_NAMES
TOPICS = _schema.TOPICS
assemble_action = _schema.assemble_action
assemble_state = _schema.assemble_state
validate_vector_array = _schema.validate_vector_array

# ======================================================================================
# Terminal colors: only affects printed text, not conversion logic.
# ======================================================================================
COLOR_YELLOW = "\033[93m"
COLOR_RESET = "\033[0m"


def color_text(text, color):
    return f"{color}{text}{COLOR_RESET}"


import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


# cam_high is the reference stream, so converter rejection attribution covers
# the two matched cameras plus every state/action component.
DETAILED_SKIP_KEYS = [
    "cam_left_wrist",
    "cam_right_wrist",
    *STATE_KEYS,
    *ACTION_KEYS,
]

STATE_ORDER = list(STATE_NAMES)
ACTION_ORDER = list(ACTION_NAMES)


def normalize_task_label(task: str) -> str:
    """Convert task text to a stable underscore label.

    Examples:
      "pick up the turtle" -> "pick_up_the_turtle"
      "put the cup into the box" -> "put_the_cup_into_the_box"
    """
    text = str(task).strip().lower()
    out = []
    for ch in text:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    text = "".join(out)
    while "__" in text:
        text = text.replace("__", "_")
    text = text.strip("_")
    return text or "unknown_task"


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def msg_time_sec(msg, fallback_rosbag_time_ns: int) -> float:
    if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
        t = stamp_to_sec(msg.header.stamp)
        if t > 0:
            return t
    return float(fallback_rosbag_time_ns) * 1e-9


def image_ext_from_format(fmt: str) -> str:
    fmt = (fmt or "").lower()
    if "png" in fmt:
        return ".png"
    return ".jpg"


def extract_compressed_image_payload(msg):
    """
    Keep the original compressed image bytes.
    This avoids requiring cv2 in the ROS system Python.
    """
    ext = image_ext_from_format(getattr(msg, "format", "jpeg"))
    data = bytes(msg.data)
    if len(data) == 0:
        raise RuntimeError("CompressedImage data is empty")
    return {"bytes": data, "ext": ext}


def _array_from_value(value):
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size > 0:
            return arr
    except Exception:
        return None
    return None


def extract_numeric_vector(msg) -> np.ndarray:
    """
    Generic numeric extractor.

    Expected to work with:
      sensor_msgs/msg/JointState.position
      std_msgs/msg/Float64.data
      std_msgs/msg/Float32.data
      common custom message fields such as position, data, qpos, joint_position, etc.
    """
    candidate_fields = [
        "position",
        "positions",
        "joint_position",
        "joint_positions",
        "qpos",
        "pos",
        "data",
        "value",
        "values",
        "command",
        "target",
        "current_position",
        "current",
    ]

    for name in candidate_fields:
        if hasattr(msg, name):
            arr = _array_from_value(getattr(msg, name))
            if arr is not None:
                return arr

    nested_fields = [
        "joint_state",
        "state",
        "feedback",
        "status",
        "msg",
    ]

    for name in nested_fields:
        if hasattr(msg, name):
            sub = getattr(msg, name)
            try:
                return extract_numeric_vector(sub)
            except Exception:
                pass

    raise RuntimeError(f"Cannot extract numeric vector from message type: {type(msg)}")


def extract_arm7(msg) -> np.ndarray:
    arr = extract_numeric_vector(msg)
    if arr.size < 7:
        raise RuntimeError(f"Expected at least 7 arm values, got {arr.size}; msg type={type(msg)}")
    return arr[:7].astype(np.float32)


def extract_scalar(msg) -> np.float32:
    arr = extract_numeric_vector(msg)
    if arr.size < 1:
        raise RuntimeError(f"Expected scalar value, got empty; msg type={type(msg)}")
    return np.float32(arr[0])


class Series:
    def __init__(self, items):
        items = sorted(items, key=lambda x: x[0])
        self.times = [x[0] for x in items]
        self.values = [x[1] for x in items]

    def __len__(self):
        return len(self.times)

    def nearest(self, t: float, max_dt: float):
        if not self.times:
            return None

        i = bisect.bisect_left(self.times, t)
        candidates = []

        if i < len(self.times):
            candidates.append(i)
        if i > 0:
            candidates.append(i - 1)

        best_i = min(candidates, key=lambda idx: abs(self.times[idx] - t))
        if abs(self.times[best_i] - t) > max_dt:
            return None

        return self.values[best_i]


def missing_nearest_keys(series, keys, t: float, max_dt: float) -> list[str]:
    """Return every component key that has no sample close enough to t."""
    return [key for key in keys if series[key].nearest(t, max_dt) is None]


def timing_stats_from_times(times) -> dict:
    """Compact per-topic inter-arrival statistics for offline auditing."""
    times = np.asarray(times, dtype=np.float64)
    count = int(times.size)
    if count <= 1:
        return {
            "message_count": count,
            "min_dt": None,
            "mean_dt": None,
            "p95_dt": None,
            "p99_dt": None,
            "max_dt": None,
            "nonpositive_dt_count": 0,
        }

    dt = np.diff(times)
    finite_dt = dt[np.isfinite(dt)]
    if finite_dt.size == 0:
        return {
            "message_count": count,
            "min_dt": None,
            "mean_dt": None,
            "p95_dt": None,
            "p99_dt": None,
            "max_dt": None,
            "nonpositive_dt_count": 0,
        }

    return {
        "message_count": count,
        "min_dt": float(np.min(finite_dt)),
        "mean_dt": float(np.mean(finite_dt)),
        "p95_dt": float(np.percentile(finite_dt, 95)),
        "p99_dt": float(np.percentile(finite_dt, 99)),
        "max_dt": float(np.max(finite_dt)),
        "nonpositive_dt_count": int(np.sum(finite_dt <= 0)),
    }


def topic_timing_stats(item_series: Series) -> dict:
    return timing_stats_from_times(item_series.times)


def open_rosbag_reader(bag_dir: Path):
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    return reader


def read_bag_episode(bag_dir: Path):
    reader = open_rosbag_reader(bag_dir)

    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    required_topics = set(TOPICS.values())
    missing_topics = sorted(required_topics - set(topic_types.keys()))

    if missing_topics:
        raise RuntimeError(
            "Bag is missing required topics:\n"
            + "\n".join(f"  {x}" for x in missing_topics)
        )

    msg_classes = {}
    for topic in required_topics:
        msg_type = topic_types[topic]
        try:
            msg_classes[topic] = get_message(msg_type)
        except Exception as e:
            raise RuntimeError(
                f"Cannot load message type for topic:\n"
                f"  topic: {topic}\n"
                f"  type : {msg_type}\n"
                f"Reason: {e}\n\n"
                f"If this is a custom Galaxea/HDAS message type, the host needs the corresponding ROS2 interface package."
            )

    topic_to_key = {topic: key for key, topic in TOPICS.items()}
    raw_data = {key: [] for key in TOPICS.keys()}
    # Collect rosbag timestamps for every recorded topic, including optional
    # EE pose, IMU, chassis, and the fourth camera.
    all_topic_rosbag_times = {}

    count_all = 0
    count_used = 0

    while reader.has_next():
        topic, raw, rosbag_time_ns = reader.read_next()
        count_all += 1
        all_topic_rosbag_times.setdefault(topic, []).append(
            float(rosbag_time_ns) * 1e-9
        )

        if topic not in topic_to_key:
            continue

        key = topic_to_key[topic]
        msg = deserialize_message(raw, msg_classes[topic])
        t = msg_time_sec(msg, rosbag_time_ns)

        if key in CAM_KEYS:
            value = extract_compressed_image_payload(msg)
        elif "arm" in key:
            value = extract_arm7(msg)
        elif "gripper" in key:
            value = extract_scalar(msg)
        else:
            raise RuntimeError(f"Unhandled key: {key}")

        raw_data[key].append((t, value))
        count_used += 1

    series = {key: Series(items) for key, items in raw_data.items()}

    time_alignment_fixes = {}

    summary = {
        "bag_total_messages_read": count_all,
        "bag_required_messages_used": count_used,
        "topic_counts_used": {key: len(series[key]) for key in series},
        "topic_types": {topic: topic_types[topic] for topic in required_topics},
        "time_alignment_fixes": time_alignment_fixes,
        "topic_timing_stats": {
            key: topic_timing_stats(series[key])
            for key in sorted(series.keys())
        },
        "all_recorded_topic_timing_stats": {
            topic: timing_stats_from_times(times)
            for topic, times in sorted(all_topic_rosbag_times.items())
        },
    }

    return series, summary


def build_state(series, t: float, max_vec_dt: float):
    left_arm = series["state_left_arm"].nearest(t, max_vec_dt)
    left_gripper = series["state_left_gripper"].nearest(t, max_vec_dt)
    right_arm = series["state_right_arm"].nearest(t, max_vec_dt)
    right_gripper = series["state_right_gripper"].nearest(t, max_vec_dt)
    if left_arm is None or left_gripper is None or right_arm is None or right_gripper is None:
        return None
    return assemble_state(
        left_arm=left_arm,
        left_gripper=np.asarray([left_gripper]),
        right_arm=right_arm,
        right_gripper=np.asarray([right_gripper]),
    )


def build_action(series, t: float, max_vec_dt: float):
    left_arm = series["action_left_arm"].nearest(t, max_vec_dt)
    left_gripper = series["action_left_gripper"].nearest(t, max_vec_dt)
    right_arm = series["action_right_arm"].nearest(t, max_vec_dt)
    right_gripper = series["action_right_gripper"].nearest(t, max_vec_dt)
    if left_arm is None or left_gripper is None or right_arm is None or right_gripper is None:
        return None
    return assemble_action(
        left_arm=left_arm,
        left_gripper=np.asarray([left_gripper]),
        right_arm=right_arm,
        right_gripper=np.asarray([right_gripper]),
    )


def find_episode_dirs(input_path: Path):
    input_path = input_path.expanduser().resolve()

    if (input_path / "bag").exists():
        return [input_path]

    eps = sorted([p for p in input_path.glob("episode_*") if (p / "bag").exists()])
    if eps:
        return eps

    eps = sorted([p for p in input_path.glob("*/*") if p.name.startswith("episode_") and (p / "bag").exists()])
    if eps:
        return eps

    raise RuntimeError(f"No episode_*/bag directories found under: {input_path}")


def read_task(episode_dir: Path) -> str:
    task_file = episode_dir / "task.txt"
    if task_file.exists():
        task = task_file.read_text().strip()
        if task:
            return normalize_task_label(task)

    # fallback: try record_config.json
    cfg_file = episode_dir / "record_config.json"
    if cfg_file.exists():
        try:
            cfg = json.loads(cfg_file.read_text())
            task = str(cfg.get("task", "")).strip()
            if task:
                return normalize_task_label(task)
        except Exception:
            pass

    return "unknown_task"


def write_image_payload(path: Path, payload):
    path.write_bytes(payload["bytes"])


def convert_episode(episode_dir: Path, output_episode_dir: Path, args):
    bag_dir = episode_dir / "bag"
    task = read_task(episode_dir)

    if output_episode_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_episode_dir)
        else:
            raise RuntimeError(f"Output episode dir already exists: {output_episode_dir}")

    output_episode_dir.mkdir(parents=True, exist_ok=True)

    for cam in CAM_KEYS:
        (output_episode_dir / "images" / cam).mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print(f"[EPISODE] {episode_dir.name}")
    print(f"  input bag : {bag_dir}")
    print(f"  output    : {output_episode_dir}")
    print(f"  task      : {task}")

    series, bag_summary = read_bag_episode(bag_dir)

    ref_times = series["cam_high"].times
    ref_values = series["cam_high"].values

    timestamps = []
    states = []
    actions = []

    skipped = {
        "cam_left_wrist": 0,
        "cam_right_wrist": 0,
        "state": 0,
        "action": 0,
    }
    skipped_by_topic = {key: 0 for key in DETAILED_SKIP_KEYS}
    skip_events = []

    kept = 0

    for ref_idx, (t, cam_high_payload) in enumerate(zip(ref_times, ref_values)):
        cam_left_payload = series["cam_left_wrist"].nearest(t, args.max_image_dt)
        if cam_left_payload is None:
            skipped["cam_left_wrist"] += 1
            skipped_by_topic["cam_left_wrist"] += 1
            skip_events.append({
                "reference_frame_index": int(ref_idx),
                "reference_timestamp": float(t),
                "stage": "image",
                "missing_keys": ["cam_left_wrist"],
                "missing_topics": [TOPICS["cam_left_wrist"]],
            })
            continue

        cam_right_payload = series["cam_right_wrist"].nearest(t, args.max_image_dt)
        if cam_right_payload is None:
            skipped["cam_right_wrist"] += 1
            skipped_by_topic["cam_right_wrist"] += 1
            skip_events.append({
                "reference_frame_index": int(ref_idx),
                "reference_timestamp": float(t),
                "stage": "image",
                "missing_keys": ["cam_right_wrist"],
                "missing_topics": [TOPICS["cam_right_wrist"]],
            })
            continue

        missing_state_keys = missing_nearest_keys(
            series, STATE_KEYS, t, args.max_vec_dt
        )
        if missing_state_keys:
            skipped["state"] += 1
            for key in missing_state_keys:
                skipped_by_topic[key] += 1
            skip_events.append({
                "reference_frame_index": int(ref_idx),
                "reference_timestamp": float(t),
                "stage": "state",
                "missing_keys": missing_state_keys,
                "missing_topics": [TOPICS[key] for key in missing_state_keys],
            })
            continue

        state = build_state(series, t, args.max_vec_dt)
        if state is None:
            skipped["state"] += 1
            # Defensive fallback; normally missing_state_keys catches this.
            skip_events.append({
                "reference_frame_index": int(ref_idx),
                "reference_timestamp": float(t),
                "stage": "state",
                "missing_keys": ["unknown_state_component"],
                "missing_topics": [],
            })
            continue

        missing_action_keys = missing_nearest_keys(
            series, ACTION_KEYS, t, args.max_vec_dt
        )
        if missing_action_keys:
            skipped["action"] += 1
            for key in missing_action_keys:
                skipped_by_topic[key] += 1
            skip_events.append({
                "reference_frame_index": int(ref_idx),
                "reference_timestamp": float(t),
                "stage": "action",
                "missing_keys": missing_action_keys,
                "missing_topics": [TOPICS[key] for key in missing_action_keys],
            })
            continue

        action = build_action(series, t, args.max_vec_dt)
        if action is None:
            skipped["action"] += 1
            # Defensive fallback; normally missing_action_keys catches this.
            skip_events.append({
                "reference_frame_index": int(ref_idx),
                "reference_timestamp": float(t),
                "stage": "action",
                "missing_keys": ["unknown_action_component"],
                "missing_topics": [],
            })
            continue

        frame_name = f"{kept:06d}"
        write_image_payload(output_episode_dir / "images" / "cam_high" / f"{frame_name}{cam_high_payload['ext']}", cam_high_payload)
        write_image_payload(output_episode_dir / "images" / "cam_left_wrist" / f"{frame_name}{cam_left_payload['ext']}", cam_left_payload)
        write_image_payload(output_episode_dir / "images" / "cam_right_wrist" / f"{frame_name}{cam_right_payload['ext']}", cam_right_payload)

        timestamps.append(t)
        states.append(state)
        actions.append(action)

        kept += 1

    if kept == 0:
        raise RuntimeError(f"No valid frames exported for episode: {episode_dir}")

    timestamps = np.asarray(timestamps, dtype=np.float64)
    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    validate_vector_array(states, name="state", episode=episode_dir.name)
    validate_vector_array(actions, name="action", episode=episode_dir.name)

    np.save(output_episode_dir / "timestamps.npy", timestamps)
    np.save(output_episode_dir / "state.npy", states)
    np.save(output_episode_dir / "action.npy", actions)
    (output_episode_dir / "task.txt").write_text(task + "\n")

    skip_events_path = output_episode_dir / "skip_events.jsonl"
    with skip_events_path.open("w") as f:
        for event in skip_events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    meta = {
        "source_episode_dir": str(episode_dir),
        "source_bag_dir": str(bag_dir),
        "task": task,
        "schema": SCHEMA_NAME,
        "frame_count": int(kept),
        "reference_camera": "cam_high",
        "reference_topic": TOPICS["cam_high"],
        "fps_target": args.fps,
        "max_image_dt": args.max_image_dt,
        "max_vec_dt": args.max_vec_dt,
        "state_shape": list(states.shape),
        "action_shape": list(actions.shape),
        "state_order": STATE_ORDER,
        "action_order": ACTION_ORDER,
        "topics": TOPICS,
        "skipped": skipped,
        "skipped_by_topic": skipped_by_topic,
        "skip_event_count": int(len(skip_events)),
        "skip_events_file": skip_events_path.name,
        "skip_attribution_note": "One rejected reference frame may be attributed to multiple missing state/action topics.",
        "bag_summary": bag_summary,
    }

    (output_episode_dir / "episode_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False)
    )

    print(f"  kept frames: {kept}")
    print(f"  skipped    : {skipped}")
    nonzero_skipped_by_topic = {
        key: value for key, value in skipped_by_topic.items() if int(value) > 0
    }
    print(f"  skipped by topic: {nonzero_skipped_by_topic}")
    print(f"  skip events file: {skip_events_path}")
    print(f"  state shape: {states.shape}")
    print(f"  action shape: {actions.shape}")
    print("[OK]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Task directory, root directory, or a single episode directory.")
    parser.add_argument("--output", required=True, help="Intermediate output directory.")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--max-image-dt", type=float, default=0.12)
    parser.add_argument("--max-vec-dt", type=float, default=0.20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()

    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    episode_dirs = find_episode_dirs(input_path)
    print(f"[INFO] Found {len(episode_dirs)} episodes")

    manifest = {
        "input": str(input_path),
        "output": str(output_root),
        "fps": args.fps,
        "max_image_dt": args.max_image_dt,
        "max_vec_dt": args.max_vec_dt,
        "episodes": [],
    }

    for episode_dir in episode_dirs:
        output_episode_dir = output_root / episode_dir.name
        convert_episode(episode_dir, output_episode_dir, args)
        manifest["episodes"].append(
            {
                "source": str(episode_dir),
                "intermediate": str(output_episode_dir),
            }
        )

    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print("=" * 100)
    print("[DONE] MCAP -> intermediate finished")
    print(f"Output: {output_root}")


if __name__ == "__main__":
    main()
