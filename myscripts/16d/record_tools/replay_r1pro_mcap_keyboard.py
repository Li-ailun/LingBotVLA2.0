#!/usr/bin/env python3
"""Keyboard replay for raw R1Pro rosbag2/MCAP camera episodes."""

from __future__ import annotations

import argparse
import bisect
from datetime import datetime
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


OPERATORS = ("li", "liu", "yu", "xing")
INSTRUCTION_ALIASES = {
    "1": "stack_plates_cups_01",
    "2": "stack_plates_cups_02",
    "3": "stack_plates_cups_03",
    "stack_plates_cups_01": "stack_plates_cups_01",
    "stack_plates_cups_02": "stack_plates_cups_02",
    "stack_plates_cups_03": "stack_plates_cups_03",
}
CAMERA_TOPICS = {
    "head": "/hdas/camera_head/left_raw/image_raw_color/compressed",
    "left_wrist": "/hdas/camera_wrist_left/color/image_raw/compressed",
    "right_wrist": "/hdas/camera_wrist_right/color/image_raw/compressed",
}
CAMERA_LABELS = {
    "head": "HEAD",
    "left_wrist": "LEFT WRIST",
    "right_wrist": "RIGHT WRIST",
}
EPISODE_RE = re.compile(r"^episode_(\d+)(?:_|$)")
WINDOW_NAME = "R1Pro raw MCAP replay"


@dataclass(frozen=True)
class Episode:
    episode_index: int
    episode_dir: Path
    bag_dir: Path
    mcap_files: tuple[Path, ...]


@dataclass(frozen=True)
class ImageSample:
    timestamp: float
    payload: bytes


@dataclass(frozen=True)
class CameraStream:
    samples: tuple[ImageSample, ...]
    timestamps: tuple[float, ...]


def resolve_operator(value: str) -> str:
    operator = str(value).strip().lower()
    if operator not in OPERATORS:
        raise ValueError(f"operator must be one of: {', '.join(OPERATORS)}")
    return operator


def resolve_instruction(value: str) -> str:
    instruction = INSTRUCTION_ALIASES.get(str(value).strip().lower())
    if instruction is None:
        raise ValueError("instruction must be 1, 2, or 3")
    return instruction


def selected_task_directory(root: Path, operator: str, instruction: str) -> Path:
    return (
        Path(root).expanduser().resolve()
        / "stack_plates_and_cups"
        / resolve_operator(operator)
        / resolve_instruction(instruction)
    )


def episode_index_from_name(name: str) -> int:
    match = EPISODE_RE.match(name)
    if not match:
        raise ValueError(f"invalid episode directory name: {name}")
    return int(match.group(1))


def discover_episodes(task_dir: Path) -> list[Episode]:
    task_dir = Path(task_dir)
    if not task_dir.is_dir():
        raise FileNotFoundError(f"task directory does not exist: {task_dir}")

    episodes = []
    for episode_dir in sorted(task_dir.glob("episode_*")):
        if not episode_dir.is_dir():
            continue
        try:
            episode_index = episode_index_from_name(episode_dir.name)
        except ValueError:
            continue
        bag_dir = episode_dir / "bag"
        mcap_files = tuple(sorted(bag_dir.glob("*.mcap")))
        if not mcap_files:
            raise RuntimeError(f"episode_{episode_index:06d} is missing MCAP: {bag_dir}")
        if not (bag_dir / "metadata.yaml").is_file():
            raise RuntimeError(
                f"episode_{episode_index:06d} is missing metadata.yaml: {bag_dir}"
            )
        episodes.append(Episode(episode_index, episode_dir, bag_dir, mcap_files))

    if not episodes:
        raise FileNotFoundError(f"no episode directories found under: {task_dir}")

    episodes.sort(key=lambda episode: (episode.episode_index, episode.episode_dir.name))
    duplicates = {}
    for episode in episodes:
        duplicates.setdefault(episode.episode_index, []).append(episode.episode_dir)
    duplicates = {key: value for key, value in duplicates.items() if len(value) > 1}
    if duplicates:
        detail = "\n".join(
            f"  episode_{index:06d}: " + ", ".join(str(path) for path in paths)
            for index, paths in sorted(duplicates.items())
        )
        raise RuntimeError(f"duplicate MCAP episode indices:\n{detail}")
    return episodes


def quarantine_episode(
    dataset_root: Path,
    episode: Episode,
    operator: str,
    instruction: str,
    *,
    deleted_at: str | None = None,
) -> Path:
    """Move one raw episode out of the active dataset and record the move."""
    dataset_root = Path(dataset_root).expanduser().resolve()
    source = episode.episode_dir.resolve()
    try:
        source.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError(f"episode is outside dataset root: {source}") from exc

    operator = resolve_operator(operator)
    instruction = resolve_instruction(instruction)
    deleted_at = deleted_at or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    quarantine_root = dataset_root / "_deleted_episodes"
    destination = (
        quarantine_root / deleted_at / operator / instruction / source.name
    )
    if destination.exists():
        raise FileExistsError(
            f"quarantine destination already exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))

    record = {
        "action": "quarantine",
        "deleted_at": deleted_at,
        "episode_index": episode.episode_index,
        "instruction_id": instruction,
        "operator_id": operator,
        "original_path": str(source),
        "quarantine_path": str(destination.resolve()),
    }
    manifest = quarantine_root / "deletion_manifest.jsonl"
    with manifest.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return destination


def open_reader(bag_dir: Path):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    return reader


def message_timestamp(message, rosbag_time_ns: int) -> float:
    if hasattr(message, "header") and hasattr(message.header, "stamp"):
        stamp = message.header.stamp
        value = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        if value > 0:
            return value
    return float(rosbag_time_ns) * 1e-9


def load_camera_streams(episode: Episode) -> dict[str, CameraStream]:
    print(f"[LOAD] {episode.episode_dir}", flush=True)
    reader = open_reader(episode.bag_dir)
    topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    missing = sorted(set(CAMERA_TOPICS.values()) - set(topic_types))
    if missing:
        raise RuntimeError("bag is missing camera topics:\n" + "\n".join(missing))

    topic_to_key = {topic: key for key, topic in CAMERA_TOPICS.items()}
    message_classes = {
        topic: get_message(topic_types[topic]) for topic in topic_to_key
    }
    items: dict[str, list[ImageSample]] = {key: [] for key in CAMERA_TOPICS}

    while reader.has_next():
        topic, raw, rosbag_time_ns = reader.read_next()
        key = topic_to_key.get(topic)
        if key is None:
            continue
        message = deserialize_message(raw, message_classes[topic])
        payload = bytes(message.data)
        if not payload:
            raise RuntimeError(f"empty compressed image on {topic}")
        items[key].append(
            ImageSample(message_timestamp(message, rosbag_time_ns), payload)
        )

    streams = {}
    for key, samples in items.items():
        if not samples:
            raise RuntimeError(f"camera topic has no frames: {CAMERA_TOPICS[key]}")
        samples.sort(key=lambda sample: sample.timestamp)
        frozen = tuple(samples)
        streams[key] = CameraStream(
            samples=frozen,
            timestamps=tuple(sample.timestamp for sample in frozen),
        )
        print(f"[LOAD] {key}: {len(frozen)} frames", flush=True)
    return streams


def nearest_sample(stream: CameraStream, timestamp: float) -> tuple[ImageSample, float]:
    pos = bisect.bisect_left(stream.timestamps, timestamp)
    choices = []
    if pos < len(stream.samples):
        choices.append(stream.samples[pos])
    if pos > 0:
        choices.append(stream.samples[pos - 1])
    sample = min(choices, key=lambda item: abs(item.timestamp - timestamp))
    return sample, abs(sample.timestamp - timestamp)


def decode_image(sample: ImageSample, *, key: str, frame_index: int) -> np.ndarray:
    encoded = np.frombuffer(sample.payload, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to decode {key} frame {frame_index}")
    return image


def resize_to_height(image: np.ndarray, height: int = 360) -> np.ndarray:
    scale = height / image.shape[0]
    width = max(1, int(round(image.shape[1] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def label_image(image: np.ndarray, label: str) -> np.ndarray:
    image = image.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        image, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        (255, 255, 255), 2, cv2.LINE_AA,
    )
    return image


def render_frame(
    streams: dict[str, CameraStream],
    reference_index: int,
    episode: Episode,
    operator: str,
    instruction: str,
    rate: float,
    max_image_dt: float,
) -> np.ndarray:
    reference = streams["head"].samples[reference_index]
    images = []
    sync_parts = []
    for key in ("head", "left_wrist", "right_wrist"):
        sample, delta = nearest_sample(streams[key], reference.timestamp)
        if delta > max_image_dt:
            raise RuntimeError(
                f"{key} has no frame within {max_image_dt:.3f}s at "
                f"head frame {reference_index}; nearest dt={delta:.3f}s"
            )
        image = resize_to_height(decode_image(sample, key=key, frame_index=reference_index))
        images.append(label_image(image, f"{CAMERA_LABELS[key]}  dt={delta:.3f}s"))
        sync_parts.append(f"{key}={delta:.3f}s")

    min_width = min(image.shape[1] for image in images)
    images = [
        cv2.resize(image, (min_width, image.shape[0]), interpolation=cv2.INTER_AREA)
        for image in images
    ]
    canvas = cv2.hconcat(images)
    elapsed = reference.timestamp - streams["head"].samples[0].timestamp
    duration = (
        streams["head"].samples[-1].timestamp
        - streams["head"].samples[0].timestamp
    )
    footer = np.zeros((72, canvas.shape[1], 3), dtype=np.uint8)
    line1 = (
        f"operator={operator}  task={instruction}  "
        f"episode={episode.episode_index:06d}  frame={reference_index + 1}/"
        f"{len(streams['head'].samples)}"
    )
    line2 = (
        f"time={elapsed:.2f}/{duration:.2f}s  rate={rate:g}x  "
        "SPACE/p pause  n next  b previous  r replay  d remove  1/2/3 speed  q quit"
    )
    cv2.putText(footer, line1, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(footer, line2, (10, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 255, 200), 1, cv2.LINE_AA)
    return cv2.vconcat([canvas, footer])


def render_delete_confirmation(frame: np.ndarray, episode: Episode) -> np.ndarray:
    frame = frame.copy()
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 120), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    center_x = frame.shape[1] // 2
    center_y = frame.shape[0] // 2
    lines = (
        "REMOVE FROM ACTIVE DATASET?",
        f"episode_{episode.episode_index:06d}",
        "ENTER: confirm and move to _deleted_episodes",
        "ESC: cancel",
    )
    for offset, line in zip((-75, -20, 40, 85), lines):
        scale = 1.0 if offset < 0 else 0.7
        thickness = 3 if offset < 0 else 2
        size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
        cv2.putText(
            frame,
            line,
            (max(10, center_x - size[0] // 2), center_y + offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return frame


def key_action(key: int) -> str | None:
    if key in (ord("q"), 27):
        return "quit"
    if key == ord("n"):
        return "next"
    if key == ord("b"):
        return "previous"
    if key == ord("r"):
        return "replay"
    return None


def play_episode(
    episode: Episode,
    operator: str,
    instruction: str,
    rate: float,
    max_image_dt: float,
) -> tuple[str, float]:
    streams = load_camera_streams(episode)
    head = streams["head"].samples
    paused = False
    delete_armed = False
    paused_before_delete = False
    index = 0
    print(
        f"[PLAY] operator={operator} task={instruction} "
        f"episode_{episode.episode_index:06d} frames={len(head)} rate={rate:g}x",
        flush=True,
    )

    while index < len(head):
        frame = render_frame(
            streams, index, episode, operator, instruction, rate, max_image_dt
        )
        if delete_armed:
            frame = render_delete_confirmation(frame, episode)
        cv2.imshow(WINDOW_NAME, frame)

        if index + 1 < len(head):
            frame_dt = max(0.001, head[index + 1].timestamp - head[index].timestamp)
        else:
            frame_dt = 1.0 / 15.0
        delay_ms = max(1, int(round(frame_dt / rate * 1000.0)))
        key = cv2.waitKey(50 if paused or delete_armed else delay_ms) & 0xFF

        if delete_armed:
            if key in (10, 13):
                return "delete", rate
            if key == 27:
                delete_armed = False
                paused = paused_before_delete
            continue

        action = key_action(key)
        if action:
            return action, rate
        if key in (ord("p"), ord(" ")):
            paused = not paused
            continue
        if key == ord("d"):
            paused_before_delete = paused
            paused = True
            delete_armed = True
            print(
                f"[DELETE ARMED] episode_{episode.episode_index:06d}; "
                "press ENTER to move it to _deleted_episodes, ESC to cancel.",
                flush=True,
            )
            continue
        if key == ord("1"):
            rate = 0.5
        elif key == ord("2"):
            rate = 1.0
        elif key == ord("3"):
            rate = 2.0

        if not paused:
            index += 1

    return "next", rate


def prompt_choice(prompt: str, resolver):
    while True:
        try:
            return resolver(input(prompt))
        except ValueError as exc:
            print(f"[INPUT] {exc}")


def choose_start_position(episodes: list[Episode], requested: int | None) -> int:
    by_index = {episode.episode_index: pos for pos, episode in enumerate(episodes)}
    if requested is not None:
        if requested not in by_index:
            raise ValueError(f"episode_{requested:06d} not found")
        return by_index[requested]

    first = episodes[0].episode_index
    while True:
        raw = input(f"Start episode index [first: {first}]: ").strip()
        if raw == "":
            return 0
        try:
            index = int(raw)
        except ValueError:
            print("[INPUT] Enter an existing non-negative episode index.")
            continue
        if index in by_index:
            return by_index[index]
        print(f"[INPUT] episode_{index:06d} not found in this selection.")


def print_episode_summary(episodes: list[Episode], task_dir: Path) -> None:
    indices = [episode.episode_index for episode in episodes]
    print("=" * 88)
    print("task_dir:", task_dir)
    print("episodes:", len(episodes))
    print("index range:", f"{min(indices)}..{max(indices)}")
    print("indices:")
    for start in range(0, len(indices), 20):
        print(" ", " ".join(f"{index:06d}" for index in indices[start:start + 20]))
    print("=" * 88)


def parse_args():
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Replay three synchronized camera views directly from R1Pro MCAP."
    )
    parser.add_argument(
        "--root", type=Path,
        default=project_root / "datasets/r1pro_mcap_raw",
        help="Raw MCAP root containing stack_plates_and_cups.",
    )
    parser.add_argument("--operator", choices=OPERATORS)
    parser.add_argument("--instruction-id", help="Task 1, 2, or 3.")
    parser.add_argument("--episode-index", type=int)
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--max-image-dt", type=float, default=0.12)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()
    if args.rate <= 0:
        parser.error("--rate must be > 0")
    if args.max_image_dt <= 0:
        parser.error("--max-image-dt must be > 0")
    if args.episode_index is not None and args.episode_index < 0:
        parser.error("--episode-index must be >= 0")
    return args


def main() -> int:
    args = parse_args()
    operator = args.operator or prompt_choice(
        "operator [li/liu/yu/xing]> ", resolve_operator
    )
    instruction = (
        resolve_instruction(args.instruction_id)
        if args.instruction_id
        else prompt_choice("instruction [1/2/3]> ", resolve_instruction)
    )
    task_dir = selected_task_directory(args.root, operator, instruction)
    episodes = discover_episodes(task_dir)
    print_episode_summary(episodes, task_dir)
    if args.list_only:
        return 0

    position = choose_start_position(episodes, args.episode_index)
    rate = float(args.rate)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        while 0 <= position < len(episodes):
            action, rate = play_episode(
                episodes[position], operator, instruction, rate, args.max_image_dt
            )
            if action == "quit":
                break
            if action == "delete":
                removed = episodes[position]
                destination = quarantine_episode(
                    args.root, removed, operator, instruction
                )
                print(
                    f"[REMOVED] episode_{removed.episode_index:06d}\n"
                    f"  from: {removed.episode_dir}\n"
                    f"  to:   {destination}",
                    flush=True,
                )
                episodes.pop(position)
                if not episodes:
                    print("[DONE] No active episodes remain in this selection.")
                    break
                if position >= len(episodes):
                    position = len(episodes) - 1
                continue
            if action == "previous":
                position = max(0, position - 1)
            elif action == "replay":
                pass
            else:
                position += 1
        if position >= len(episodes):
            print("[DONE] Reached the last episode.")
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)
