#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R1Pro MCAP -> intermediate -> LeRobot v2.1 missing/timestamp/mapping audit.

检查 raw MCAP source_episode_dir 是否存在
检查 raw MCAP → intermediate → LeRobot episode 序号对应
检查 manifest source 是否和 episode_meta.json 对得上
检查 converter skipped frames 及原因
检查 intermediate state/action/timestamps/images 数量一致性
检查 intermediate 真实 cam_high timestamps.npy 的 gap
检查 LeRobot parquet 数量、帧数、state/action 是否和 intermediate 完全一致
输出 raw_episode_delete_candidates.txt

Example:

（1）fps=15
python /home/lixiang/PROJECT/LingBotVLA2.0/myscripts/16d/lerobot_tools/check_lerobot_v21_missing_audit.py \
  --raw-root /home/lixiang/PROJECT/LingBotVLA2.0/datasets/r1pro_mcap_raw/put_pen_and_tape_into_pen_holder \
  --inter-root /home/lixiang/PROJECT/LingBotVLA2.0/datasets/r1pro_intermediate_16d/put_pen_and_tape_into_pen_holder \
  --lerobot-root /home/lixiang/PROJECT/LingBotVLA2.0/datasets/lerobot/lixiang/put_pen_and_tape_into_pen_holder \
  --out-dir /home/lixiang/PROJECT/LingBotVLA2.0/reports/lerobot_quality_reports/put_pen_and_tape_into_pen_holder_missing_audit \
  --gap-warn-threshold 0.08 \
  --gap-bad-threshold 0.12 \
  --gap-severe-threshold 0.15 \
  --delete-candidate-policy bad_only \
  --top 80

（2）fps=30  
python /home/lixiang/PROJECT/LingBotVLA2.0/myscripts/16d/lerobot_tools/check_lerobot_v21_missing_audit.py \
  --raw-root /home/lixiang/PROJECT/LingBotVLA2.0/datasets/r1pro_mcap_raw/hang_screwdriver_on_tool_rack_30hz \
  --inter-root /home/lixiang/PROJECT/LingBotVLA2.0/datasets/r1pro_intermediate_16d/hang_screwdriver_on_tool_rack_30hz \
  --lerobot-root /home/lixiang/PROJECT/LingBotVLA2.0/datasets/lerobot/lixiang/hang_screwdriver_on_tool_rack_30hz \
  --out-dir /home/lixiang/PROJECT/LingBotVLA2.0/reports/lerobot_quality_reports/hang_screwdriver_on_tool_rack_30hz_missing_audit \
  --gap-warn-threshold 0.05 \
  --gap-bad-threshold 0.065 \
  --gap-severe-threshold 0.10 \
  --delete-candidate-policy bad_only \
  --top 80

  
输出：
missing_audit_summary.csv
missing_audit_bad_or_warn.csv
timestamp_gap_events.csv
raw_to_lerobot_episode_mapping.csv
raw_episode_delete_candidates.txt
missing_audit_summary.json

其中最重要的：
1. missing_audit_summary.csv
这是最核心的总表，每一行对应一个 LeRobot episode，同时记录它对应的 raw MCAP episode 和 intermediate episode。

2. missing_audit_summary.json
全局汇总

3. timestamp_gap_events.csv
episode 出现 dt > 0.08 的真实 cam_high 时间戳间隔

4. raw_to_lerobot_episode_mapping.csv
这个表专门记录：LeRobot episode 编号 ↔ raw MCAP episode 编号

5. missing_audit_bad_or_warn.csv
这个文件只保存 异常 episode 的详细信息。

2. raw_episode_delete_candidates.txt 
这个文件是从 missing_audit_bad_or_warn.csv 里面提取出来的 raw MCAP episode 路径列表。
方便你直接批量移动或删除有问题的 raw MCAP episode。

"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_RAW_ROOT = Path("/home/lixiang/PROJECT/LingBotVLA2.0/datasets/r1pro_mcap_raw/hang_wire_stripper_on_wall_hook")
DEFAULT_INTER_ROOT = Path("/home/lixiang/PROJECT/LingBotVLA2.0/datasets/r1pro_intermediate_16d/hang_wire_stripper_on_wall_hook")
DEFAULT_LEROBOT_ROOT = Path("/home/lixiang/PROJECT/LingBotVLA2.0/datasets/lerobot/lixiang/hang_wire_stripper_on_wall_hook")
DEFAULT_OUT_DIR = Path("/home/lixiang/PROJECT/LingBotVLA2.0/reports/lerobot_quality_reports/hang_wire_stripper_on_wall_hook_missing_audit")

# Detailed converter attribution keys. New converter metadata stores per-topic
# matching failures under episode_meta.json["skipped_by_topic"].
DETAILED_SKIP_KEYS = [
    "cam_left_wrist",
    "cam_right_wrist",
    "state_left_arm",
    "state_right_arm",
    "state_left_gripper",
    "state_right_gripper",
    "action_left_arm",
    "action_right_arm",
    "action_left_gripper",
    "action_right_gripper",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit MCAP/intermediate/LeRobot missing frames, timestamp gaps, and episode mapping.")
    p.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT, help=f"Raw MCAP task root. Default: {DEFAULT_RAW_ROOT}")
    p.add_argument("--inter-root", type=Path, default=DEFAULT_INTER_ROOT, help=f"Intermediate root. Default: {DEFAULT_INTER_ROOT}")
    p.add_argument("--lerobot-root", type=Path, default=DEFAULT_LEROBOT_ROOT, help=f"LeRobot root. Default: {DEFAULT_LEROBOT_ROOT}")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help=f"Output report directory. Default: {DEFAULT_OUT_DIR}")
    p.add_argument("--fps", type=float, default=15.0, help="Expected reference camera fps. Default: 15")
    p.add_argument("--gap-warn-threshold", type=float, default=0.08)
    p.add_argument("--gap-bad-threshold", type=float, default=0.12)
    p.add_argument("--gap-severe-threshold", type=float, default=0.15)
    p.add_argument("--value-threshold", type=float, default=1e-6, help="Allowed max abs error between intermediate and LeRobot state/action")
    p.add_argument("--expected-dim", type=int, default=16, help="Expected state/action dim; fixed to 16 for this toolchain.")
    p.add_argument("--delete-candidate-policy", choices=["warn_or_bad", "bad_only", "gap_bad_or_converter_skip"], default="warn_or_bad")
    p.add_argument("--top", type=int, default=80)
    return p.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def safe_np_load(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.load(path)


def count_images(cam_dir: Path) -> int:
    if not cam_dir.exists():
        return 0
    return len(sorted(list(cam_dir.glob("*.jpg")) + list(cam_dir.glob("*.png"))))


def episode_index_from_parquet(path: Path) -> int:
    try:
        return int(path.stem.split("_")[1])
    except Exception as exc:
        raise ValueError(f"Cannot parse episode index from parquet filename: {path}") from exc


def raw_number_from_episode_name(name: str) -> int | None:
    # episode_000148_202606... -> 148
    parts = name.split("_")
    if len(parts) >= 2 and parts[0] == "episode" and parts[1].isdigit():
        return int(parts[1])
    return None


def array_from_lerobot_column(series: pd.Series) -> np.ndarray:
    return np.stack(series.map(lambda x: np.asarray(x, dtype=np.float32)).to_numpy())


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def infer_lerobot_dim(lerobot_root: Path, key: str) -> int | None:
    info_path = lerobot_root / "meta" / "info.json"
    info = load_json(info_path)
    feat = info.get("features", {}).get(key, {})
    shape = feat.get("shape")
    if isinstance(shape, list) and shape:
        try:
            return int(shape[0])
        except Exception:
            pass
    names = feat.get("names")
    if isinstance(names, list):
        return len(names)
    return None


def compact_tail(xs, n: int = 4) -> str:
    if not isinstance(xs, list):
        return ""
    return "|".join(str(x) for x in xs[-n:])


def load_manifest(inter_root: Path) -> dict[str, dict[str, str]]:
    manifest_path = inter_root / "manifest.json"
    manifest = load_json(manifest_path)
    out: dict[str, dict[str, str]] = {}
    for item in manifest.get("episodes", []) or []:
        inter = Path(str(item.get("intermediate", ""))).name
        source = str(item.get("source", ""))
        if inter:
            out[inter] = {"source": source, "intermediate": str(item.get("intermediate", ""))}
    return out


def list_raw_episode_dirs(raw_root: Path) -> list[Path]:
    if not raw_root.exists():
        return []
    return sorted([p for p in raw_root.glob("episode_*") if p.is_dir()])


def make_status(hard_bad: list[str], warn: list[str]) -> str:
    if hard_bad:
        return "BAD"
    if warn:
        return "WARN"
    return "OK"


def classify_missing_reason(skipped_total: int, raw_cam_high_count: Any, inter_frames: int, gap_bad_count: int) -> str:
    reasons = []
    if skipped_total > 0:
        reasons.append("converter_dropped_reference_frames_due_to_missing_matched_modality")
    if isinstance(raw_cam_high_count, int) and inter_frames >= 0:
        drop = raw_cam_high_count - inter_frames
        if drop > 0 and skipped_total == 0:
            reasons.append("raw_to_intermediate_frame_count_drop_not_explained_by_skipped_metadata")
        elif drop > 0 and skipped_total > 0:
            reasons.append("raw_to_intermediate_drop_matches_converter_skip")
    if gap_bad_count > 0 and skipped_total == 0:
        reasons.append("source_cam_high_timestamp_gap_or_raw_camera_drop")
    elif gap_bad_count > 0 and skipped_total > 0:
        reasons.append("timestamp_gap_after_converter_skip")
    return "|".join(reasons)


def should_delete_candidate(row: dict[str, Any], policy: str) -> bool:
    if policy == "bad_only":
        return row.get("status") == "BAD"
    if policy == "gap_bad_or_converter_skip":
        return int(row.get("skipped_total", 0)) > 0 or int(row.get("gap_bad_count", 0)) > 0 or row.get("status") == "BAD"
    return row.get("status") in {"WARN", "BAD"}


def audit(args: argparse.Namespace) -> None:
    raw_root = args.raw_root.expanduser().resolve()
    inter_root = args.inter_root.expanduser().resolve()
    lerobot_root = args.lerobot_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not inter_root.exists():
        raise FileNotFoundError(f"Intermediate root not found: {inter_root}")
    if not lerobot_root.exists():
        raise FileNotFoundError(f"LeRobot root not found: {lerobot_root}")

    expected_dim = args.expected_dim
    if expected_dim is None:
        state_dim_meta = infer_lerobot_dim(lerobot_root, "observation.state")
        action_dim_meta = infer_lerobot_dim(lerobot_root, "action")
        if state_dim_meta is not None and action_dim_meta is not None and state_dim_meta == action_dim_meta:
            expected_dim = state_dim_meta

    summary_csv = out_dir / "missing_audit_summary.csv"
    bad_csv = out_dir / "missing_audit_bad_or_warn.csv"
    gap_csv = out_dir / "timestamp_gap_events.csv"
    mapping_csv = out_dir / "raw_to_lerobot_episode_mapping.csv"
    delete_candidates_txt = out_dir / "raw_episode_delete_candidates.txt"
    summary_json = out_dir / "missing_audit_summary.json"
    skipped_topic_by_episode_csv = out_dir / "skipped_topic_by_episode.csv"
    skipped_topic_summary_csv = out_dir / "skipped_topic_summary.csv"
    raw_topic_timing_by_episode_csv = out_dir / "raw_topic_timing_by_episode.csv"
    raw_topic_timing_summary_csv = out_dir / "raw_topic_timing_summary.csv"

    manifest_by_inter = load_manifest(inter_root)
    raw_dirs = list_raw_episode_dirs(raw_root)
    raw_names_existing = {p.name for p in raw_dirs}

    episode_dirs = sorted([p for p in inter_root.glob("episode_*") if p.is_dir()])
    parquet_files = sorted(lerobot_root.glob("data/chunk-*/*.parquet"))
    parquet_by_idx = {episode_index_from_parquet(p): p for p in parquet_files}

    rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    skipped_topic_rows: list[dict[str, Any]] = []
    raw_topic_timing_rows: list[dict[str, Any]] = []

    for lr_idx, ep in enumerate(episode_dirs):
        meta_path = ep / "episode_meta.json"
        meta = load_json(meta_path)
        skipped = meta.get("skipped", {}) or {}
        skipped_total = sum(int(v) for v in skipped.values()) if skipped else 0
        skipped_by_topic = meta.get("skipped_by_topic", {}) or {}
        detailed_skip_available = bool(skipped_by_topic)
        detailed_topic_skip_total = sum(int(v) for v in skipped_by_topic.values()) if skipped_by_topic else 0
        bag_summary = meta.get("bag_summary", {}) or {}
        topic_timing_stats = bag_summary.get("topic_timing_stats", {}) or {}
        all_recorded_topic_timing_stats = (
            bag_summary.get("all_recorded_topic_timing_stats", {}) or {}
        )

        hard_bad: list[str] = []
        warn: list[str] = []
        if not meta_path.exists():
            warn.append("missing_episode_meta_json")

        source_episode_dir = str(meta.get("source_episode_dir", ""))
        source_bag_dir = str(meta.get("source_bag_dir", ""))
        raw_episode_name = Path(source_episode_dir).name if source_episode_dir else ""
        raw_episode_number = raw_number_from_episode_name(raw_episode_name) if raw_episode_name else None
        raw_source_exists = bool(source_episode_dir) and Path(source_episode_dir).exists()
        raw_bag_exists = bool(source_bag_dir) and Path(source_bag_dir).exists()

        # Per-topic converter rejection attribution. A single rejected reference
        # frame can be attributed to multiple missing state/action topics, so
        # detailed_topic_skip_total may be greater than skipped_total.
        for topic_key in DETAILED_SKIP_KEYS:
            skip_count = int(skipped_by_topic.get(topic_key, 0))
            if skip_count > 0:
                skipped_topic_rows.append({
                    "lerobot_episode_index": lr_idx,
                    "lerobot_episode_file": f"episode_{lr_idx:06d}.parquet",
                    "raw_episode_name": raw_episode_name,
                    "raw_episode_number": raw_episode_number if raw_episode_number is not None else "",
                    "intermediate_episode": ep.name,
                    "topic_key": topic_key,
                    "ros_topic": (meta.get("topics", {}) or {}).get(topic_key, ""),
                    "skipped_reference_frames_attributed": skip_count,
                    "aggregate_skipped_total_episode": skipped_total,
                    "detailed_topic_skip_total_episode": detailed_topic_skip_total,
                    "source_episode_dir": source_episode_dir,
                    "source_bag_dir": source_bag_dir,
                })

        # Raw MCAP per-topic inter-arrival timing statistics generated by the
        # patched converter. Prefer all recorded ROS topics (including EE pose,
        # IMU, chassis, etc.); fall back to the 13 conversion topics for older
        # detailed metadata.
        logical_topics = meta.get("topics", {}) or {}
        reverse_logical_topics = {str(v): str(k) for k, v in logical_topics.items()}
        if all_recorded_topic_timing_stats:
            timing_items = [
                (reverse_logical_topics.get(str(ros_topic), str(ros_topic)), str(ros_topic), timing, "all_recorded_rosbag_time")
                for ros_topic, timing in sorted(all_recorded_topic_timing_stats.items())
            ]
        else:
            timing_items = [
                (str(topic_key), str(logical_topics.get(topic_key, "")), timing, "conversion_topic_time")
                for topic_key, timing in sorted(topic_timing_stats.items())
            ]

        for topic_key, ros_topic, timing, timing_source in timing_items:
            if not isinstance(timing, dict):
                continue
            max_topic_dt = timing.get("max_dt", "")
            try:
                max_topic_dt_float = float(max_topic_dt)
            except (TypeError, ValueError):
                max_topic_dt_float = None
            raw_topic_timing_rows.append({
                "lerobot_episode_index": lr_idx,
                "lerobot_episode_file": f"episode_{lr_idx:06d}.parquet",
                "raw_episode_name": raw_episode_name,
                "raw_episode_number": raw_episode_number if raw_episode_number is not None else "",
                "intermediate_episode": ep.name,
                "topic_key": topic_key,
                "ros_topic": ros_topic,
                "timing_source": timing_source,
                "message_count": int(timing.get("message_count", 0) or 0),
                "min_dt": timing.get("min_dt", ""),
                "mean_dt": timing.get("mean_dt", ""),
                "p95_dt": timing.get("p95_dt", ""),
                "p99_dt": timing.get("p99_dt", ""),
                "max_dt": timing.get("max_dt", ""),
                "nonpositive_dt_count": int(timing.get("nonpositive_dt_count", 0) or 0),
                "max_dt_gt_warn": bool(max_topic_dt_float is not None and max_topic_dt_float > args.gap_warn_threshold),
                "max_dt_gt_bad": bool(max_topic_dt_float is not None and max_topic_dt_float > args.gap_bad_threshold),
                "max_dt_gt_severe": bool(max_topic_dt_float is not None and max_topic_dt_float > args.gap_severe_threshold),
                "source_episode_dir": source_episode_dir,
                "source_bag_dir": source_bag_dir,
            })

        if source_episode_dir and not raw_source_exists:
            warn.append("source_raw_episode_dir_missing_or_moved")
        if source_bag_dir and not raw_bag_exists:
            warn.append("source_raw_bag_dir_missing_or_moved")

        manifest_source = manifest_by_inter.get(ep.name, {}).get("source", "")
        manifest_source_name = Path(manifest_source).name if manifest_source else ""
        if manifest_source and raw_episode_name and manifest_source_name != raw_episode_name:
            hard_bad.append("manifest_source_mismatch")

        state = safe_np_load(ep / "state.npy")
        action = safe_np_load(ep / "action.npy")
        ts = safe_np_load(ep / "timestamps.npy")
        if ts is not None:
            ts = ts.astype(float)

        if state is None:
            hard_bad.append("missing_intermediate_state_npy")
        if action is None:
            hard_bad.append("missing_intermediate_action_npy")
        if ts is None:
            hard_bad.append("missing_intermediate_timestamps_npy")

        state_len = len(state) if state is not None else -1
        action_len = len(action) if action is not None else -1
        ts_len = len(ts) if ts is not None else -1
        state_dim = int(state.shape[1]) if state is not None and state.ndim == 2 else -1
        action_dim = int(action.shape[1]) if action is not None and action.ndim == 2 else -1
        meta_state_shape = meta.get("state_shape", "")
        meta_action_shape = meta.get("action_shape", "")
        state_order = meta.get("state_order", []) or []
        action_order = meta.get("action_order", []) or []
        if expected_dim is not None:
            if state_dim != int(expected_dim):
                hard_bad.append(f"intermediate_state_dim_not_{expected_dim}")
            if action_dim != int(expected_dim):
                hard_bad.append(f"intermediate_action_dim_not_{expected_dim}")
        if isinstance(state_order, list) and state_dim > 0 and len(state_order) != state_dim:
            warn.append("state_order_length_mismatch")
        if isinstance(action_order, list) and action_dim > 0 and len(action_order) != action_dim:
            warn.append("action_order_length_mismatch")
        if state_len != action_len or state_len != ts_len:
            hard_bad.append("intermediate_state_action_timestamp_length_mismatch")

        image_counts = {
            "cam_high": count_images(ep / "images" / "cam_high"),
            "cam_left_wrist": count_images(ep / "images" / "cam_left_wrist"),
            "cam_right_wrist": count_images(ep / "images" / "cam_right_wrist"),
        }
        expected_len = state_len if state_len >= 0 else 0
        for cam_name, cam_count in image_counts.items():
            if cam_count != expected_len:
                hard_bad.append(f"{cam_name}_image_count_mismatch")

        topic_counts = meta.get("bag_summary", {}).get("topic_counts_used", {}) or {}
        raw_cam_high_count = topic_counts.get("cam_high", "")
        inferred_raw_to_inter_drop = ""
        if isinstance(raw_cam_high_count, int) and state_len >= 0:
            inferred_raw_to_inter_drop = raw_cam_high_count - state_len

        pq = parquet_by_idx.get(lr_idx)
        lerobot_frames = -1
        max_state_err: float | None = None
        max_action_err: float | None = None
        lr_state_shape = ""
        lr_action_shape = ""
        lerobot_timestamp_max_gap = ""
        lerobot_timestamp_min_gap = ""

        if pq is None:
            hard_bad.append("missing_lerobot_parquet")
        else:
            df = pd.read_parquet(pq)
            lerobot_frames = len(df)
            if lerobot_frames != state_len:
                hard_bad.append("lerobot_frame_count_mismatch")
            if "timestamp" in df.columns and len(df) > 1:
                lts = np.asarray(df["timestamp"].to_list(), dtype=np.float64)
                ldt = np.diff(lts)
                lerobot_timestamp_max_gap = float(ldt.max())
                lerobot_timestamp_min_gap = float(ldt.min())
                if np.any(ldt <= 0):
                    hard_bad.append("lerobot_timestamp_non_monotonic")
            if "observation.state" not in df.columns:
                hard_bad.append("missing_lerobot_observation_state_column")
            elif state is not None and len(df) == state_len:
                lr_state = array_from_lerobot_column(df["observation.state"])
                lr_state_shape = str(tuple(lr_state.shape))
                if lr_state.shape != state.shape:
                    hard_bad.append("state_shape_mismatch")
                else:
                    max_state_err = float(np.max(np.abs(lr_state - state.astype(np.float32)))) if len(state) else 0.0
                    if max_state_err > args.value_threshold:
                        hard_bad.append("state_value_mismatch")
            if "action" not in df.columns:
                hard_bad.append("missing_lerobot_action_column")
            elif action is not None and len(df) == action_len:
                lr_action = array_from_lerobot_column(df["action"])
                lr_action_shape = str(tuple(lr_action.shape))
                if lr_action.shape != action.shape:
                    hard_bad.append("action_shape_mismatch")
                else:
                    max_action_err = float(np.max(np.abs(lr_action - action.astype(np.float32)))) if len(action) else 0.0
                    if max_action_err > args.value_threshold:
                        hard_bad.append("action_value_mismatch")

        max_dt = 0.0
        min_dt = 0.0
        mean_dt = 0.0
        p95_dt = 0.0
        p99_dt = 0.0
        gap_warn_count = 0
        gap_bad_count = 0
        gap_severe_count = 0
        if ts is not None and len(ts) > 1:
            dt = np.diff(ts)
            max_dt = float(dt.max())
            min_dt = float(dt.min())
            mean_dt = float(dt.mean())
            p95_dt = float(np.percentile(dt, 95))
            p99_dt = float(np.percentile(dt, 99))
            gap_warn_count = int((dt > args.gap_warn_threshold).sum())
            gap_bad_count = int((dt > args.gap_bad_threshold).sum())
            gap_severe_count = int((dt > args.gap_severe_threshold).sum())
            for i, d in enumerate(dt):
                if d > args.gap_warn_threshold:
                    gap_rows.append({
                        "lerobot_episode_index": lr_idx,
                        "lerobot_episode_file": f"episode_{lr_idx:06d}.parquet",
                        "raw_episode_name": raw_episode_name,
                        "raw_episode_number": raw_episode_number if raw_episode_number is not None else "",
                        "intermediate_episode": ep.name,
                        "gap_after_frame": i,
                        "prev_timestamp": float(ts[i]),
                        "next_timestamp": float(ts[i + 1]),
                        "dt": float(d),
                        "approx_missing_15hz_frames": round(float(d) / (1.0 / args.fps) - 1),
                        "skipped_total_episode": skipped_total,
                        "skipped_cam_left_wrist": int(skipped.get("cam_left_wrist", 0)),
                        "skipped_cam_right_wrist": int(skipped.get("cam_right_wrist", 0)),
                        "skipped_state": int(skipped.get("state", 0)),
                        "skipped_action": int(skipped.get("action", 0)),
                        "source_episode_dir": source_episode_dir,
                        "source_bag_dir": source_bag_dir,
                    })

        if skipped_total > 0:
            warn.append("converter_skipped_frames")
        if gap_bad_count > 0:
            warn.append(f"cam_high_timestamp_gap_gt_{args.gap_bad_threshold:g}")
        elif gap_warn_count > 0:
            warn.append(f"cam_high_timestamp_gap_gt_{args.gap_warn_threshold:g}")

        missing_reason_detail = classify_missing_reason(skipped_total, raw_cam_high_count, state_len, gap_bad_count)
        status = make_status(hard_bad, warn)
        reasons = hard_bad + warn

        row = {
            "status": status,
            "reasons": "|".join(reasons),
            "missing_reason_detail": missing_reason_detail,
            "lerobot_episode_index": lr_idx,
            "lerobot_episode_file": f"episode_{lr_idx:06d}.parquet",
            "raw_episode_name": raw_episode_name,
            "raw_episode_number": raw_episode_number if raw_episode_number is not None else "",
            "raw_source_exists": raw_source_exists,
            "raw_bag_exists": raw_bag_exists,
            "intermediate_episode": ep.name,
            "manifest_source_episode": manifest_source_name,
            "raw_cam_high_count": raw_cam_high_count,
            "intermediate_frames": state_len,
            "lerobot_frames": lerobot_frames,
            "inferred_raw_to_inter_drop": inferred_raw_to_inter_drop,
            "skipped_total": skipped_total,
            "skipped_cam_left_wrist": int(skipped.get("cam_left_wrist", 0)),
            "skipped_cam_right_wrist": int(skipped.get("cam_right_wrist", 0)),
            "skipped_state": int(skipped.get("state", 0)),
            "skipped_action": int(skipped.get("action", 0)),
            "detailed_skip_available": detailed_skip_available,
            "detailed_topic_skip_total": detailed_topic_skip_total,
            **{
                f"skipped_topic_{key}": int(skipped_by_topic.get(key, 0))
                for key in DETAILED_SKIP_KEYS
            },
            "state_len": state_len,
            "action_len": action_len,
            "timestamps_len": ts_len,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "expected_dim": expected_dim if expected_dim is not None else "",
            "meta_state_shape": str(meta_state_shape),
            "meta_action_shape": str(meta_action_shape),
            "state_order_tail": compact_tail(state_order, 4),
            "action_order_tail": compact_tail(action_order, 4),
            "cam_high_images": image_counts["cam_high"],
            "cam_left_wrist_images": image_counts["cam_left_wrist"],
            "cam_right_wrist_images": image_counts["cam_right_wrist"],
            "max_state_err": max_state_err,
            "max_action_err": max_action_err,
            "lr_state_shape": lr_state_shape,
            "lr_action_shape": lr_action_shape,
            "intermediate_min_dt": min_dt,
            "intermediate_mean_dt": mean_dt,
            "intermediate_p95_dt": p95_dt,
            "intermediate_p99_dt": p99_dt,
            "intermediate_max_dt": max_dt,
            "gap_warn_count": gap_warn_count,
            "gap_bad_count": gap_bad_count,
            "gap_severe_count": gap_severe_count,
            "lerobot_timestamp_max_gap": lerobot_timestamp_max_gap,
            "lerobot_timestamp_min_gap": lerobot_timestamp_min_gap,
            "source_episode_dir": source_episode_dir,
            "source_bag_dir": source_bag_dir,
            "lerobot_parquet_path": str(pq) if pq else "",
        }
        rows.append(row)
        mapping_rows.append({
            "lerobot_episode_index": lr_idx,
            "lerobot_episode_file": f"episode_{lr_idx:06d}.parquet",
            "raw_episode_name": raw_episode_name,
            "raw_episode_number": raw_episode_number if raw_episode_number is not None else "",
            "intermediate_episode": ep.name,
            "source_episode_dir": source_episode_dir,
            "lerobot_parquet_path": str(pq) if pq else "",
            "status": status,
            "reasons": "|".join(reasons),
        })

    if not rows:
        raise RuntimeError(f"No intermediate episodes found under {inter_root}")

    summary_fields = list(rows[0].keys())
    write_csv(summary_csv, rows, summary_fields)
    bad_or_warn_rows = [r for r in rows if r["status"] != "OK"]
    write_csv(bad_csv, bad_or_warn_rows, summary_fields)
    write_csv(mapping_csv, mapping_rows, list(mapping_rows[0].keys()) if mapping_rows else [])
    gap_fields = list(gap_rows[0].keys()) if gap_rows else [
        "lerobot_episode_index", "lerobot_episode_file", "raw_episode_name", "raw_episode_number", "intermediate_episode",
        "gap_after_frame", "prev_timestamp", "next_timestamp", "dt", "approx_missing_15hz_frames",
        "skipped_total_episode", "skipped_cam_left_wrist", "skipped_cam_right_wrist", "skipped_state", "skipped_action",
        "source_episode_dir", "source_bag_dir",
    ]
    write_csv(gap_csv, gap_rows, gap_fields)

    skipped_topic_fields = [
        "lerobot_episode_index", "lerobot_episode_file", "raw_episode_name",
        "raw_episode_number", "intermediate_episode", "topic_key", "ros_topic",
        "skipped_reference_frames_attributed", "aggregate_skipped_total_episode",
        "detailed_topic_skip_total_episode", "source_episode_dir", "source_bag_dir",
    ]
    write_csv(skipped_topic_by_episode_csv, skipped_topic_rows, skipped_topic_fields)

    skipped_topic_summary_rows: list[dict[str, Any]] = []
    for topic_key in DETAILED_SKIP_KEYS:
        topic_rows = [r for r in skipped_topic_rows if r["topic_key"] == topic_key]
        if topic_rows:
            worst = max(topic_rows, key=lambda r: int(r["skipped_reference_frames_attributed"]))
            total = sum(int(r["skipped_reference_frames_attributed"]) for r in topic_rows)
            episodes_affected = len({int(r["lerobot_episode_index"]) for r in topic_rows})
            ros_topic = next((str(r["ros_topic"]) for r in topic_rows if r.get("ros_topic")), "")
            worst_episode = worst["raw_episode_name"]
            worst_count = int(worst["skipped_reference_frames_attributed"])
        else:
            total = 0
            episodes_affected = 0
            ros_topic = ""
            worst_episode = ""
            worst_count = 0
        skipped_topic_summary_rows.append({
            "topic_key": topic_key,
            "ros_topic": ros_topic,
            "skipped_reference_frames_attributed_total": total,
            "episodes_affected": episodes_affected,
            "worst_raw_episode": worst_episode,
            "worst_episode_skip_count": worst_count,
        })
    write_csv(
        skipped_topic_summary_csv,
        skipped_topic_summary_rows,
        list(skipped_topic_summary_rows[0].keys()) if skipped_topic_summary_rows else [],
    )

    raw_topic_timing_fields = [
        "lerobot_episode_index", "lerobot_episode_file", "raw_episode_name",
        "raw_episode_number", "intermediate_episode", "topic_key", "ros_topic",
        "timing_source", "message_count", "min_dt", "mean_dt", "p95_dt", "p99_dt", "max_dt",
        "nonpositive_dt_count", "max_dt_gt_warn", "max_dt_gt_bad",
        "max_dt_gt_severe", "source_episode_dir", "source_bag_dir",
    ]
    write_csv(raw_topic_timing_by_episode_csv, raw_topic_timing_rows, raw_topic_timing_fields)

    raw_topic_timing_summary_rows: list[dict[str, Any]] = []
    for topic_key in sorted({str(r["topic_key"]) for r in raw_topic_timing_rows}):
        topic_rows = [r for r in raw_topic_timing_rows if r["topic_key"] == topic_key]
        numeric_max = []
        numeric_p99 = []
        for r in topic_rows:
            try:
                numeric_max.append(float(r["max_dt"]))
            except (TypeError, ValueError):
                pass
            try:
                numeric_p99.append(float(r["p99_dt"]))
            except (TypeError, ValueError):
                pass
        worst = None
        if topic_rows:
            def _max_dt_value(row):
                try:
                    return float(row["max_dt"])
                except (TypeError, ValueError):
                    return float("-inf")
            worst = max(topic_rows, key=_max_dt_value)
        raw_topic_timing_summary_rows.append({
            "topic_key": topic_key,
            "ros_topic": next((str(r["ros_topic"]) for r in topic_rows if r.get("ros_topic")), ""),
            "timing_sources": "|".join(sorted({str(r.get("timing_source", "")) for r in topic_rows if r.get("timing_source")})),
            "episodes_reported": len(topic_rows),
            "messages_total": sum(int(r["message_count"]) for r in topic_rows),
            "episodes_max_dt_gt_warn": sum(bool(r["max_dt_gt_warn"]) for r in topic_rows),
            "episodes_max_dt_gt_bad": sum(bool(r["max_dt_gt_bad"]) for r in topic_rows),
            "episodes_max_dt_gt_severe": sum(bool(r["max_dt_gt_severe"]) for r in topic_rows),
            "largest_max_dt": max(numeric_max) if numeric_max else "",
            "largest_p99_dt": max(numeric_p99) if numeric_p99 else "",
            "worst_raw_episode": worst["raw_episode_name"] if worst else "",
        })
    write_csv(
        raw_topic_timing_summary_csv,
        raw_topic_timing_summary_rows,
        list(raw_topic_timing_summary_rows[0].keys()) if raw_topic_timing_summary_rows else [
            "topic_key", "ros_topic", "timing_sources", "episodes_reported", "messages_total",
            "episodes_max_dt_gt_warn", "episodes_max_dt_gt_bad",
            "episodes_max_dt_gt_severe", "largest_max_dt", "largest_p99_dt",
            "worst_raw_episode",
        ],
    )

    delete_candidates: list[str] = []
    seen = set()
    for r in rows:
        if should_delete_candidate(r, args.delete_candidate_policy):
            src = str(r.get("source_episode_dir", ""))
            if src and src not in seen:
                seen.add(src)
                delete_candidates.append(src)
    delete_candidates_txt.write_text("\n".join(delete_candidates) + ("\n" if delete_candidates else ""))

    stats = {
        "raw_root": str(raw_root),
        "intermediate_root": str(inter_root),
        "lerobot_root": str(lerobot_root),
        "out_dir": str(out_dir),
        "raw_episodes_currently_in_raw_root": len(raw_dirs),
        "intermediate_episodes": len(episode_dirs),
        "lerobot_parquet_files": len(parquet_files),
        "ok_episodes": sum(r["status"] == "OK" for r in rows),
        "warn_episodes": sum(r["status"] == "WARN" for r in rows),
        "bad_episodes": sum(r["status"] == "BAD" for r in rows),
        "converter_skipped_frames_total": sum(int(r["skipped_total"]) for r in rows),
        "episodes_with_converter_skipped": sum(int(r["skipped_total"]) > 0 for r in rows),
        "episodes_with_gap_bad": sum(int(r["gap_bad_count"]) > 0 for r in rows),
        "episodes_with_gap_severe": sum(int(r["gap_severe_count"]) > 0 for r in rows),
        "detailed_skip_metadata_episodes": sum(bool(r.get("detailed_skip_available")) for r in rows),
        "legacy_aggregate_only_episodes": sum(
            int(r.get("skipped_total", 0)) > 0 and not bool(r.get("detailed_skip_available"))
            for r in rows
        ),
        "skipped_by_topic_total": {
            r["topic_key"]: int(r["skipped_reference_frames_attributed_total"])
            for r in skipped_topic_summary_rows
        },
        "raw_topic_timing_metadata_episodes": len(
            {int(r["lerobot_episode_index"]) for r in raw_topic_timing_rows}
        ),
        "delete_candidate_policy": args.delete_candidate_policy,
        "delete_candidates": len(delete_candidates),
        "gap_warn_threshold": args.gap_warn_threshold,
        "gap_bad_threshold": args.gap_bad_threshold,
        "gap_severe_threshold": args.gap_severe_threshold,
        "expected_dim": expected_dim if expected_dim is not None else "",
    }
    write_json(summary_json, stats)

    print("=" * 100)
    print("Missing audit done")
    print("=" * 100)
    print("Raw root          :", raw_root)
    print("Intermediate root :", inter_root)
    print("LeRobot root      :", lerobot_root)
    print("Output dir        :", out_dir)
    print()
    for k in ["raw_episodes_currently_in_raw_root", "intermediate_episodes", "lerobot_parquet_files", "ok_episodes", "warn_episodes", "bad_episodes", "converter_skipped_frames_total", "episodes_with_converter_skipped", "episodes_with_gap_bad", "episodes_with_gap_severe", "detailed_skip_metadata_episodes", "legacy_aggregate_only_episodes", "raw_topic_timing_metadata_episodes", "delete_candidates"]:
        print(f"{k}: {stats[k]}")
    print("\nSaved:")
    print(summary_csv)
    print(bad_csv)
    print(gap_csv)
    print(skipped_topic_by_episode_csv)
    print(skipped_topic_summary_csv)
    print(raw_topic_timing_by_episode_csv)
    print(raw_topic_timing_summary_csv)
    print(mapping_csv)
    print(delete_candidates_txt)
    print(summary_json)

    if skipped_topic_summary_rows:
        print("\nDetailed skipped-reference-frame attribution by topic:")
        for r in sorted(
            skipped_topic_summary_rows,
            key=lambda x: int(x["skipped_reference_frames_attributed_total"]),
            reverse=True,
        ):
            total = int(r["skipped_reference_frames_attributed_total"])
            if total <= 0:
                continue
            print(
                f'  {r["topic_key"]}: total={total}, '
                f'episodes={r["episodes_affected"]}, '
                f'worst={r["worst_raw_episode"]}:{r["worst_episode_skip_count"]}'
            )

    if bad_or_warn_rows:
        print("\nTop BAD/WARN:")
        for r in bad_or_warn_rows[: args.top]:
            print(
                f'{r["status"]:4s} | {r["lerobot_episode_file"]} <- {r["raw_episode_name"]} | '
                f'reasons={r["reasons"]} | detail={r["missing_reason_detail"]} | '
                f'skipped={r["skipped_total"]} | max_dt={float(r["intermediate_max_dt"]):.6f}'
            )
    else:
        print("\n[OK] No missing, no skipped frames, no timestamp gaps above threshold.")


def main() -> None:
    args = parse_args()
    audit(args)


if __name__ == "__main__":
    main()
