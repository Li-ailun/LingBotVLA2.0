#!/usr/bin/env python3
"""
####################

把中间数据转换成lerobot：
（1）fps=15
cd /home/lixiang/PROJECT/LingBotVLA2.0

export HF_LEROBOT_HOME=/home/lixiang/PROJECT/LingBotVLA2.0/datasets/lerobot

.venv/bin/python /home/lixiang/PROJECT/LingBotVLA2.0/myscripts/16d/convert_tools/build_lerobot_from_r1pro_intermediate.py \
  --input /home/lixiang/PROJECT/LingBotVLA2.0/datasets/r1pro_intermediate_16d/pick_place \
  --repo-id lixiang/pick_place \
  --fps 15 \
  --overwrite

（2）fps=30
cd /home/lixiang/PROJECT/LingBotVLA2.0

export HF_LEROBOT_HOME=/home/lixiang/PROJECT/LingBotVLA2.0/datasets/lerobot

.venv/bin/python /home/lixiang/PROJECT/LingBotVLA2.0/myscripts/16d/convert_tools/build_lerobot_from_r1pro_intermediate.py \
  --input /home/lixiang/PROJECT/LingBotVLA2.0/datasets/r1pro_intermediate_16d/hang_screwdriver_on_tool_rack_30hz \
  --repo-id lixiang/hang_screwdriver_on_tool_rack_30hz \
  --fps 30 \
  --overwrite

############
  
####
检查从intermdediate转换来的lerobot是否帧数等完全一一转换
cd /home/lixiang/PROJECT/LingBotVLA2.0
export HF_LEROBOT_HOME=/home/lixiang/PROJECT/LingBotVLA2.0/datasets/lerobot

uv run python - <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd
import json

INTER_ROOT = Path("/home/lixiang/PROJECT/LingBotVLA2.0/datasets/r1pro_intermediate/hang_wire_stripper_on_wall_hook")
LEROBOT_ROOT = Path("/home/lixiang/PROJECT/LingBotVLA2.0/datasets/lerobot/lixiang/hang_wire_stripper_on_wall_hook")

episode_dirs = sorted(INTER_ROOT.glob("episode_*"))
parquet_files = sorted(LEROBOT_ROOT.glob("data/chunk-*/*.parquet"))

print("=" * 100)
print("Intermediate -> LeRobot copy check")
print("=" * 100)
print("intermediate episodes:", len(episode_dirs))
print("lerobot parquet files:", len(parquet_files))

if len(episode_dirs) != len(parquet_files):
    raise RuntimeError(f"episode count mismatch: intermediate={len(episode_dirs)} parquet={len(parquet_files)}")

def episode_index_from_parquet(p: Path):
    # episode_000123.parquet
    return int(p.stem.split("_")[1])

parquet_by_idx = {episode_index_from_parquet(p): p for p in parquet_files}

total_inter_frames = 0
total_lerobot_frames = 0
bad = []

global_max_state_err = 0.0
global_max_action_err = 0.0

for ep_idx, ep_dir in enumerate(episode_dirs):
    state = np.load(ep_dir / "state.npy").astype(np.float32)
    action = np.load(ep_dir / "action.npy").astype(np.float32)
    timestamps = np.load(ep_dir / "timestamps.npy")

    if ep_idx not in parquet_by_idx:
        bad.append((ep_idx, ep_dir.name, "missing parquet"))
        continue

    pq = parquet_by_idx[ep_idx]
    df = pd.read_parquet(pq)

    n_inter = len(state)
    n_lr = len(df)

    total_inter_frames += n_inter
    total_lerobot_frames += n_lr

    if len(action) != n_inter or len(timestamps) != n_inter:
        bad.append((ep_idx, ep_dir.name, f"intermediate length mismatch state={len(state)} action={len(action)} timestamps={len(timestamps)}"))
        continue

    if n_lr != n_inter:
        bad.append((ep_idx, ep_dir.name, f"frame count mismatch inter={n_inter} lerobot={n_lr}"))
        continue

    lr_state = np.stack(df["observation.state"].map(lambda x: np.asarray(x, dtype=np.float32)).to_numpy())
    lr_action = np.stack(df["action"].map(lambda x: np.asarray(x, dtype=np.float32)).to_numpy())

    if lr_state.shape != state.shape:
        bad.append((ep_idx, ep_dir.name, f"state shape mismatch inter={state.shape} lerobot={lr_state.shape}"))
        continue

    if lr_action.shape != action.shape:
        bad.append((ep_idx, ep_dir.name, f"action shape mismatch inter={action.shape} lerobot={lr_action.shape}"))
        continue

    state_err = float(np.max(np.abs(lr_state - state))) if state.size else 0.0
    action_err = float(np.max(np.abs(lr_action - action))) if action.size else 0.0

    global_max_state_err = max(global_max_state_err, state_err)
    global_max_action_err = max(global_max_action_err, action_err)

    if state_err > 1e-6:
        bad.append((ep_idx, ep_dir.name, f"state value mismatch max_err={state_err}"))

    if action_err > 1e-6:
        bad.append((ep_idx, ep_dir.name, f"action value mismatch max_err={action_err}"))

print("total intermediate frames:", total_inter_frames)
print("total lerobot frames:", total_lerobot_frames)
print("max state abs err:", global_max_state_err)
print("max action abs err:", global_max_action_err)

print("\nBAD count:", len(bad))
for x in bad[:50]:
    print(x)

if bad:
    raise SystemExit("FAILED: intermediate -> LeRobot mismatch found")
else:
    print("\n[OK] LeRobot state/action/frame counts match intermediate exactly.")
PY
  
######################


#########################
检查 intermediate 的真实头部时间戳

cd /home/lixiang/PROJECT/LingBotVLA2.0

python - <<'PY'
from pathlib import Path
import numpy as np
import json
import pandas as pd

root = Path("/home/lixiang/PROJECT/LingBotVLA2.0/datasets/r1pro_intermediate/hang_wire_stripper_on_wall_hook")
out_dir = Path("/home/lixiang/PROJECT/LingBotVLA2.0/datasets/r1pro_alignment_reports/hang_wire_stripper_on_wall_hook_intermediate_timestamp_check")
out_dir.mkdir(parents=True, exist_ok=True)

rows = []
gap_rows = []

for ep_idx, ep in enumerate(sorted(root.glob("episode_*"))):
    t = np.load(ep / "timestamps.npy").astype(float)
    dt = np.diff(t)

    meta_path = ep / "episode_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    skipped = meta.get("skipped", {})
    skipped_total = sum(int(v) for v in skipped.values()) if skipped else 0

    if len(dt) > 0:
        max_dt = float(dt.max())
        min_dt = float(dt.min())
        mean_dt = float(dt.mean())
        p95_dt = float(np.percentile(dt, 95))
        p99_dt = float(np.percentile(dt, 99))
        gap_gt_008 = int((dt > 0.08).sum())
        gap_gt_010 = int((dt > 0.10).sum())
        gap_gt_012 = int((dt > 0.12).sum())
        gap_gt_015 = int((dt > 0.15).sum())
    else:
        max_dt = min_dt = mean_dt = p95_dt = p99_dt = 0.0
        gap_gt_008 = gap_gt_010 = gap_gt_012 = gap_gt_015 = 0

    rows.append({
        "episode_index": ep_idx,
        "episode_name": ep.name,
        "frames": len(t),
        "duration_sec": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
        "min_dt": min_dt,
        "mean_dt": mean_dt,
        "p95_dt": p95_dt,
        "p99_dt": p99_dt,
        "max_dt": max_dt,
        "gap_gt_0p08_count": gap_gt_008,
        "gap_gt_0p10_count": gap_gt_010,
        "gap_gt_0p12_count": gap_gt_012,
        "gap_gt_0p15_count": gap_gt_015,
        "skipped_total": skipped_total,
        "skipped_cam_left_wrist": int(skipped.get("cam_left_wrist", 0)),
        "skipped_cam_right_wrist": int(skipped.get("cam_right_wrist", 0)),
        "skipped_state": int(skipped.get("state", 0)),
        "skipped_action": int(skipped.get("action", 0)),
    })

    for i, d in enumerate(dt):
        if d > 0.08:
            gap_rows.append({
                "episode_index": ep_idx,
                "episode_name": ep.name,
                "gap_after_frame": i,
                "prev_timestamp": float(t[i]),
                "next_timestamp": float(t[i + 1]),
                "dt": float(d),
                "approx_missing_15hz_frames": round(float(d) / (1.0 / 15.0) - 1),
                "skipped_total_episode": skipped_total,
                "skipped_cam_left_wrist": int(skipped.get("cam_left_wrist", 0)),
                "skipped_cam_right_wrist": int(skipped.get("cam_right_wrist", 0)),
                "skipped_state": int(skipped.get("state", 0)),
                "skipped_action": int(skipped.get("action", 0)),
            })

summary = pd.DataFrame(rows)
gaps = pd.DataFrame(gap_rows)

summary_csv = out_dir / "intermediate_timestamp_summary.csv"
gaps_csv = out_dir / "intermediate_timestamp_gaps.csv"

summary.to_csv(summary_csv, index=False)
gaps.to_csv(gaps_csv, index=False)

print("=" * 100)
print("Intermediate cam_high timestamp check")
print("=" * 100)
print("episodes:", len(summary))
print("total frames:", int(summary["frames"].sum()))
print()
print("dt describe:")
print(summary[["min_dt", "mean_dt", "p95_dt", "p99_dt", "max_dt"]].describe(percentiles=[0.5, 0.9, 0.95, 0.99]))

print()
print("episodes with gap > 0.08:", int((summary["gap_gt_0p08_count"] > 0).sum()))
print("total gaps > 0.08:", int(summary["gap_gt_0p08_count"].sum()))
print("episodes with gap > 0.12:", int((summary["gap_gt_0p12_count"] > 0).sum()))
print("total gaps > 0.12:", int(summary["gap_gt_0p12_count"].sum()))
print("episodes with gap > 0.15:", int((summary["gap_gt_0p15_count"] > 0).sum()))
print("total gaps > 0.15:", int(summary["gap_gt_0p15_count"].sum()))

print()
print("Top gap episodes:")
print(summary.sort_values("max_dt", ascending=False)[[
    "episode_index",
    "episode_name",
    "frames",
    "max_dt",
    "p95_dt",
    "gap_gt_0p08_count",
    "gap_gt_0p12_count",
    "gap_gt_0p15_count",
    "skipped_total",
    "skipped_cam_left_wrist",
    "skipped_cam_right_wrist",
    "skipped_state",
    "skipped_action",
]].head(50).to_string(index=False))

print()
print("Saved:")
print(summary_csv)
print(gaps_csv)
PY


###########################

###########################
  
验证转换的lerobot：

find /home/lixiang/PROJECT/LingBotVLA2.0/datasets/lerobot/lixiang/hang_wire_stripper_on_wall_hook \
  -maxdepth 4 -type f | sort | head -100

+

cd /home/lixiang/PROJECT/LingBotVLA2.0

export HF_LEROBOT_HOME=/home/lixiang/PROJECT/LingBotVLA2.0/datasets/lerobot/lixiang/hang_wire_stripper_on_wall_hook

uv run python - <<'PY'
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME

print("HF_LEROBOT_HOME:", HF_LEROBOT_HOME)

ds = LeRobotDataset("lixiang/r1pro_pi05")
print(ds)
print("num_frames:", len(ds))
print("num_episodes:", ds.num_episodes)

sample = ds[0]
print("state shape:", sample["observation.state"].shape)
print("action shape:", sample["action"].shape)
print("task:", sample["task"])
PY
########################


#########################
总结/home/lixiang/PROJECT/LingBotVLA2.0/datasets/r1pro_intermediate/hang_wire_stripper_on_wall_hook/episode_*/episode_meta.json离得跳帧事件：


python - <<'PY'
from pathlib import Path
import json

root = Path("/home/lixiang/PROJECT/LingBotVLA2.0/datasets/r1pro_intermediate/hang_wire_stripper_on_wall_hook")

bad = []
total_skipped_all = 0

for ep in sorted(root.glob("episode_*")):
    meta_path = ep / "episode_meta.json"
    if not meta_path.exists():
        continue

    meta = json.loads(meta_path.read_text())
    skipped = meta.get("skipped", {})

    total = sum(int(v) for v in skipped.values())
    total_skipped_all += total

    if total > 0:
        bad.append({
            "episode": ep.name,
            "frame_count": meta.get("frame_count"),
            "skipped_total": total,
            "cam_left_wrist": int(skipped.get("cam_left_wrist", 0)),
            "cam_right_wrist": int(skipped.get("cam_right_wrist", 0)),
            "state": int(skipped.get("state", 0)),
            "action": int(skipped.get("action", 0)),
        })

print("=" * 100)
print("Episodes with skipped frames")
print("=" * 100)
print("total episodes scanned:", len(list(root.glob("episode_*"))))
print("episodes with skipped frames:", len(bad))
print("total skipped frames:", total_skipped_all)
print()

for r in bad:
    print(
        f'{r["episode"]}: '
        f'skipped_total={r["skipped_total"]}, '
        f'frames={r["frame_count"]}, '
        f'left_wrist={r["cam_left_wrist"]}, '
        f'right_wrist={r["cam_right_wrist"]}, '
        f'state={r["state"]}, '
        f'action={r["action"]}'
    )
PY



##############

##########
mcap到lerobot的序号对应：
cd /home/lixiang/PROJECT/LingBotVLA2.0

python - <<'PY'
from pathlib import Path
import json
import csv

inter_root = Path("/home/lixiang/PROJECT/LingBotVLA2.0/datasets/r1pro_intermediate/hang_wire_stripper_on_wall_hook")
lerobot_root = Path("/home/lixiang/PROJECT/LingBotVLA2.0/datasets/lerobot/lixiang/hang_wire_stripper_on_wall_hook")
out_csv = Path("/home/lixiang/PROJECT/LingBotVLA2.0/reports/lerobot_quality_reports/hang_wire_stripper_on_wall_hook_raw_to_lerobot_mapping.csv")
out_csv.parent.mkdir(parents=True, exist_ok=True)

manifest_path = inter_root / "manifest.json"
manifest = json.loads(manifest_path.read_text())

# raw -> intermediate
raw_by_inter_name = {}
for item in manifest["episodes"]:
    source = Path(item["source"])
    inter = Path(item["intermediate"])
    raw_by_inter_name[inter.name] = source.name

# intermediate -> lerobot
episode_dirs = sorted([p for p in inter_root.glob("episode_*") if p.is_dir()])

rows = []
for lr_idx, ep in enumerate(episode_dirs):
    meta_path = ep / "episode_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    skipped = meta.get("skipped", {})
    skipped_total = sum(int(v) for v in skipped.values()) if skipped else 0

    rows.append({
        "lerobot_episode_index": lr_idx,
        "lerobot_episode_file": f"episode_{lr_idx:06d}.parquet",
        "raw_mcap_episode": raw_by_inter_name.get(ep.name, ""),
        "intermediate_episode": ep.name,
        "frame_count": meta.get("frame_count", ""),
        "skipped_total": skipped_total,
        "skipped_cam_left_wrist": int(skipped.get("cam_left_wrist", 0)),
        "skipped_cam_right_wrist": int(skipped.get("cam_right_wrist", 0)),
        "skipped_state": int(skipped.get("state", 0)),
        "skipped_action": int(skipped.get("action", 0)),
        "source_episode_dir": meta.get("source_episode_dir", ""),
        "source_bag_dir": meta.get("source_bag_dir", ""),
    })

with out_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print("saved:", out_csv)
print("rows:", len(rows))
print()
for r in rows[:10]:
    print(r["lerobot_episode_file"], "<-", r["raw_mcap_episode"])
PY

#############################

Build OpenPI-compatible LeRobot dataset from R1Pro intermediate data.

This script MUST run in OpenPI uv environment:

    cd /home/lixiang/PROJECT/LingBotVLA2.0
    uv run python build_lerobot_from_r1pro_intermediate.py ...

Input:
    r1pro_intermediate/<task_dir>/episode_xxx/
        task.txt
        episode_meta.json
        timestamps.npy
        state.npy
        action.npy
        images/
            cam_high/
            cam_left_wrist/
            cam_right_wrist/

Output:
    ~/.cache/huggingface/lerobot/<repo_id>/
        data/
        videos/
        meta/
"""

import argparse
import csv
import gc
import importlib
import inspect
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
_schema = importlib.import_module("myscripts.16d.common.schema")
SCHEMA_NAME = _schema.SCHEMA_NAME
STATE_NAMES = list(_schema.STATE_NAMES)
ACTION_NAMES = list(_schema.ACTION_NAMES)
VECTOR_DIM = _schema.VECTOR_DIM
validate_vector_array = _schema.validate_vector_array


IMAGE_SHAPES = {
    "cam_high": (360, 640, 3),
    "cam_left_wrist": (480, 640, 3),
    "cam_right_wrist": (480, 640, 3),
}

INSTRUCTION_ORDER = {
    "stack_plates_cups_01": 0,
    "stack_plates_cups_02": 1,
    "stack_plates_cups_03": 2,
}
OPERATOR_ORDER = {"li": 0, "liu": 1, "yu": 2, "xing": 3}


def plan_worker_ranges(total_episodes: int, start_index: int, episodes_per_worker: int) -> list[tuple[int, int]]:
    """Split a conversion into bounded child-process ranges."""
    if episodes_per_worker <= 0:
        raise ValueError("episodes_per_worker must be positive")
    if start_index < 0 or start_index > total_episodes:
        raise ValueError("start_index must be between zero and total_episodes")
    return [
        (start, min(start + episodes_per_worker, total_episodes))
        for start in range(start_index, total_episodes, episodes_per_worker)
    ]

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


def read_rgb_image(path: Path, target_shape: tuple[int, int, int]) -> np.ndarray:
    target_height, target_width, target_channels = target_shape
    if target_channels != 3:
        raise ValueError(f"RGB image target must have 3 channels, got {target_shape}")

    with Image.open(path) as source:
        img = source.convert("RGB")
        if img.size != (target_width, target_height):
            img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        return np.asarray(img, dtype=np.uint8)


def find_image_path(cam_dir: Path, frame_idx: int) -> Path:
    stem = f"{frame_idx:06d}"
    candidates = sorted(cam_dir.glob(stem + ".*"))
    if not candidates:
        raise FileNotFoundError(f"No image found for frame {stem} in {cam_dir}")
    return candidates[0]


def find_episode_dirs(input_dir: Path):
    input_dir = input_dir.expanduser().resolve()
    eps = sorted([
        p for p in input_dir.rglob("episode_*")
        if p.is_dir()
        if (p / "state.npy").exists()
        and (p / "action.npy").exists()
        and (p / "images").exists()
    ])
    if not eps:
        raise RuntimeError(f"No intermediate episode directories found under: {input_dir}")
    return eps


def read_episode_meta(ep_dir: Path) -> dict:
    meta_file = ep_dir / "episode_meta.json"
    if not meta_file.exists():
        return {}
    try:
        return json.loads(meta_file.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _episode_index_from_name(name: str) -> int:
    prefix = "episode_"
    if not name.startswith(prefix):
        return 10**12
    digits = name[len(prefix):].split("_", 1)[0]
    return int(digits) if digits.isdigit() else 10**12


def episode_sort_key(ep_dir: Path):
    meta = read_episode_meta(ep_dir)
    task_id = str(meta.get("task_id") or "unknown_task")
    instruction_id = str(meta.get("instruction_id") or "")
    operator_id = str(meta.get("operator_id") or "unknown").lower()
    source_index = meta.get("source_mcap_episode_index")
    if source_index is None:
        source_index = _episode_index_from_name(
            str(meta.get("source_mcap_episode_name") or ep_dir.name)
        )
    return (
        task_id,
        INSTRUCTION_ORDER.get(instruction_id, 999),
        instruction_id,
        OPERATOR_ORDER.get(operator_id, 999),
        operator_id,
        int(source_index),
        str(meta.get("source_mcap_episode_name") or ep_dir.name),
        str(ep_dir),
    )


def order_episode_dirs(episode_dirs):
    return sorted(episode_dirs, key=episode_sort_key)


def build_episode_source_record(ep_dir: Path, lerobot_index: int, input_dir: Path) -> dict:
    meta = read_episode_meta(ep_dir)
    try:
        intermediate_relative = str(ep_dir.relative_to(input_dir))
    except ValueError:
        intermediate_relative = ep_dir.name
    source_relative = str(
        meta.get("source_episode_relative_path")
        or meta.get("source_mcap_relative_path")
        or meta.get("source_mcap_episode_name")
        or ep_dir.name
    )
    source_index = meta.get("source_mcap_episode_index")
    if source_index is None:
        source_index = _episode_index_from_name(
            str(meta.get("source_mcap_episode_name") or ep_dir.name)
        )
        if source_index == 10**12:
            source_index = None
    return {
        "lerobot_episode_index": int(lerobot_index),
        "lerobot_sequence_number": int(lerobot_index) + 1,
        "lerobot_episode_file": f"episode_{lerobot_index:06d}.parquet",
        "operator_id": str(meta.get("operator_id") or "unknown"),
        "task_id": str(meta.get("task_id") or "unknown_task"),
        "instruction_id": meta.get("instruction_id"),
        "instruction_text": str(meta.get("instruction_text") or meta.get("task") or ""),
        "source_mcap_episode_index": source_index,
        "source_mcap_episode_name": str(
            meta.get("source_mcap_episode_name") or ep_dir.name
        ),
        "source_mcap_relative_path": source_relative,
        "source_mcap_absolute_path": str(meta.get("source_episode_dir") or ""),
        "intermediate_relative_path": intermediate_relative,
        "intermediate_absolute_path": str(ep_dir),
        "frame_count": int(meta.get("frame_count") or 0),
    }


def write_episode_source_maps(dataset_root: Path, records: list[dict]) -> None:
    meta_dir = dataset_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    resolved_records = []
    for record in records:
        resolved = dict(record)
        episode_index = int(resolved["lerobot_episode_index"])
        matches = sorted(
            dataset_root.glob(f"data/chunk-*/episode_{episode_index:06d}.parquet")
        )
        if len(matches) == 1:
            resolved["lerobot_episode_file"] = str(matches[0].relative_to(dataset_root))
        resolved_records.append(resolved)
    jsonl_path = meta_dir / "episode_source_map.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in resolved_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    csv_path = meta_dir / "episode_source_map.csv"
    fieldnames = list(resolved_records[0]) if resolved_records else ["lerobot_episode_index"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(resolved_records)


def make_features():
    return {
        "observation.images.cam_high": {
            "dtype": "image",
            "shape": IMAGE_SHAPES["cam_high"],
            "names": ["height", "width", "channel"],
        },
        "observation.images.cam_left_wrist": {
            "dtype": "image",
            "shape": IMAGE_SHAPES["cam_left_wrist"],
            "names": ["height", "width", "channel"],
        },
        "observation.images.cam_right_wrist": {
            "dtype": "image",
            "shape": IMAGE_SHAPES["cam_right_wrist"],
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (VECTOR_DIM,),
            "names": STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (VECTOR_DIM,),
            "names": ACTION_NAMES,
        },
    }


def create_dataset(repo_id: str, fps: int, overwrite: bool):
    output_path = HF_LEROBOT_HOME / repo_id

    if output_path.exists():
        if not overwrite:
            raise RuntimeError(f"Output dataset already exists: {output_path}. Use --overwrite to rebuild.")
        print(f"[INFO] Removing existing dataset: {output_path}")
        shutil.rmtree(output_path)

    features = make_features()

    create_sig = inspect.signature(LeRobotDataset.create)
    kwargs = {
        "repo_id": repo_id,
        "robot_type": "r1pro",
        "fps": fps,
        "features": features,
    }

    # Make this script compatible with slightly different LeRobot versions.
    if "image_writer_threads" in create_sig.parameters:
        kwargs["image_writer_threads"] = 2
    if "image_writer_processes" in create_sig.parameters:
        kwargs["image_writer_processes"] = 1
    if "use_videos" in create_sig.parameters:
        kwargs["use_videos"] = True

    print("[INFO] Creating LeRobotDataset")
    print(f"  repo_id: {repo_id}")
    print(f"  path   : {output_path}")
    print(f"  fps    : {fps}")

    return LeRobotDataset.create(**kwargs)


def add_frame_compatible(dataset, frame, task: str):
    sig = inspect.signature(dataset.add_frame)
    if "task" in sig.parameters:
        dataset.add_frame(frame, task=task)
    else:
        dataset.add_frame(frame)


def save_episode_compatible(dataset, task: str):
    sig = inspect.signature(dataset.save_episode)
    if "task" in sig.parameters:
        dataset.save_episode(task=task)
    else:
        dataset.save_episode()


def read_task(ep_dir: Path):
    task_file = ep_dir / "task.txt"
    if task_file.exists():
        task = task_file.read_text().strip()
        if task:
            return task

    meta_file = ep_dir / "episode_meta.json"
    if meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text())
            task = str(meta.get("task", "")).strip()
            if task:
                return task
        except Exception:
            pass

    return "unknown_task"


def convert_episode(dataset, ep_dir: Path):
    task = read_task(ep_dir)
    state = np.load(ep_dir / "state.npy").astype(np.float32)
    action = np.load(ep_dir / "action.npy").astype(np.float32)
    timestamps = np.load(ep_dir / "timestamps.npy").astype(np.float64)

    validate_vector_array(state, name="state", episode=ep_dir.name)
    validate_vector_array(action, name="action", episode=ep_dir.name)
    if len(state) != len(action):
        raise RuntimeError(f"state/action length mismatch in {ep_dir}: {len(state)} vs {len(action)}")
    if len(timestamps) != len(state):
        raise RuntimeError(f"timestamps/state length mismatch in {ep_dir}: {len(timestamps)} vs {len(state)}")

    n = len(state)

    print("=" * 100)
    print(f"[EPISODE] {ep_dir.name}")
    print(f"  task  : {task}")
    print(f"  frames: {n}")

    img_dirs = {
        "cam_high": ep_dir / "images" / "cam_high",
        "cam_left_wrist": ep_dir / "images" / "cam_left_wrist",
        "cam_right_wrist": ep_dir / "images" / "cam_right_wrist",
    }

    for i in range(n):
        cam_high = read_rgb_image(
            find_image_path(img_dirs["cam_high"], i), IMAGE_SHAPES["cam_high"]
        )
        cam_left = read_rgb_image(
            find_image_path(img_dirs["cam_left_wrist"], i), IMAGE_SHAPES["cam_left_wrist"]
        )
        cam_right = read_rgb_image(
            find_image_path(img_dirs["cam_right_wrist"], i), IMAGE_SHAPES["cam_right_wrist"]
        )

        frame = {
            "observation.images.cam_high": cam_high,
            "observation.images.cam_left_wrist": cam_left,
            "observation.images.cam_right_wrist": cam_right,
            "observation.state": state[i],
            "action": action[i],
            "task": task,
        }

        add_frame_compatible(dataset, frame, task)

    save_episode_compatible(dataset, task)
    del state, action, timestamps
    gc.collect()
    print("[OK] saved LeRobot episode")


def existing_episode_count(dataset_root: Path) -> int:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        return 0
    info = json.loads(info_path.read_text(encoding="utf-8"))
    return int(info.get("total_episodes", 0))


def write_schema_and_source_maps(dataset_root: Path, episode_dirs: list[Path], input_dir: Path) -> None:
    schema_meta = dataset_root / "meta" / "r1pro_schema.json"
    schema_meta.parent.mkdir(parents=True, exist_ok=True)
    schema_meta.write_text(json.dumps({
        "schema": SCHEMA_NAME,
        "state_names": STATE_NAMES,
        "action_names": ACTION_NAMES,
    }, indent=2, ensure_ascii=False) + "\n")
    source_records = [
        build_episode_source_record(ep_dir, lerobot_index, input_dir)
        for lerobot_index, ep_dir in enumerate(episode_dirs)
    ]
    write_episode_source_maps(dataset_root, source_records)


def run_worker(args, input_dir: Path, episode_dirs: list[Path]) -> None:
    dataset_root = HF_LEROBOT_HOME / args.repo_id
    if args.worker_start == 0:
        dataset = create_dataset(args.repo_id, args.fps, args.overwrite)
    else:
        dataset = LeRobotDataset(args.repo_id, root=dataset_root)
        if dataset.num_episodes != args.worker_start:
            raise RuntimeError(
                f"append position mismatch: dataset has {dataset.num_episodes} episodes, "
                f"worker starts at {args.worker_start}"
            )
    for lerobot_index in range(args.worker_start, args.worker_end):
        ep_dir = episode_dirs[lerobot_index]
        convert_episode(dataset, ep_dir)
        source_record = build_episode_source_record(ep_dir, lerobot_index, input_dir)
        print(
            f"[MAP] episode_{lerobot_index:06d} <- "
            f"operator={source_record['operator_id']} "
            f"instruction={source_record['instruction_id']} "
            f"mcap={source_record['source_mcap_episode_name']}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Intermediate task directory.")
    parser.add_argument("--repo-id", default="lixiang/r1pro_16d_left_first")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from the completed episode count")
    parser.add_argument("--episodes-per-worker", type=int, default=10,
                        help="Restart the converter after this many episodes to bound memory")
    parser.add_argument("--worker-start", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-end", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    episode_dirs = order_episode_dirs(find_episode_dirs(input_dir))

    print(f"[INFO] Found {len(episode_dirs)} intermediate episodes")

    if (args.worker_start is None) != (args.worker_end is None):
        parser.error("--worker-start and --worker-end must be used together")
    if args.worker_start is not None:
        run_worker(args, input_dir, episode_dirs)
        return

    dataset_root = HF_LEROBOT_HOME / args.repo_id
    if args.overwrite and args.resume:
        parser.error("--overwrite and --resume are mutually exclusive")
    if dataset_root.exists() and not (args.overwrite or args.resume):
        raise RuntimeError(f"Output dataset already exists: {dataset_root}. Use --overwrite or --resume.")
    start_index = existing_episode_count(dataset_root) if args.resume else 0
    if start_index > len(episode_dirs):
        raise RuntimeError(
            f"Existing dataset has {start_index} episodes but only {len(episode_dirs)} sources remain"
        )
    ranges = plan_worker_ranges(len(episode_dirs), start_index, args.episodes_per_worker)
    print(f"[INFO] Starting at episode {start_index}; worker batches: {ranges}")
    for batch_number, (worker_start, worker_end) in enumerate(ranges, start=1):
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--input", str(input_dir), "--repo-id", args.repo_id,
            "--fps", str(args.fps),
            "--worker-start", str(worker_start), "--worker-end", str(worker_end),
        ]
        if worker_start == 0 and args.overwrite:
            command.append("--overwrite")
        print(f"[WORKER {batch_number}/{len(ranges)}] episodes {worker_start}..{worker_end - 1}")
        subprocess.run(command, check=True)
        write_schema_and_source_maps(dataset_root, episode_dirs[:worker_end], input_dir)
        print(f"[CHECKPOINT] completed {worker_end}/{len(episode_dirs)} episodes")

    if not ranges:
        write_schema_and_source_maps(dataset_root, episode_dirs[:start_index], input_dir)

    print("=" * 100)
    print("[DONE] intermediate -> LeRobot finished")
    print(f"repo_id: {args.repo_id}")
    print(f"path   : {HF_LEROBOT_HOME / args.repo_id}")


if __name__ == "__main__":
    main()
