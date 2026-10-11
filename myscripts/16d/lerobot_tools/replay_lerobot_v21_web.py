#!/usr/bin/env python3
"""
LeRobot v2.1 web replay with synchronized RGB views and 16D state/action curves.

This script is based on:
  replay_lerobot_v21_parquet_images_keyboard.py

Features:
  - Select, previous, and next episode
  - Play/pause, restart, stop, previous frame, next frame
  - 0.25x / 0.5x / 1x / 1.5x / 2x playback
  - Three synchronized camera views
  - Progressive state/action curves synchronized to the displayed frame
  - Left arm 7D, left gripper, right arm 7D, and right gripper
  - State-only, action-only, or state+action overlay
  - Current-frame 16D numeric table
  - At the end of an episode, the complete curves remain visible

Example:
cd /home/lixiang/PROJECT/LingBotVLA2.0

.venv/bin/python \
  /home/lixiang/PROJECT/LingBotVLA2.0/myscripts/16d/lerobot_tools/replay_lerobot_v21_web.py \
  --root /home/lixiang/PROJECT/LingBotVLA2.0/datasets/lerobot/lixiang/hang_screwdriver_on_tool_rack \
  --episode-index 0 \
  --rate 1 \
  --ui-fps 15 \
  --host 127.0.0.1 \
  --port 8050 \
  --open-browser

Open:
  http://127.0.0.1:8050

For LAN access:
  --host 0.0.0.0
Then open:
  http://<host-ip>:8050
"""

from __future__ import annotations

import argparse
import base64
import importlib
import json
import math
import threading
import time
import webbrowser
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
import sys

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
_schema = importlib.import_module("myscripts.16d.common.schema")

try:
    from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html
except ImportError as exc:
    raise SystemExit(
        "Missing Dash. Install dependencies with:\n"
        "  cd /home/lixiang/PROJECT/LingBotVLA2.0\n"
        "  .venv/bin/python -m pip install dash plotly pandas pyarrow opencv-python\n"
    ) from exc


# ======================================================================================
# R1 Pro left-first 16D order
# ======================================================================================

DEFAULT_STATE_NAMES = list(_schema.STATE_NAMES)
DEFAULT_ACTION_NAMES = list(_schema.ACTION_NAMES)

GROUPS = {
    "left_arm": {
        "title": "Left arm / 左臂 7关节",
        "indices": list(range(0, 7)),
        "display_names": [f"L-J{i}" for i in range(7)],
    },
    "left_gripper": {
        "title": "Left gripper / 左夹爪",
        "indices": [7],
        "display_names": ["left_gripper"],
    },
    "right_arm": {
        "title": "Right arm / 右臂 7关节",
        "indices": list(range(8, 15)),
        "display_names": [f"R-J{i}" for i in range(7)],
    },
    "right_gripper": {
        "title": "Right gripper / 右夹爪",
        "indices": [15],
        "display_names": ["right_gripper"],
    },
}

PLOT_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


# ======================================================================================
# Data loading
# ======================================================================================

def log(message: str) -> None:
    print(message, flush=True)


def load_info(root: Path) -> dict:
    path = root / "meta" / "info.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset metadata: {path}")
    return json.loads(path.read_text())


def infer_camera_keys(info: dict) -> list[str]:
    keys = [
        key
        for key, feature in info.get("features", {}).items()
        if key.startswith("observation.images.")
    ]

    def priority(key: str) -> tuple[int, str]:
        lower = key.lower()
        if "cam_high" in lower or "head" in lower or "high" in lower:
            return 0, key
        if "left" in lower and "wrist" in lower:
            return 1, key
        if "right" in lower and "wrist" in lower:
            return 2, key
        return 10, key

    return sorted(keys, key=priority)


def short_camera_name(camera_key: str | None) -> str:
    if not camera_key:
        return "missing_camera"
    return camera_key.split(".")[-1]


def pick_camera(
    camera_keys: list[str],
    required_tokens: list[str],
    fallback_index: int | None = None,
) -> str | None:
    for camera_key in camera_keys:
        lower = camera_key.lower()
        if all(token in lower for token in required_tokens):
            return camera_key
    if fallback_index is not None and 0 <= fallback_index < len(camera_keys):
        return camera_keys[fallback_index]
    return None


def find_parquets(root: Path) -> list[Path]:
    paths = sorted((root / "data").glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found under: {root / 'data'}")
    return paths


def parse_episode_index_from_name(path: Path) -> int | None:
    for token in path.stem.replace("-", "_").split("_"):
        if token.isdigit():
            return int(token)
    return None


def build_episode_list(root: Path) -> list[dict]:
    by_episode: dict[int, dict] = {}

    for parquet in find_parquets(root):
        try:
            df_index = pd.read_parquet(parquet, columns=["episode_index"])
        except Exception:
            df_index = pd.read_parquet(parquet)

        if "episode_index" in df_index.columns:
            episode_indices = sorted(
                set(int(value) for value in df_index["episode_index"].dropna().unique())
            )
        else:
            inferred = parse_episode_index_from_name(parquet)
            episode_indices = [inferred] if inferred is not None else []

        if not episode_indices:
            episode_indices = [len(by_episode)]

        for episode_index in episode_indices:
            if episode_index in by_episode:
                raise RuntimeError(
                    "The current web replay expects one parquet source per episode, "
                    f"but episode {episode_index} appeared more than once:\n"
                    f"  previous: {by_episode[episode_index]['parquet']}\n"
                    f"  current : {parquet}"
                )
            by_episode[episode_index] = {
                "episode_index": int(episode_index),
                "parquet": parquet,
            }

    episodes = [by_episode[index] for index in sorted(by_episode)]
    for list_index, episode in enumerate(episodes):
        episode["list_index"] = list_index
    return episodes


def infer_feature_names(
    info: dict,
    feature_key: str,
    fallback: list[str],
) -> list[str]:
    feature = info.get("features", {}).get(feature_key, {})
    names = feature.get("names")
    if isinstance(names, list) and len(names) == 16:
        return [str(name) for name in names]
    return list(fallback)


def validate_16d_info(info: dict) -> None:
    for key, expected_names in (
        ("observation.state", DEFAULT_STATE_NAMES),
        ("action", DEFAULT_ACTION_NAMES),
    ):
        feature = info.get("features", {}).get(key, {})
        shape = feature.get("shape")
        if not isinstance(shape, (list, tuple)) or list(shape) != [16]:
            raise ValueError(f"{key}: expected metadata shape [16], got {shape}")
        names = feature.get("names")
        if names is not None and list(names) != list(expected_names):
            raise ValueError(f"{key}: feature names do not match {_schema.SCHEMA_NAME}")


def load_episode_dataframe(episode: dict, camera_keys: list[str]) -> pd.DataFrame:
    parquet = episode["parquet"]

    requested_columns = [
        "episode_index",
        "frame_index",
        "timestamp",
        "task",
        "observation.state",
        "action",
        *camera_keys,
    ]

    try:
        dataframe = pd.read_parquet(parquet, columns=requested_columns)
    except Exception:
        # Some parquet files do not expose every optional column.
        dataframe = pd.read_parquet(parquet)

    if "episode_index" in dataframe.columns:
        dataframe = dataframe[
            dataframe["episode_index"] == episode["episode_index"]
        ].copy()

    if "frame_index" in dataframe.columns:
        dataframe = dataframe.sort_values("frame_index").reset_index(drop=True)
    else:
        dataframe = dataframe.reset_index(drop=True)

    required = ["observation.state", "action"]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise RuntimeError(
            f"Missing required parquet columns in {parquet}: {missing}"
        )

    return dataframe


def value_to_vector(value: Any, expected_dim: int, column_name: str) -> np.ndarray:
    if hasattr(value, "as_py"):
        value = value.as_py()

    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != expected_dim:
        raise RuntimeError(
            f"Bad {column_name} vector size: expected {expected_dim}, got {array.size}"
        )
    return array


def stack_vector_column(
    dataframe: pd.DataFrame,
    column_name: str,
    expected_dim: int = 20,
) -> np.ndarray:
    vectors = [
        value_to_vector(value, expected_dim, column_name)
        for value in dataframe[column_name].tolist()
    ]
    if not vectors:
        return np.empty((0, expected_dim), dtype=np.float32)
    return np.stack(vectors, axis=0).astype(np.float32)


def make_relative_times(dataframe: pd.DataFrame, fps: float) -> np.ndarray:
    if "timestamp" in dataframe.columns:
        times = pd.to_numeric(dataframe["timestamp"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if (
            times.size > 0
            and np.all(np.isfinite(times))
            and np.all(np.diff(times) >= 0)
        ):
            return times - times[0]

    return np.arange(len(dataframe), dtype=np.float64) / max(fps, 1e-6)


def extract_task(dataframe: pd.DataFrame) -> str:
    if "task" in dataframe.columns:
        values = dataframe["task"].dropna()
        if len(values) > 0:
            return str(values.iloc[0])
    return "unknown_task"


@dataclass
class EpisodeData:
    episode_index: int
    list_index: int
    parquet: Path
    dataframe: pd.DataFrame
    times: np.ndarray
    state: np.ndarray
    action: np.ndarray
    task: str

    @property
    def frame_count(self) -> int:
        return int(len(self.dataframe))

    @property
    def duration(self) -> float:
        if self.times.size <= 1:
            return 0.0
        return float(self.times[-1])


class ReplayDataset:
    def __init__(
        self,
        root: Path,
        fps_override: float | None = None,
        camera_override: list[str] | None = None,
    ):
        self.root = root.expanduser().resolve()
        self.info = load_info(self.root)
        validate_16d_info(self.info)

        inferred_fps = float(self.info.get("fps", 15))
        self.fps = float(fps_override if fps_override is not None else inferred_fps)
        if not np.isfinite(self.fps) or self.fps <= 0:
            self.fps = 15.0

        inferred_cameras = infer_camera_keys(self.info)
        self.camera_keys = list(camera_override) if camera_override else inferred_cameras
        if not self.camera_keys:
            raise RuntimeError(
                "No observation.images.* keys found in meta/info.json"
            )

        self.high_camera = (
            pick_camera(self.camera_keys, ["cam_high"])
            or pick_camera(self.camera_keys, ["high"])
            or pick_camera(self.camera_keys, ["head"])
            or self.camera_keys[0]
        )
        self.left_wrist_camera = (
            pick_camera(self.camera_keys, ["left", "wrist"])
            or (self.camera_keys[1] if len(self.camera_keys) > 1 else None)
        )
        self.right_wrist_camera = (
            pick_camera(self.camera_keys, ["right", "wrist"])
            or (self.camera_keys[2] if len(self.camera_keys) > 2 else None)
        )

        self.episodes = build_episode_list(self.root)
        if not self.episodes:
            raise RuntimeError("No episodes found")

        self.episode_by_index = {
            episode["episode_index"]: episode for episode in self.episodes
        }
        self.list_position_by_episode = {
            episode["episode_index"]: episode["list_index"]
            for episode in self.episodes
        }

        self.state_names = infer_feature_names(
            self.info,
            "observation.state",
            DEFAULT_STATE_NAMES,
        )
        self.action_names = infer_feature_names(
            self.info,
            "action",
            DEFAULT_ACTION_NAMES,
        )

    def episode_indices(self) -> list[int]:
        return [episode["episode_index"] for episode in self.episodes]

    def find_nearest_valid_episode(self, requested: int) -> int:
        indices = self.episode_indices()
        if requested in self.episode_by_index:
            return requested
        return min(indices, key=lambda value: abs(value - requested))

    def previous_episode(self, episode_index: int) -> int:
        position = self.list_position_by_episode[episode_index]
        position = max(0, position - 1)
        return int(self.episodes[position]["episode_index"])

    def next_episode(self, episode_index: int) -> int:
        position = self.list_position_by_episode[episode_index]
        position = min(len(self.episodes) - 1, position + 1)
        return int(self.episodes[position]["episode_index"])

    @lru_cache(maxsize=4)
    def load_episode(self, episode_index: int) -> EpisodeData:
        if episode_index not in self.episode_by_index:
            raise KeyError(f"Unknown episode index: {episode_index}")

        episode = self.episode_by_index[episode_index]
        dataframe = load_episode_dataframe(episode, self.camera_keys)
        state = stack_vector_column(dataframe, "observation.state", expected_dim=16)
        action = stack_vector_column(dataframe, "action", expected_dim=16)
        times = make_relative_times(dataframe, self.fps)

        if len(state) != len(action) or len(state) != len(dataframe):
            raise RuntimeError(
                f"Length mismatch in episode {episode_index}: "
                f"df={len(dataframe)} state={len(state)} action={len(action)}"
            )

        return EpisodeData(
            episode_index=episode_index,
            list_index=int(episode["list_index"]),
            parquet=episode["parquet"],
            dataframe=dataframe,
            times=times,
            state=state,
            action=action,
            task=extract_task(dataframe),
        )

    def frame_count(self, episode_index: int) -> int:
        return self.load_episode(episode_index).frame_count

    @lru_cache(maxsize=768)
    def image_data_uri(
        self,
        episode_index: int,
        frame_index: int,
        camera_key: str | None,
    ) -> str:
        if camera_key is None:
            return make_placeholder_data_uri("camera not available")

        episode = self.load_episode(episode_index)
        frame_index = int(np.clip(frame_index, 0, max(episode.frame_count - 1, 0)))

        if camera_key not in episode.dataframe.columns:
            return make_placeholder_data_uri(short_camera_name(camera_key))

        value = episode.dataframe.iloc[frame_index][camera_key]
        image = decode_image_value(value, self.root)
        if image is None:
            return make_placeholder_data_uri(
                f"{short_camera_name(camera_key)} decode failed"
            )

        return bgr_to_data_uri(image)


# ======================================================================================
# Image decoding
# ======================================================================================

def array_to_bgr(array: np.ndarray) -> np.ndarray | None:
    array = np.asarray(array)

    if array.ndim == 1 and array.dtype == np.uint8:
        return cv2.imdecode(array, cv2.IMREAD_COLOR)

    if array.ndim == 3:
        if (
            array.shape[0] in (1, 3, 4)
            and array.shape[-1] not in (1, 3, 4)
        ):
            array = np.transpose(array, (1, 2, 0))

        if array.dtype != np.uint8:
            finite = array[np.isfinite(array)]
            maximum = float(np.max(finite)) if finite.size else 0.0
            if maximum <= 1.5:
                array = array * 255.0
            array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
            array = np.clip(array, 0, 255).astype(np.uint8)

        if array.shape[2] == 3:
            return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        if array.shape[2] == 4:
            return cv2.cvtColor(array, cv2.COLOR_RGBA2BGR)
        if array.shape[2] == 1:
            return cv2.cvtColor(array[:, :, 0], cv2.COLOR_GRAY2BGR)

    if array.ndim == 2:
        if array.dtype != np.uint8:
            finite = array[np.isfinite(array)]
            maximum = float(np.max(finite)) if finite.size else 0.0
            if maximum <= 1.5:
                array = array * 255.0
            array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
            array = np.clip(array, 0, 255).astype(np.uint8)
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)

    return None


def decode_image_value(value: Any, root: Path) -> np.ndarray | None:
    if value is None:
        return None

    if hasattr(value, "as_py"):
        value = value.as_py()

    try:
        if isinstance(value, float) and np.isnan(value):
            return None
    except Exception:
        pass

    if isinstance(value, dict):
        raw_bytes = value.get("bytes")
        if raw_bytes is not None:
            if isinstance(raw_bytes, memoryview):
                raw_bytes = raw_bytes.tobytes()
            elif isinstance(raw_bytes, np.ndarray):
                raw_bytes = raw_bytes.astype(np.uint8, copy=False).tobytes()
            else:
                raw_bytes = bytes(raw_bytes)

            encoded = np.frombuffer(raw_bytes, dtype=np.uint8)
            return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        path_value = value.get("path")
        if path_value is not None:
            path = Path(str(path_value))
            if not path.is_absolute():
                path = root / path
            if path.exists():
                return cv2.imread(str(path), cv2.IMREAD_COLOR)
            return None

        for nested_key in ("array", "data", "image"):
            if nested_key in value:
                return decode_image_value(value[nested_key], root)

    if isinstance(value, (bytes, bytearray, memoryview)):
        raw_bytes = value.tobytes() if isinstance(value, memoryview) else bytes(value)
        encoded = np.frombuffer(raw_bytes, dtype=np.uint8)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    if isinstance(value, str):
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        if path.exists():
            return cv2.imread(str(path), cv2.IMREAD_COLOR)
        return None

    if hasattr(value, "__array__"):
        return array_to_bgr(np.asarray(value))

    return None


def bgr_to_data_uri(image: np.ndarray, quality: int = 88) -> str:
    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not success:
        return make_placeholder_data_uri("JPEG encode failed")

    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


@lru_cache(maxsize=32)
def make_placeholder_data_uri(text: str) -> str:
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(
        image,
        text[:60],
        (24, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return bgr_to_data_uri(image)


# ======================================================================================
# Plot generation
# ======================================================================================

def safe_axis_range(values: list[np.ndarray]) -> list[float] | None:
    finite_parts = []
    for value in values:
        array = np.asarray(value, dtype=np.float64)
        finite = array[np.isfinite(array)]
        if finite.size:
            finite_parts.append(finite)

    if not finite_parts:
        return None

    merged = np.concatenate(finite_parts)
    minimum = float(np.min(merged))
    maximum = float(np.max(merged))

    if math.isclose(minimum, maximum, rel_tol=0.0, abs_tol=1e-9):
        margin = max(abs(minimum) * 0.05, 0.05)
    else:
        margin = max((maximum - minimum) * 0.08, 1e-4)

    return [minimum - margin, maximum + margin]


def make_group_figure(
    episode: EpisodeData,
    frame_index: int,
    group_key: str,
    plot_mode: str,
) -> go.Figure:
    group = GROUPS[group_key]
    indices = group["indices"]
    display_names = group["display_names"]

    frame_index = int(np.clip(frame_index, 0, max(episode.frame_count - 1, 0)))
    end = frame_index + 1
    x_visible = episode.times[:end]
    current_time = float(episode.times[frame_index]) if episode.frame_count else 0.0
    total_time = max(float(episode.times[-1]) if episode.frame_count else 0.0, 1e-6)

    figure = go.Figure()

    include_state = plot_mode in ("state", "overlay")
    include_action = plot_mode in ("action", "overlay")

    for local_index, dimension_index in enumerate(indices):
        label = display_names[local_index]

        if include_state:
            figure.add_trace(
                go.Scatter(
                    x=x_visible,
                    y=episode.state[:end, dimension_index],
                    mode="lines",
                    name=f"state {label}",
                    line={"width": 2},
                    hovertemplate=(
                        f"state {label}<br>"
                        "t=%{x:.3f}s<br>"
                        "value=%{y:.6f}<extra></extra>"
                    ),
                )
            )

        if include_action:
            figure.add_trace(
                go.Scatter(
                    x=x_visible,
                    y=episode.action[:end, dimension_index],
                    mode="lines",
                    name=f"action {label}",
                    line={"width": 2, "dash": "dash"},
                    hovertemplate=(
                        f"action {label}<br>"
                        "t=%{x:.3f}s<br>"
                        "value=%{y:.6f}<extra></extra>"
                    ),
                )
            )

    full_range_values: list[np.ndarray] = []
    if include_state:
        full_range_values.append(episode.state[:, indices])
    if include_action:
        full_range_values.append(episode.action[:, indices])
    y_range = safe_axis_range(full_range_values)

    figure.add_vline(
        x=current_time,
        line_width=1,
        line_dash="dot",
    )

    figure.update_layout(
        title={
            "text": group["title"],
            "x": 0.01,
            "xanchor": "left",
        },
        margin={"l": 58, "r": 24, "t": 55, "b": 48},
        height=360,
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 10},
        },
        xaxis={
            "title": "time / s",
            "range": [0.0, total_time],
            "showgrid": True,
        },
        yaxis={
            "title": "value",
            "range": y_range,
            "showgrid": True,
            "zeroline": True,
        },
        uirevision=f"{episode.episode_index}-{group_key}-{plot_mode}",
        template="plotly_white",
    )

    return figure


def make_current_value_rows(
    episode: EpisodeData,
    frame_index: int,
) -> list[dict]:
    frame_index = int(np.clip(frame_index, 0, max(episode.frame_count - 1, 0)))
    state = episode.state[frame_index]
    action = episode.action[frame_index]

    group_by_dimension = {}
    for group_key, group in GROUPS.items():
        for dimension_index in group["indices"]:
            group_by_dimension[dimension_index] = group["title"].split("/")[0].strip()

    rows = []
    for dimension_index in range(16):
        rows.append(
            {
                "index": dimension_index,
                "group": group_by_dimension.get(dimension_index, ""),
                "state_name": episode_state_name(dimension_index),
                "state": f"{float(state[dimension_index]):.6f}",
                "action_name": episode_action_name(dimension_index),
                "action": f"{float(action[dimension_index]):.6f}",
                "action-state": f"{float(action[dimension_index] - state[dimension_index]):.6f}",
            }
        )
    return rows


def episode_state_name(index: int) -> str:
    return APP_DATASET.state_names[index]


def episode_action_name(index: int) -> str:
    return APP_DATASET.action_names[index]


# Set during main().
APP_DATASET: ReplayDataset


# ======================================================================================
# Dash application
# ======================================================================================

PAGE_STYLE = {
    "fontFamily": "Arial, sans-serif",
    "background": "#f3f5f7",
    "minHeight": "100vh",
    "padding": "18px",
    "boxSizing": "border-box",
}

CARD_STYLE = {
    "background": "white",
    "border": "1px solid #dfe3e8",
    "borderRadius": "10px",
    "padding": "12px",
    "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
}

BUTTON_STYLE = {
    "height": "38px",
    "padding": "0 15px",
    "border": "1px solid #c8ced6",
    "borderRadius": "7px",
    "background": "white",
    "cursor": "pointer",
    "fontSize": "14px",
}

PRIMARY_BUTTON_STYLE = {
    **BUTTON_STYLE,
    "fontWeight": "700",
    "minWidth": "100px",
}

IMAGE_STYLE = {
    "width": "100%",
    "height": "auto",
    "display": "block",
    "background": "black",
    "borderRadius": "7px",
    "objectFit": "contain",
}


def build_layout(dataset: ReplayDataset, start_episode: int, start_rate: float, ui_fps: float):
    episode_options = [
        {
            "label": (
                f"episode_{episode['episode_index']:06d} "
                f"(list {episode['list_index']})"
            ),
            "value": int(episode["episode_index"]),
        }
        for episode in dataset.episodes
    ]

    initial_episode = dataset.load_episode(start_episode)
    initial_store = {
        "episode_index": int(start_episode),
        "frame_position": 0.0,
        "frame_index": 0,
        "playing": False,
        "ended": False,
        "rate": float(start_rate),
        "last_wall_time": None,
    }

    return html.Div(
        style=PAGE_STYLE,
        children=[
            dcc.Store(id="playback-store", data=initial_store),
            dcc.Interval(
                id="playback-timer",
                interval=max(20, int(round(1000.0 / max(ui_fps, 1.0)))),
                disabled=True,
                n_intervals=0,
            ),

            html.Div(
                style={**CARD_STYLE, "marginBottom": "14px"},
                children=[
                    html.Div(
                        style={
                            "display": "flex",
                            "gap": "10px",
                            "alignItems": "center",
                            "flexWrap": "wrap",
                        },
                        children=[
                            html.H2(
                                "LeRobot synchronized replay",
                                style={"margin": "0 18px 0 0"},
                            ),
                            html.Button(
                                "上一条",
                                id="previous-episode-button",
                                n_clicks=0,
                                style=BUTTON_STYLE,
                            ),
                            html.Button(
                                "下一条",
                                id="next-episode-button",
                                n_clicks=0,
                                style=BUTTON_STYLE,
                            ),
                            dcc.Dropdown(
                                id="episode-selector",
                                options=episode_options,
                                value=int(start_episode),
                                clearable=False,
                                searchable=True,
                                style={"width": "300px"},
                            ),
                            html.Button(
                                "播放",
                                id="play-pause-button",
                                n_clicks=0,
                                style=PRIMARY_BUTTON_STYLE,
                            ),
                            html.Button(
                                "重播",
                                id="restart-button",
                                n_clicks=0,
                                style=BUTTON_STYLE,
                            ),
                            html.Button(
                                "停止",
                                id="stop-button",
                                n_clicks=0,
                                style=BUTTON_STYLE,
                            ),
                            html.Button(
                                "上一帧",
                                id="previous-frame-button",
                                n_clicks=0,
                                style=BUTTON_STYLE,
                            ),
                            html.Button(
                                "下一帧",
                                id="next-frame-button",
                                n_clicks=0,
                                style=BUTTON_STYLE,
                            ),
                            html.Div("倍速:", style={"marginLeft": "8px"}),
                            dcc.Dropdown(
                                id="rate-selector",
                                options=[
                                    {"label": "0.25×", "value": 0.25},
                                    {"label": "0.5×", "value": 0.5},
                                    {"label": "1×", "value": 1.0},
                                    {"label": "1.5×", "value": 1.5},
                                    {"label": "2×", "value": 2.0},
                                ],
                                value=float(start_rate),
                                clearable=False,
                                style={"width": "105px"},
                            ),
                            html.Div("曲线:", style={"marginLeft": "8px"}),
                            dcc.Dropdown(
                                id="plot-mode-selector",
                                options=[
                                    {"label": "State 反馈", "value": "state"},
                                    {"label": "Action 目标", "value": "action"},
                                    {"label": "State + Action", "value": "overlay"},
                                ],
                                value="overlay",
                                clearable=False,
                                style={"width": "155px"},
                            ),
                        ],
                    ),
                    html.Div(
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "1fr auto",
                            "gap": "14px",
                            "alignItems": "center",
                            "marginTop": "12px",
                        },
                        children=[
                            html.Progress(
                                id="frame-progress",
                                value=0,
                                max=max(initial_episode.frame_count - 1, 1),
                                style={"width": "100%", "height": "18px"},
                            ),
                            html.Div(
                                id="playback-summary",
                                style={
                                    "fontFamily": "monospace",
                                    "whiteSpace": "nowrap",
                                },
                            ),
                        ],
                    ),
                    html.Div(
                        id="episode-metadata",
                        style={
                            "marginTop": "8px",
                            "fontSize": "13px",
                            "color": "#4c5561",
                            "wordBreak": "break-all",
                        },
                    ),
                ],
            ),

            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "gap": "14px",
                    "marginBottom": "14px",
                },
                children=[
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.H3(
                                id="high-camera-title",
                                style={"margin": "0 0 8px 0"},
                            ),
                            html.Img(id="high-camera-image", style=IMAGE_STYLE),
                        ],
                    ),
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.H3(
                                id="episode-title",
                                style={"margin": "0 0 10px 0"},
                            ),
                            html.Div(
                                id="status-panel",
                                style={
                                    "fontFamily": "monospace",
                                    "fontSize": "15px",
                                    "lineHeight": "1.8",
                                    "whiteSpace": "pre-wrap",
                                },
                            ),
                        ],
                    ),
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.H3(
                                id="left-camera-title",
                                style={"margin": "0 0 8px 0"},
                            ),
                            html.Img(id="left-camera-image", style=IMAGE_STYLE),
                        ],
                    ),
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.H3(
                                id="right-camera-title",
                                style={"margin": "0 0 8px 0"},
                            ),
                            html.Img(id="right-camera-image", style=IMAGE_STYLE),
                        ],
                    ),
                ],
            ),

            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "gap": "14px",
                },
                children=[
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            dcc.Graph(
                                id="left-arm-graph",
                                config=PLOT_CONFIG,
                            )
                        ],
                    ),
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            dcc.Graph(
                                id="left-gripper-graph",
                                config=PLOT_CONFIG,
                            )
                        ],
                    ),
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            dcc.Graph(
                                id="right-arm-graph",
                                config=PLOT_CONFIG,
                            )
                        ],
                    ),
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            dcc.Graph(
                                id="right-gripper-graph",
                                config=PLOT_CONFIG,
                            )
                        ],
                    ),
                ],
            ),

            html.Div(
                style={**CARD_STYLE, "marginTop": "14px"},
                children=[
                    html.H3(
                        "Current frame values / 当前帧数值",
                        style={"margin": "0 0 10px 0"},
                    ),
                    dash_table.DataTable(
                        id="current-value-table",
                        columns=[
                            {"name": "index", "id": "index"},
                            {"name": "group", "id": "group"},
                            {"name": "state name", "id": "state_name"},
                            {"name": "state", "id": "state"},
                            {"name": "action name", "id": "action_name"},
                            {"name": "action", "id": "action"},
                            {"name": "action-state", "id": "action-state"},
                        ],
                        data=[],
                        page_size=20,
                        style_table={"overflowX": "auto"},
                        style_cell={
                            "fontFamily": "monospace",
                            "fontSize": "12px",
                            "padding": "6px",
                            "textAlign": "left",
                        },
                        style_header={"fontWeight": "bold"},
                    ),
                ],
            ),
        ],
    )


def register_callbacks(app: Dash, dataset: ReplayDataset, ui_fps: float) -> None:
    timer_interval_ms = max(20, int(round(1000.0 / max(ui_fps, 1.0))))

    @app.callback(
        Output("playback-store", "data"),
        Output("episode-selector", "value"),
        Output("play-pause-button", "children"),
        Output("playback-timer", "disabled"),
        Output("playback-timer", "interval"),
        Input("playback-timer", "n_intervals"),
        Input("play-pause-button", "n_clicks"),
        Input("restart-button", "n_clicks"),
        Input("stop-button", "n_clicks"),
        Input("previous-frame-button", "n_clicks"),
        Input("next-frame-button", "n_clicks"),
        Input("previous-episode-button", "n_clicks"),
        Input("next-episode-button", "n_clicks"),
        Input("episode-selector", "value"),
        Input("rate-selector", "value"),
        State("playback-store", "data"),
    )
    def update_playback_state(
        _timer_ticks,
        _play_clicks,
        _restart_clicks,
        _stop_clicks,
        _previous_frame_clicks,
        _next_frame_clicks,
        _previous_episode_clicks,
        _next_episode_clicks,
        selected_episode,
        selected_rate,
        stored,
    ):
        if not stored:
            selected_episode = dataset.find_nearest_valid_episode(
                int(selected_episode or dataset.episode_indices()[0])
            )
            stored = {
                "episode_index": selected_episode,
                "frame_position": 0.0,
                "frame_index": 0,
                "playing": False,
                "ended": False,
                "rate": float(selected_rate or 1.0),
                "last_wall_time": None,
            }

        state = dict(stored)
        current_episode = dataset.find_nearest_valid_episode(
            int(state.get("episode_index", selected_episode))
        )
        state["episode_index"] = current_episode
        state["rate"] = float(selected_rate or state.get("rate", 1.0))
        trigger = ctx.triggered_id
        now = time.monotonic()

        if trigger == "episode-selector":
            requested = dataset.find_nearest_valid_episode(int(selected_episode))
            if requested != current_episode:
                current_episode = requested
                state.update(
                    {
                        "episode_index": current_episode,
                        "frame_position": 0.0,
                        "frame_index": 0,
                        "playing": False,
                        "ended": False,
                        "last_wall_time": None,
                    }
                )

        elif trigger == "previous-episode-button":
            current_episode = dataset.previous_episode(current_episode)
            state.update(
                {
                    "episode_index": current_episode,
                    "frame_position": 0.0,
                    "frame_index": 0,
                    "playing": False,
                    "ended": False,
                    "last_wall_time": None,
                }
            )

        elif trigger == "next-episode-button":
            current_episode = dataset.next_episode(current_episode)
            state.update(
                {
                    "episode_index": current_episode,
                    "frame_position": 0.0,
                    "frame_index": 0,
                    "playing": False,
                    "ended": False,
                    "last_wall_time": None,
                }
            )

        frame_count = dataset.frame_count(current_episode)
        last_frame = max(frame_count - 1, 0)

        if trigger == "play-pause-button":
            if state.get("playing", False):
                state["playing"] = False
                state["last_wall_time"] = None
            else:
                if int(state.get("frame_index", 0)) >= last_frame:
                    state["frame_position"] = 0.0
                    state["frame_index"] = 0
                    state["ended"] = False
                state["playing"] = True
                state["last_wall_time"] = now

        elif trigger == "restart-button":
            state["frame_position"] = 0.0
            state["frame_index"] = 0
            state["playing"] = True
            state["ended"] = False
            state["last_wall_time"] = now

        elif trigger == "stop-button":
            state["frame_position"] = 0.0
            state["frame_index"] = 0
            state["playing"] = False
            state["ended"] = False
            state["last_wall_time"] = None

        elif trigger == "previous-frame-button":
            frame_index = max(0, int(state.get("frame_index", 0)) - 1)
            state["frame_index"] = frame_index
            state["frame_position"] = float(frame_index)
            state["playing"] = False
            state["ended"] = False
            state["last_wall_time"] = None

        elif trigger == "next-frame-button":
            frame_index = min(last_frame, int(state.get("frame_index", 0)) + 1)
            state["frame_index"] = frame_index
            state["frame_position"] = float(frame_index)
            state["playing"] = False
            state["ended"] = frame_index >= last_frame
            state["last_wall_time"] = None

        elif trigger == "rate-selector":
            state["rate"] = float(selected_rate or 1.0)
            if state.get("playing", False):
                state["last_wall_time"] = now

        elif trigger == "playback-timer" and state.get("playing", False):
            last_wall_time = state.get("last_wall_time")
            if last_wall_time is None:
                state["last_wall_time"] = now
            else:
                elapsed = max(0.0, now - float(last_wall_time))
                position = float(state.get("frame_position", 0.0))
                position += elapsed * dataset.fps * float(state["rate"])

                if position >= last_frame:
                    position = float(last_frame)
                    state["playing"] = False
                    state["ended"] = True
                    state["last_wall_time"] = None
                else:
                    state["last_wall_time"] = now

                state["frame_position"] = position
                state["frame_index"] = int(np.clip(math.floor(position), 0, last_frame))

        state["episode_index"] = int(current_episode)
        state["frame_index"] = int(
            np.clip(int(state.get("frame_index", 0)), 0, last_frame)
        )
        state["frame_position"] = float(
            np.clip(float(state.get("frame_position", 0.0)), 0.0, float(last_frame))
        )

        play_label = "暂停" if state.get("playing", False) else "播放"
        timer_disabled = not bool(state.get("playing", False))

        return (
            state,
            int(current_episode),
            play_label,
            timer_disabled,
            timer_interval_ms,
        )

    @app.callback(
        Output("episode-title", "children"),
        Output("status-panel", "children"),
        Output("playback-summary", "children"),
        Output("episode-metadata", "children"),
        Output("frame-progress", "value"),
        Output("frame-progress", "max"),
        Output("high-camera-title", "children"),
        Output("left-camera-title", "children"),
        Output("right-camera-title", "children"),
        Output("high-camera-image", "src"),
        Output("left-camera-image", "src"),
        Output("right-camera-image", "src"),
        Output("left-arm-graph", "figure"),
        Output("left-gripper-graph", "figure"),
        Output("right-arm-graph", "figure"),
        Output("right-gripper-graph", "figure"),
        Output("current-value-table", "data"),
        Input("playback-store", "data"),
        Input("plot-mode-selector", "value"),
    )
    def render_frame(stored, plot_mode):
        episode_index = int(stored["episode_index"])
        episode = dataset.load_episode(episode_index)
        frame_index = int(
            np.clip(
                int(stored.get("frame_index", 0)),
                0,
                max(episode.frame_count - 1, 0),
            )
        )

        playing = bool(stored.get("playing", False))
        ended = bool(stored.get("ended", False))
        if playing:
            status = "PLAYING"
        elif ended:
            status = "ENDED"
        elif frame_index == 0:
            status = "STOPPED / READY"
        else:
            status = "PAUSED"

        current_time = (
            float(episode.times[frame_index]) if episode.frame_count else 0.0
        )
        percentage = (
            100.0 * frame_index / max(episode.frame_count - 1, 1)
        )

        episode_title = (
            f"episode_{episode.episode_index:06d} "
            f"(list index {episode.list_index})"
        )

        status_panel = (
            f"status : {status}\n"
            f"frame  : {frame_index} / {max(episode.frame_count - 1, 0)}\n"
            f"time   : {current_time:.3f} / {episode.duration:.3f} s\n"
            f"rate   : {float(stored.get('rate', 1.0)):g}×\n"
            f"task   : {episode.task}\n"
            f"progress: {percentage:.1f}%"
        )

        playback_summary = (
            f"EP {episode.episode_index:06d} | "
            f"frame {frame_index}/{max(episode.frame_count - 1, 0)} | "
            f"{current_time:.2f}s | {status}"
        )

        episode_metadata = (
            f"parquet: {episode.parquet} | "
            f"frames: {episode.frame_count} | "
            f"dataset fps: {dataset.fps:g}"
        )

        high_uri = dataset.image_data_uri(
            episode_index,
            frame_index,
            dataset.high_camera,
        )
        left_uri = dataset.image_data_uri(
            episode_index,
            frame_index,
            dataset.left_wrist_camera,
        )
        right_uri = dataset.image_data_uri(
            episode_index,
            frame_index,
            dataset.right_wrist_camera,
        )

        return (
            episode_title,
            status_panel,
            playback_summary,
            episode_metadata,
            frame_index,
            max(episode.frame_count - 1, 1),
            short_camera_name(dataset.high_camera),
            short_camera_name(dataset.left_wrist_camera),
            short_camera_name(dataset.right_wrist_camera),
            high_uri,
            left_uri,
            right_uri,
            make_group_figure(episode, frame_index, "left_arm", plot_mode),
            make_group_figure(episode, frame_index, "left_gripper", plot_mode),
            make_group_figure(episode, frame_index, "right_arm", plot_mode),
            make_group_figure(episode, frame_index, "right_gripper", plot_mode),
            make_current_value_rows(episode, frame_index),
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        required=True,
        help="LeRobot v2.1 dataset root.",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Initial actual episode_index.",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=1.0,
        help="Initial playback rate.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override dataset fps from meta/info.json.",
    )
    parser.add_argument(
        "--ui-fps",
        type=float,
        default=15.0,
        help=(
            "Maximum browser refresh frequency. "
            "The displayed image and curves always use the same frame index."
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Dash host. Use 0.0.0.0 for LAN access.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
    )
    parser.add_argument(
        "--cameras",
        nargs="*",
        default=None,
        help="Optional explicit observation.images.* camera keys.",
    )
    return parser.parse_args()


def main() -> None:
    global APP_DATASET

    args = parse_arguments()
    APP_DATASET = ReplayDataset(
        root=Path(args.root),
        fps_override=args.fps,
        camera_override=args.cameras,
    )

    start_episode = APP_DATASET.find_nearest_valid_episode(args.episode_index)
    start_rate = float(args.rate)
    if not np.isfinite(start_rate) or start_rate <= 0:
        start_rate = 1.0

    ui_fps = float(args.ui_fps)
    if not np.isfinite(ui_fps) or ui_fps <= 0:
        ui_fps = 15.0

    log("=" * 100)
    log("LeRobot synchronized web replay")
    log("=" * 100)
    log(f"root             : {APP_DATASET.root}")
    log(f"episodes         : {len(APP_DATASET.episodes)}")
    log(f"initial episode  : {start_episode}")
    log(f"dataset fps      : {APP_DATASET.fps}")
    log(f"UI fps           : {ui_fps}")
    log(f"cameras          : {APP_DATASET.camera_keys}")
    log(f"high camera      : {APP_DATASET.high_camera}")
    log(f"left wrist       : {APP_DATASET.left_wrist_camera}")
    log(f"right wrist      : {APP_DATASET.right_wrist_camera}")
    log(f"URL              : http://{args.host}:{args.port}")
    log("=" * 100)

    app = Dash(
        __name__,
        title="LeRobot synchronized replay",
        update_title=None,
    )
    app.layout = build_layout(
        APP_DATASET,
        start_episode=start_episode,
        start_rate=start_rate,
        ui_fps=ui_fps,
    )
    register_callbacks(app, APP_DATASET, ui_fps=ui_fps)

    if args.open_browser:
        browser_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
        url = f"http://{browser_host}:{args.port}"
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    run_kwargs = {
        "host": args.host,
        "port": args.port,
        "debug": bool(args.debug),
    }

    # Dash 3 uses app.run(); older versions may only expose run_server().
    if hasattr(app, "run"):
        app.run(**run_kwargs)
    else:
        app.run_server(**run_kwargs)


if __name__ == "__main__":
    main()
