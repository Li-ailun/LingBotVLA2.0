#!/usr/bin/env python3
from pathlib import Path
import argparse
import importlib
import json
import math
import queue
import shlex
import threading
import time
import sys

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
_schema = importlib.import_module("myscripts.16d.common.schema")


def validate_16d_info(info):
    for key, expected_names in (
        ("observation.state", _schema.STATE_NAMES),
        ("action", _schema.ACTION_NAMES),
    ):
        feature = info.get("features", {}).get(key, {})
        shape = feature.get("shape")
        if not isinstance(shape, (list, tuple)) or list(shape) != [16]:
            raise ValueError(f"{key}: expected metadata shape [16], got {shape}")
        names = feature.get("names")
        if names is not None and tuple(names) != tuple(expected_names):
            raise ValueError(f"{key}: feature names do not match {_schema.SCHEMA_NAME}")


HELP_TEXT = """

python /home/lixiang/PROJECT/LingBotVLA2.0/myscripts/16d/lerobot_tools/replay_lerobot_v21_parquet_images_keyboard.py \
  --root /home/lixiang/PROJECT/LingBotVLA2.0/datasets/lerobot/lixiang/hang_wire_stripper_on_wall_hook_16d \
  --episode-index 0 \
  --rate 1

  
Commands:
  h             help
  ls            list episodes
  p             pause/resume current episode
  n             next episode
  b             previous episode
  g 35          go to episode_000035
  i 10          go to list index 10
  rate 0.5      replay current episode at 0.5x from start
  rate 1        replay current episode at 1x from start
  rate 2        replay current episode at 2x from start
  r             replay current episode from start
  s             stop current episode
  q             quit

Keyboard shortcuts inside OpenCV window:
  h, p, n, b, r, s, q
"""


def log(msg):
    print(msg, flush=True)


def load_info(root: Path):
    p = root / "meta" / "info.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    return json.loads(p.read_text())


def infer_camera_keys(info):
    keys = []
    for k, v in info.get("features", {}).items():
        if k.startswith("observation.images."):
            keys.append(k)

    def priority(k):
        lk = k.lower()
        if "cam_high" in lk or "head" in lk or "high" in lk:
            return 0
        if "left" in lk and "wrist" in lk:
            return 1
        if "right" in lk and "wrist" in lk:
            return 2
        return 10

    return sorted(keys, key=priority)


def short_cam_name(cam_key):
    return cam_key.split(".")[-1]


def find_parquets(root: Path):
    ps = sorted((root / "data").glob("**/*.parquet"))
    if not ps:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")
    return ps


def build_episode_list(root: Path):
    episodes = []
    for p in find_parquets(root):
        try:
            df_idx = pd.read_parquet(p, columns=["episode_index"])
        except Exception:
            df_idx = pd.read_parquet(p)

        if "episode_index" in df_idx.columns:
            ep_indices = sorted(set(int(x) for x in df_idx["episode_index"].dropna().unique()))
        else:
            ep_indices = []
            stem = p.stem
            for token in stem.replace("-", "_").split("_"):
                if token.isdigit():
                    ep_indices.append(int(token))
                    break
            if not ep_indices:
                ep_indices.append(len(episodes))

        for ep_idx in ep_indices:
            episodes.append({
                "episode_index": int(ep_idx),
                "parquet": p,
            })

    episodes = sorted(episodes, key=lambda x: x["episode_index"])
    for i, ep in enumerate(episodes):
        ep["list_index"] = i
    return episodes


def find_by_episode_index(episodes, ep_idx):
    for i, ep in enumerate(episodes):
        if ep["episode_index"] == ep_idx:
            return i
    return None


def load_episode_dataframe(ep, cam_keys):
    p = ep["parquet"]

    base_cols = ["episode_index", "frame_index", "timestamp"]
    cols = []
    for c in base_cols + cam_keys:
        cols.append(c)

    try:
        df = pd.read_parquet(p, columns=cols)
    except Exception:
        df = pd.read_parquet(p)

    if "episode_index" in df.columns:
        df = df[df["episode_index"] == ep["episode_index"]].copy()

    if "frame_index" in df.columns:
        df = df.sort_values("frame_index").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df


def decode_image_value(v, root: Path):
    if v is None:
        return None

    # pandas sometimes gives NaN for missing values.
    try:
        if isinstance(v, float) and np.isnan(v):
            return None
    except Exception:
        pass

    # LeRobot / datasets image object often becomes dict: {"bytes": ..., "path": ...}
    if isinstance(v, dict):
        if v.get("bytes") is not None:
            data = v["bytes"]
            if isinstance(data, memoryview):
                data = data.tobytes()
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img

        if v.get("path") is not None:
            path = Path(v["path"])
            if not path.is_absolute():
                path = root / path
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            return img

        for key in ["array", "data", "image"]:
            if key in v:
                return decode_image_value(v[key], root)

    # Raw encoded bytes.
    if isinstance(v, (bytes, bytearray, memoryview)):
        data = v.tobytes() if isinstance(v, memoryview) else bytes(v)
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img

    # Path string.
    if isinstance(v, str):
        path = Path(v)
        if not path.is_absolute():
            path = root / path
        if path.exists():
            return cv2.imread(str(path), cv2.IMREAD_COLOR)
        return None

    # PIL image.
    if hasattr(v, "__array__"):
        arr = np.asarray(v)
        return array_to_bgr(arr)

    return None


def array_to_bgr(arr):
    arr = np.asarray(arr)

    if arr.ndim == 1 and arr.dtype == np.uint8:
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img

    if arr.ndim == 3:
        # CHW -> HWC
        if arr.shape[0] in [1, 3, 4] and arr.shape[-1] not in [1, 3, 4]:
            arr = np.transpose(arr, (1, 2, 0))

        if arr.dtype != np.uint8:
            arr = np.asarray(arr)
            if arr.max() <= 1.5:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        if arr.shape[2] == 3:
            # Usually arrays from PIL/datasets are RGB, convert to BGR for OpenCV.
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        if arr.shape[2] == 4:
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        if arr.shape[2] == 1:
            return cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)

    if arr.ndim == 2:
        if arr.dtype != np.uint8:
            if arr.max() <= 1.5:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    return None


def put_text(img, text, org, scale=0.7, color=(255, 255, 255), bg=(0, 0, 0), thickness=2):
    x, y = org
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, bg, thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def resize_with_pad(frame, box_w, box_h):
    canvas = np.zeros((box_h, box_w, 3), dtype=np.uint8)
    if frame is None:
        return canvas

    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return canvas

    scale = min(box_w / w, box_h / h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

    x0 = (box_w - nw) // 2
    y0 = (box_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def pick_cam(cam_keys, includes, fallback_idx=None):
    for cam in cam_keys:
        lk = cam.lower()
        if all(x in lk for x in includes):
            return cam
    if fallback_idx is not None and 0 <= fallback_idx < len(cam_keys):
        return cam_keys[fallback_idx]
    return None


def make_info_panel(width, height, ep, frame_idx, total_frames, fps, rate, paused, stopped):
    panel = np.zeros((height, width, 3), dtype=np.uint8)

    ep_idx = ep["episode_index"]
    list_idx = ep["list_index"]
    status = "STOPPED" if stopped else ("PAUSED" if paused else "PLAYING")
    t = frame_idx / fps if fps > 0 else 0.0

    y = 45
    put_text(panel, f"episode_{ep_idx:06d}", (30, y), scale=1.05)
    y += 50
    put_text(panel, f"list index: {list_idx}", (30, y), scale=0.8)
    y += 42
    put_text(panel, f"status: {status}", (30, y), scale=0.8)
    y += 42
    put_text(panel, f"frame: {frame_idx}/{total_frames}", (30, y), scale=0.8)
    y += 42
    put_text(panel, f"time: {t:.2f}s", (30, y), scale=0.8)
    y += 42
    put_text(panel, f"rate: {rate:g}x", (30, y), scale=0.8)
    y += 42
    put_text(panel, f"parquet:", (30, y), scale=0.62)
    y += 30
    put_text(panel, ep["parquet"].name, (30, y), scale=0.62)
    y += 45
    put_text(panel, "terminal commands:", (30, y), scale=0.65)
    y += 32
    put_text(panel, "h ls p n b g 35 i 10", (30, y), scale=0.62)
    y += 30
    put_text(panel, "rate 0.5 / 1 / 2   r s q", (30, y), scale=0.62)

    return panel


def compose_canvas(frames, cam_keys, ep, frame_idx, total_frames, fps, rate, paused, stopped):
    if len(cam_keys) >= 3:
        high_key = pick_cam(cam_keys, ["high"], fallback_idx=0)
        left_key = pick_cam(cam_keys, ["left", "wrist"], fallback_idx=1)
        right_key = pick_cam(cam_keys, ["right", "wrist"], fallback_idx=2)

        high = resize_with_pad(frames.get(high_key), 640, 360)
        left = resize_with_pad(frames.get(left_key), 640, 480)
        right = resize_with_pad(frames.get(right_key), 640, 480)

        put_text(high, short_cam_name(high_key), (15, 32), scale=0.75)
        put_text(left, short_cam_name(left_key), (15, 32), scale=0.75)
        put_text(right, short_cam_name(right_key), (15, 32), scale=0.75)

        info_panel = make_info_panel(640, 360, ep, frame_idx, total_frames, fps, rate, paused, stopped)

        top = np.concatenate([high, info_panel], axis=1)
        bottom = np.concatenate([left, right], axis=1)
        return np.concatenate([top, bottom], axis=0)

    # Generic fallback layout.
    box_w, box_h = 640, 360
    panels = []
    panels.append(make_info_panel(box_w, box_h, ep, frame_idx, total_frames, fps, rate, paused, stopped))

    for cam in cam_keys:
        img = resize_with_pad(frames.get(cam), box_w, box_h)
        put_text(img, short_cam_name(cam), (15, 32), scale=0.75)
        panels.append(img)

    cols = 2
    rows = math.ceil(len(panels) / cols)
    while len(panels) < rows * cols:
        panels.append(np.zeros((box_h, box_w, 3), dtype=np.uint8))

    row_imgs = []
    for r in range(rows):
        row_imgs.append(np.concatenate(panels[r * cols:(r + 1) * cols], axis=1))
    return np.concatenate(row_imgs, axis=0)


def print_help():
    log(HELP_TEXT)


def list_episodes(episodes, current_i):
    log("")
    log("Episodes:")
    for i, ep in enumerate(episodes):
        mark = "=>" if i == current_i else "  "
        log(f"{mark} list_index={i:04d}  episode_{ep['episode_index']:06d}  {ep['parquet'].name}")
    log("")


def input_thread_fn(cmd_q):
    while True:
        try:
            line = input()
        except EOFError:
            break
        except KeyboardInterrupt:
            cmd_q.put("q")
            break
        cmd_q.put(line.strip())


def parse_rate(x):
    try:
        r = float(x)
    except Exception:
        return None
    if not np.isfinite(r) or r <= 0:
        return None
    return r


def handle_command(cmd, state, episodes, current_i):
    cmd = cmd.strip()
    if not cmd:
        return None

    try:
        parts = shlex.split(cmd)
    except Exception:
        parts = cmd.split()

    if not parts:
        return None

    op = parts[0].lower()

    if op in ["h", "help"]:
        print_help()
        return None

    if op == "ls":
        list_episodes(episodes, current_i)
        return None

    if op == "p":
        if state.get("stopped", False):
            state["stopped"] = False
            state["paused"] = False
            return ("replay", current_i)
        state["paused"] = not state.get("paused", False)
        log(f"[CMD] paused={state['paused']}")
        return None

    if op == "n":
        ni = min(current_i + 1, len(episodes) - 1)
        state["paused"] = False
        state["stopped"] = False
        return ("goto", ni)

    if op == "b":
        ni = max(current_i - 1, 0)
        state["paused"] = False
        state["stopped"] = False
        return ("goto", ni)

    if op == "g":
        if len(parts) < 2:
            log("[ERR] usage: g 35")
            return None
        try:
            ep_idx = int(parts[1])
        except Exception:
            log("[ERR] invalid episode index")
            return None
        ni = find_by_episode_index(episodes, ep_idx)
        if ni is None:
            log(f"[ERR] episode_{ep_idx:06d} not found")
            return None
        state["paused"] = False
        state["stopped"] = False
        return ("goto", ni)

    if op == "i":
        if len(parts) < 2:
            log("[ERR] usage: i 10")
            return None
        try:
            ni = int(parts[1])
        except Exception:
            log("[ERR] invalid list index")
            return None
        if not (0 <= ni < len(episodes)):
            log(f"[ERR] list index out of range: 0~{len(episodes)-1}")
            return None
        state["paused"] = False
        state["stopped"] = False
        return ("goto", ni)

    if op == "rate":
        if len(parts) < 2:
            log("[ERR] usage: rate 0.5")
            return None
        r = parse_rate(parts[1])
        if r is None:
            log("[ERR] invalid rate")
            return None
        state["rate"] = r
        state["paused"] = False
        state["stopped"] = False
        log(f"[CMD] rate={r:g}x, replay current episode from start")
        return ("replay", current_i)

    if op == "r":
        state["paused"] = False
        state["stopped"] = False
        return ("replay", current_i)

    if op == "s":
        state["paused"] = False
        state["stopped"] = True
        return ("stop", current_i)

    if op == "q":
        state["quit"] = True
        return ("quit", current_i)

    log(f"[ERR] unknown command: {cmd}")
    log("Type h for help.")
    return None


def key_to_command(key):
    if key < 0:
        return None
    ch = chr(key & 0xFF)
    if ch in ["h", "p", "n", "b", "r", "s", "q"]:
        return ch
    return None


def poll_commands(cmd_q, state, episodes, current_i):
    while True:
        try:
            cmd = cmd_q.get_nowait()
        except queue.Empty:
            break
        action = handle_command(cmd, state, episodes, current_i)
        if action is not None:
            return action
    return None


def make_stopped_canvas(ep, cam_keys, fps, rate):
    frames = {cam: None for cam in cam_keys}
    canvas = compose_canvas(
        frames=frames,
        cam_keys=cam_keys,
        ep=ep,
        frame_idx=0,
        total_frames=0,
        fps=fps,
        rate=rate,
        paused=False,
        stopped=True,
    )
    put_text(canvas, "STOPPED", (70, 90), scale=1.4, color=(0, 255, 255))
    put_text(canvas, "type r/p to replay, n/b/g/i to switch, q to quit", (70, 145), scale=0.75, color=(0, 255, 255))
    return canvas


def show_idle_screen(window_name, cmd_q, state, episodes, current_i, cam_keys, fps):
    ep = episodes[current_i]
    canvas = make_stopped_canvas(ep, cam_keys, fps, state["rate"])

    while not state.get("quit", False):
        action = poll_commands(cmd_q, state, episodes, current_i)
        if action is not None:
            return action

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(30) & 0xFF
        cmd = key_to_command(key)
        if cmd is not None:
            action = handle_command(cmd, state, episodes, current_i)
            if action is not None:
                return action

    return ("quit", current_i)


def decode_frame_row(row, root, cam_keys):
    frames = {}
    for cam in cam_keys:
        if cam not in row:
            frames[cam] = None
        else:
            frames[cam] = decode_image_value(row[cam], root)
    return frames


def play_episode(root, window_name, cmd_q, state, episodes, current_i, cam_keys, fps):
    ep = episodes[current_i]
    ep_idx = ep["episode_index"]

    try:
        df = load_episode_dataframe(ep, cam_keys)
    except Exception as e:
        log(f"[ERR] failed to load episode_{ep_idx:06d}: {e}")
        state["stopped"] = True
        return ("stop", current_i)

    total_frames = len(df)
    if total_frames <= 0:
        log(f"[WARN] episode_{ep_idx:06d}: empty dataframe")
        state["stopped"] = True
        return ("stop", current_i)

    log("")
    log(f"[PLAY] list_index={current_i} episode_{ep_idx:06d} frames={total_frames} rate={state['rate']:g}x parquet={ep['parquet']}")

    frame_idx = 0
    last_frames = {cam: None for cam in cam_keys}

    while frame_idx < total_frames and not state.get("quit", False):
        action = poll_commands(cmd_q, state, episodes, current_i)
        if action is not None:
            return action

        if state.get("paused", False):
            canvas = compose_canvas(
                frames=last_frames,
                cam_keys=cam_keys,
                ep=ep,
                frame_idx=frame_idx,
                total_frames=total_frames,
                fps=fps,
                rate=state["rate"],
                paused=True,
                stopped=False,
            )
            cv2.imshow(window_name, canvas)

            key = cv2.waitKey(30) & 0xFF
            cmd = key_to_command(key)
            if cmd is not None:
                action = handle_command(cmd, state, episodes, current_i)
                if action is not None:
                    return action
            continue

        t0 = time.time()

        row = df.iloc[frame_idx]
        frames = decode_frame_row(row, root, cam_keys)
        last_frames = frames

        canvas = compose_canvas(
            frames=frames,
            cam_keys=cam_keys,
            ep=ep,
            frame_idx=frame_idx,
            total_frames=total_frames,
            fps=fps,
            rate=state["rate"],
            paused=False,
            stopped=False,
        )
        cv2.imshow(window_name, canvas)

        key = cv2.waitKey(1) & 0xFF
        cmd = key_to_command(key)
        if cmd is not None:
            action = handle_command(cmd, state, episodes, current_i)
            if action is not None:
                return action

        frame_idx += 1

        delay = 1.0 / max(fps * state["rate"], 1e-6)
        elapsed = time.time() - t0
        if elapsed < delay:
            time.sleep(delay - elapsed)

    if state.get("quit", False):
        return ("quit", current_i)

    return ("ended", current_i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="LeRobot v2.1 dataset root")
    ap.add_argument("--episode-index", type=int, default=0)
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--window-name", default="LeRobot v2.1 parquet image replay")
    ap.add_argument("--window-scale", type=float, default=1.0)
    ap.add_argument("--cameras", nargs="*", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    info = load_info(root)
    validate_16d_info(info)

    cam_keys = args.cameras if args.cameras else infer_camera_keys(info)
    if not cam_keys:
        raise RuntimeError("No observation.images.* camera keys found in meta/info.json")

    fps = float(args.fps if args.fps is not None else info.get("fps", 15))
    if fps <= 0:
        fps = 15.0

    episodes = build_episode_list(root)
    if not episodes:
        raise RuntimeError("No episodes found")

    start_i = find_by_episode_index(episodes, args.episode_index)
    if start_i is None:
        log(f"[WARN] episode_{args.episode_index:06d} not found, start from list index 0")
        start_i = 0

    state = {
        "paused": False,
        "stopped": False,
        "quit": False,
        "rate": max(float(args.rate), 1e-6),
    }

    cmd_q = queue.Queue()
    threading.Thread(target=input_thread_fn, args=(cmd_q,), daemon=True).start()

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(args.window_name, int(1280 * args.window_scale), int(840 * args.window_scale))

    log("")
    log("[INFO] LeRobot parquet image replay started")
    log(f"[INFO] root: {root}")
    log(f"[INFO] fps: {fps}")
    log(f"[INFO] cameras: {cam_keys}")
    log(f"[INFO] episodes: {len(episodes)}")
    print_help()

    current_i = start_i

    while not state.get("quit", False):
        if state.get("stopped", False):
            action, target_i = show_idle_screen(
                window_name=args.window_name,
                cmd_q=cmd_q,
                state=state,
                episodes=episodes,
                current_i=current_i,
                cam_keys=cam_keys,
                fps=fps,
            )
        else:
            action, target_i = play_episode(
                root=root,
                window_name=args.window_name,
                cmd_q=cmd_q,
                state=state,
                episodes=episodes,
                current_i=current_i,
                cam_keys=cam_keys,
                fps=fps,
            )

        if action == "quit":
            break

        if action in ["goto", "replay"]:
            current_i = target_i
            state["stopped"] = False
            continue

        if action == "stop":
            current_i = target_i
            state["stopped"] = True
            continue

        if action == "ended":
            if current_i + 1 < len(episodes):
                current_i += 1
                state["stopped"] = False
            else:
                log("[INFO] reached last episode")
                if args.loop:
                    current_i = 0
                    state["stopped"] = False
                else:
                    state["stopped"] = True
            continue

    cv2.destroyAllWindows()
    log("[DONE] quit")


if __name__ == "__main__":
    main()
