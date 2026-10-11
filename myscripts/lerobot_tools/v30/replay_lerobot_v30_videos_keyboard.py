#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Keyboard-controlled 3-camera replay of RoboTwin LeRobot v3.0 video episodes.


cd /home/lixiang/PROJECT/LingBotVLA2.0
conda activate lingbotvla
python myscripts/lerobot_tools/v30/replay_lerobot_v30_videos_keyboard.py \
  --root "$PWD/RoboTwin2_data/lerobot_dataset/RoboTwin_lerobot_v30" \
  --episode-index 0 \
  --rate 1 \
  --backend browser

  
LeRobot v3.0 is NOT the v2.1 embedded-image-in-Parquet format.  Video
metadata lives under meta/episodes/*.parquet and frames live in MP4 files.
This player uses system FFmpeg to decode each camera's independent MP4
video-file/time interval, and refuses to decode past the episode's declared end (prevents accidentally
showing the following episode's first frame when a tail frame is missing).

This is a visualization tool. It never writes or edits the dataset.

Frame seeking uses zero-based episode-local frame indices. Seeking automatically
pauses playback and FFmpeg decodes only the requested episode-local frame.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
import shutil
import subprocess
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import re
import webbrowser
from pathlib import Path
import queue
import shlex
import threading
import time
from typing import Any

import cv2
import numpy as np
import pandas as pd

CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)

HELP = """
Commands in terminal (also: h p n b r s q in the OpenCV window):
  h              help
  ls             show episode list (ls 20 shows 20 around current)
  p              pause / resume
  n              next episode
  b              previous episode
  g 35           go to episode_index 35
  i 10           go to list index 10
  rate 0.5       replay this episode at 0.5x
  rate 1         replay this episode at 1x (dataset FPS)
  rate 2         replay this episode at 2x
  f 64           display frame_index=64 of current episode, pause automatically
  frame 64       same as f 64 (zero-based frame index)
  [ / ]          previous / next frame, stay paused
  r              replay current episode from the beginning
  s              stop current episode
  q              quit

Red 'MISSING TAIL FRAME' panels are intentional: they identify a camera
whose metadata interval has fewer frames than the episode's state/action.
No next-episode image is used as a substitute. This tool doesn't prove
physical camera/state synchronization for otherwise valid frames.
"""


def log(msg: str) -> None:
    print(msg, flush=True)


def read_info(root: Path) -> dict[str, Any]:
    path = root / "meta" / "info.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def camera_keys(info: dict, selected: list[str] | None) -> list[str]:
    if selected:
        keys = [s if s.startswith("observation.images.") else f"observation.images.{s}" for s in selected]
    else:
        features = info.get("features", {})
        keys = [k for k in CAMERAS if k in features]
        if not keys:
            keys = sorted(k for k in features if k.startswith("observation.images."))
    if not keys:
        raise ValueError("No camera feature keys found in meta/info.json")
    unknown = [k for k in keys if k not in info.get("features", {})]
    if unknown:
        raise ValueError(f"Unknown cameras in metadata: {unknown}")
    return keys


def video_fields(cam: str) -> list[str]:
    base = f"videos/{cam}/"
    return [base + suffix for suffix in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")]


@dataclass(frozen=True)
class Segment:
    camera: str
    path: Path
    first: int  # zero-based, inclusive MP4 frame index
    end: int    # zero-based, exclusive MP4 frame index
    from_seconds: float
    to_seconds: float

    @property
    def length(self) -> int:
        return self.end - self.first


@dataclass(frozen=True)
class Episode:
    episode_index: int
    list_index: int
    length: int
    segments: dict[str, Segment]
    data_file: str


def timestamp_to_frame(timestamp: float, fps: float) -> int:
    # Match the previous video-index audit: metadata's endpoint timestamps
    # are frame-grid boundaries, not frame-local indices.
    return int(round(float(timestamp) * fps))


def make_segment(root: Path, row: dict, cam: str, fps: float) -> Segment:
    prefix = f"videos/{cam}/"
    chunk = int(row[prefix + "chunk_index"])
    file_index = int(row[prefix + "file_index"])
    start_time = float(row[prefix + "from_timestamp"])
    end_time = float(row[prefix + "to_timestamp"])
    start = timestamp_to_frame(start_time, fps)
    end = timestamp_to_frame(end_time, fps)
    if start < 0 or end < start:
        raise ValueError(f"Invalid video interval for {cam}: [{start}, {end})")
    path = root / "videos" / cam / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
    return Segment(cam, path, start, end, start_time, end_time)


def load_episodes(root: Path, cams: list[str], fps: float) -> list[Episode]:
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No v3 metadata Parquet under {root / 'meta/episodes'}")
    columns = ["episode_index", "length", "data/chunk_index", "data/file_index"]
    for cam in cams:
        columns.extend(video_fields(cam))
    all_rows: list[dict] = []
    for path in files:
        frame = pd.read_parquet(path, columns=columns)
        all_rows.extend(frame.to_dict("records"))
    rows = sorted(all_rows, key=lambda x: int(x["episode_index"]))
    episodes: list[Episode] = []
    for i, row in enumerate(rows):
        ep_idx = int(row["episode_index"])
        if episodes and ep_idx == episodes[-1].episode_index:
            raise ValueError(f"Duplicate episode_index {ep_idx}")
        length = int(row["length"])
        if length < 1:
            raise ValueError(f"Bad episode length for episode {ep_idx}: {length}")
        segments = {cam: make_segment(root, row, cam, fps) for cam in cams}
        data_file = f"data/chunk-{int(row['data/chunk_index']):03d}/file-{int(row['data/file_index']):03d}.parquet"
        episodes.append(Episode(ep_idx, i, length, segments, data_file))
    return episodes


@lru_cache(maxsize=96)
def probe_video_size(path: str) -> tuple[int, int]:
    """Probe the actual decoded image size. No video is modified."""
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe was not found in PATH (needed for FFmpeg video playback)")
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=25,
        check=False,
    )
    if proc.returncode:
        raise RuntimeError(f"ffprobe failed on {path}: {proc.stderr.strip()[:200]}")
    streams = json.loads(proc.stdout).get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream: {path}")
    width = int(streams[0]["width"])
    height = int(streams[0]["height"])
    if not (0 < width <= 8192 and 0 < height <= 8192):
        raise ValueError(f"Unexpected video resolution {width}x{height}: {path}")
    return width, height


class SegmentReader:
    """AV1-compatible FFmpeg pipe decoder with strict per-episode frame fence.

    OpenCV's CAP_PROP_POS_FRAMES can return sentinel -9007199254740991
    for AV1; repeatedly calling VideoCapture.set() also isn't reliable.
    Instead, FFmpeg seeks ONCE to the episode's video timestamp and decodes
    continuously. When the episode interval ends, the pipe is closed without
    decoding even one frame from the next episode.

    Works with system FFmpeg/ffprobe; no cv2 HighGUI and no pip changes.
    """

    def __init__(self, segment: Segment, fps: float = 15.0):
        self.segment = segment
        self.fps = fps
        self.proc: subprocess.Popen | None = None
        self.cursor: int | None = None
        self.frame_bytes: int | None = None
        self.width = 0
        self.height = 0
        self.open_error: str | None = None
        self.failed = False

    def close(self) -> None:
        proc = self.proc
        self.proc = None
        self.cursor = None
        if proc is None:
            return
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=1.0)
        else:
            proc.wait()

    def _start(self, target: int) -> bool:
        self.close()
        self.failed = False
        if not self.segment.path.is_file():
            self.open_error = "VIDEO FILE MISSING"
            return False
        if shutil.which("ffmpeg") is None:
            self.open_error = "FFMPEG NOT FOUND"
            return False
        try:
            self.width, self.height = probe_video_size(str(self.segment.path))
            self.frame_bytes = self.width * self.height * 3
            # Accurate input-side seek for transcoding. Time is relative to
            # this MP4 file, NOT an episode-local timestamp.
            absolute_frame = self.segment.first + target
            absolute_time = absolute_frame / self.fps
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                "-hwaccel", "none", "-threads", "2",
                "-ss", f"{absolute_time:.9f}", "-i", str(self.segment.path),
                "-map", "0:v:0", "-an", "-sn", "-dn",
                "-vsync", "0", "-pix_fmt", "bgr24",
                "-f", "rawvideo", "pipe:1",
            ]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL,
                                         bufsize=self.frame_bytes * 2)
            self.cursor = target
            self.open_error = None
            return True
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            self.open_error = "FFMPEG OPEN ERROR"
            log(f"[ERROR] {self.segment.camera}: {exc}")
            self.close()
            return False

    def _read_exactly(self, size: int) -> bytes | None:
        if self.proc is None or self.proc.stdout is None:
            return None
        parts = bytearray()
        while len(parts) < size:
            chunk = self.proc.stdout.read(size - len(parts))
            if not chunk:
                return None
            parts.extend(chunk)
        return bytes(parts)

    def get(self, frame_idx: int) -> tuple[np.ndarray | None, str | None]:
        if frame_idx < 0 or frame_idx >= self.segment.length:
            return None, "MISSING TAIL FRAME"
        if self.failed:
            # A manual seek can recover from a failed read by reopening FFmpeg;
            # don't poison the entire EpisodeReader after one decode error.
            self.failed = False
        if self.proc is None or self.cursor != frame_idx:
            if not self._start(frame_idx):
                self.failed = True
                return None, self.open_error or "FFMPEG OPEN ERROR"
        assert self.frame_bytes is not None
        raw = self._read_exactly(self.frame_bytes)
        if raw is None:
            # A decode failure is distinct from a known episode tail gap.
            self.failed = True
            self.open_error = "FFMPEG DECODE ERROR"
            self.close()
            return None, self.open_error
        self.cursor = frame_idx + 1
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3).copy()
        return frame, None


class EpisodeReader:
    def __init__(self, ep: Episode, video_fps: float = 15.0):
        self.readers = {cam: SegmentReader(seg, fps=video_fps) for cam, seg in ep.segments.items()}

    def get(self, index: int) -> tuple[dict[str, np.ndarray | None], dict[str, str]]:
        frames: dict[str, np.ndarray | None] = {}
        errors: dict[str, str] = {}
        for cam, reader in self.readers.items():
            frame, error = reader.get(index)
            frames[cam] = frame
            if error is not None:
                errors[cam] = error
        return frames, errors

    def close(self) -> None:
        for reader in self.readers.values():
            reader.close()


def put_text(img: np.ndarray, text: str, pos: tuple[int, int], scale: float = .7,
             color: tuple[int, int, int] = (255, 255, 255)) -> None:
    # mimic the original viewer's outlined labels
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def resize_with_pad(image: np.ndarray | None, width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width, 3), np.uint8)
    if image is None or image.size == 0:
        return canvas
    h, w = image.shape[:2]
    s = min(width / w, height / h)
    dst_w, dst_h = max(1, round(w * s)), max(1, round(h * s))
    resized = cv2.resize(image, (dst_w, dst_h), interpolation=cv2.INTER_AREA if s <= 1 else cv2.INTER_LINEAR)
    x0 = (width - dst_w) // 2
    y0 = (height - dst_h) // 2
    canvas[y0:y0 + dst_h, x0:x0 + dst_w] = resized
    return canvas


def render_camera(frame: np.ndarray | None, name: str, error: str | None,
                  width: int, height: int) -> np.ndarray:
    panel = resize_with_pad(frame, width, height)
    put_text(panel, name, (14, 32), .7)
    if error:
        cv2.rectangle(panel, (6, height // 2 - 47), (width - 6, height // 2 + 50), (0, 0, 130), -1)
        put_text(panel, error, (24, height // 2 + 10), .72, (255, 255, 255))
    return panel


def render_info(width: int, height: int, ep: Episode, frame_idx: int, fps: float,
                rate: float, status: str, errors: dict[str, str]) -> np.ndarray:
    panel = np.zeros((height, width, 3), np.uint8)
    lines = [
        (f"episode_{ep.episode_index:06d}", 1.0),
        (f"list index: {ep.list_index}", .66),
        (f"status: {status}", .7),
        (f"frame_index: {frame_idx}  ({frame_idx + 1}/{ep.length})", .66),
        (f"time: {frame_idx/fps:.3f}s  ({fps:g} FPS)", .65),
        (f"rate: {rate:g}x", .65),
        (f"missing cameras: {len(errors)}", .62),
        ("h ls p n b g 35 i 10", .61),
        ("f 64  [ / ]  rate 0.5  r s q", .58),
    ]
    y = 39
    for line, scale in lines:
        color = (70, 170, 255) if "missing cameras:" in line and errors else (255, 255, 255)
        put_text(panel, line, (20, y), scale, color)
        y += 36 if scale > .8 else 33
    return panel


def compose_canvas(frames: dict[str, np.ndarray | None], errors: dict[str, str],
                   ep: Episode, frame_idx: int, fps: float, rate: float, status: str,
                   cams: list[str]) -> np.ndarray:
    high = next((c for c in cams if "cam_high" in c), cams[0])
    left = next((c for c in cams if "left_wrist" in c), None)
    right = next((c for c in cams if "right_wrist" in c), None)
    if len(cams) == 3 and left and right:
        top = np.concatenate([
            render_camera(frames.get(high), high.split(".")[-1], errors.get(high), 640, 360),
            render_info(640, 360, ep, frame_idx, fps, rate, status, errors),
        ], axis=1)
        bottom = np.concatenate([
            render_camera(frames.get(left), left.split(".")[-1], errors.get(left), 640, 480),
            render_camera(frames.get(right), right.split(".")[-1], errors.get(right), 640, 480),
        ], axis=1)
        return np.concatenate([top, bottom], axis=0)

    panels = [render_info(640, 360, ep, frame_idx, fps, rate, status, errors)]
    panels += [render_camera(frames.get(c), c.split(".")[-1], errors.get(c), 640, 360) for c in cams]
    while len(panels) % 2:
        panels.append(np.zeros((360, 640, 3), dtype=np.uint8))
    return np.concatenate([
        np.concatenate(panels[i:i + 2], axis=1)
        for i in range(0, len(panels), 2)
    ], axis=0)


def keyboard_command(key: int) -> str | None:
    if key < 0:
        return None
    k = chr(key & 0xFF)
    return k if k in "hpnbrsq" else ("prevframe" if k == "[" else "nextframe" if k == "]" else None)


def input_thread(commands: queue.Queue[str]) -> None:
    while True:
        try:
            cmd = input()
        except EOFError:
            return
        except KeyboardInterrupt:
            commands.put("q")
            return
        commands.put(cmd.strip())


def find_episode_position(eps: list[Episode], ep_idx: int) -> int | None:
    for i, ep in enumerate(eps):
        if ep.episode_index == ep_idx:
            return i
    return None


def list_episodes(eps: list[Episode], position: int, limit: int | None = None) -> None:
    if limit is None:
        lo, hi = 0, len(eps)
    else:
        lo = max(0, position - max(0, limit // 2))
        hi = min(len(eps), lo + limit)
    for i in range(lo, hi):
        ep = eps[i]
        missing = ", ".join(f"{cam.split('.')[-1]}:-{ep.length-seg.length}" for cam, seg in ep.segments.items() if seg.length < ep.length)
        log(f"{' =>' if i == position else '   '} [{i:04d}] episode_{ep.episode_index:06d} T={ep.length:4d} {missing}")


def handle_command(command: str, state: dict, eps: list[Episode], current: int) -> tuple[str, int] | None:
    try:
        args = shlex.split(command)
    except ValueError:
        args = command.split()
    if not args:
        return None
    op = args[0].lower()
    if op in ("h", "help"):
        log(HELP)
    elif op == "ls":
        lim = int(args[1]) if len(args) > 1 and args[1].isdigit() else None
        list_episodes(eps, current, lim)
    elif op == "p":
        if state["stopped"]:
            state["stopped"] = False
            state["paused"] = False
            return "replay", current
        state["paused"] = not state["paused"]
        log(f"[CMD] paused={state['paused']}")
    elif op == "n":
        return "goto", min(len(eps) - 1, current + 1)
    elif op == "b":
        return "goto", max(0, current - 1)
    elif op in ("g", "i"):
        if len(args) != 2:
            log(f"[ERR] usage: {op} <index>")
            return None
        try:
            target = int(args[1])
        except ValueError:
            log(f"[ERR] invalid index: {args[1]}")
            return None
        target_idx = find_episode_position(eps, target) if op == "g" else target
        if target_idx is None or not 0 <= target_idx < len(eps):
            log(f"[ERR] episode/list index not found: {target}")
            return None
        return "goto", target_idx
    elif op in ("f", "frame", "seek"):
        if len(args) != 2:
            log("[ERR] usage: f <zero-based episode-local frame index>")
            return None
        try:
            frame = int(args[1])
        except ValueError:
            log(f"[ERR] invalid frame index: {args[1]}")
            return None
        if not 0 <= frame < eps[current].length:
            log(f"[ERR] frame index out of range: 0..{eps[current].length-1}")
            return None
        state["paused"] = True
        state["stopped"] = False
        log(f"[SEEK] episode_{eps[current].episode_index:06d} frame_index={frame}; paused")
        return "seek", frame
    elif op in ("prevframe", "nextframe"):
        delta = -1 if op == "prevframe" else 1
        frame = min(eps[current].length - 1, max(0, state.get("frame", -1) + delta))
        state["paused"] = True
        state["stopped"] = False
        return "seek", frame
    elif op == "rate":
        if len(args) != 2:
            log("[ERR] usage: rate <positive float>")
            return None
        try:
            rate = float(args[1])
        except ValueError:
            log(f"[ERR] invalid rate: {args[1]}")
            return None
        if not math.isfinite(rate) or rate <= 0:
            log("[ERR] rate must be positive and finite")
            return None
        state["rate"] = rate
        log(f"[CMD] rate={rate:g}x, replay from start")
        return "replay", current
    elif op == "r":
        return "replay", current
    elif op == "s":
        return "stop", current
    elif op == "q":
        return "quit", current
    else:
        log(f"[ERR] unknown command: {command}")
    return None


def poll_commands(q: queue.Queue[str], state: dict, eps: list[Episode], cur: int) -> tuple[str, int] | None:
    while True:
        try:
            cmd = q.get_nowait()
        except queue.Empty:
            break
        action = handle_command(cmd, state, eps, cur)
        if action is not None:
            return action
    return None



class OpenCVViewer:
    """Existing native-window behavior; needs OpenCV built with HighGUI."""

    def __init__(self, args, commands: queue.Queue[str]):
        self.window_name = args.window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, round(1280 * args.window_scale), round(840 * args.window_scale))

    def show(self, frame: np.ndarray) -> None:
        cv2.imshow(self.window_name, frame)

    def poll(self, milliseconds: int) -> str | None:
        return keyboard_command(cv2.waitKey(milliseconds))

    def set_status(self, status: dict[str, Any]) -> None:
        pass  # Native OpenCV window shows the status in its image panel.

    def close(self) -> None:
        cv2.destroyWindow(self.window_name)


_BROWSER_HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RoboTwin LeRobot v3.0 video replay</title>
<style>
body{margin:0;padding:18px;background:#101827;color:#eff6ff;font:15px system-ui,sans-serif}
main{max-width:1450px;margin:auto}h2{margin:0 0 12px}
#viewer{display:block;width:100%;max-width:1280px;height:auto;background:#000;border:1px solid #344055}
.toolbar{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:12px 0}
.frame-panel{max-width:1240px;margin:14px 0;padding:14px 18px;border:1px solid #44526a;background:#172336;border-radius:10px}
.frame-row{display:flex;align-items:center;flex-wrap:wrap;gap:12px;margin:8px 0}
button,input{font:inherit;padding:7px 12px;border-radius:6px;border:1px solid #66758d;background:#1f2937;color:#fff}
button:hover{background:#364659;cursor:pointer}input[type=number]{width:92px}
#frameSlider{width:100%;accent-color:#5ca9ff;cursor:pointer;padding:0;border:0;height:24px}
.info{color:#bbc6d6;font-size:13px;line-height:1.7}
kbd{background:#374151;padding:2px 5px;border-radius:3px}
#frameStatus{font-weight:600;color:#a8d4ff}
</style></head><body><main>
<h2>RoboTwin · LeRobot v3.0 三路相机回放</h2>
<img id="viewer" src="/stream" alt="Waiting for episode frames..." />
<section class="frame-panel">
<div class="frame-row"><strong>选帧查看（自动暂停、不连续播放）</strong>
<span id="frameStatus">读取中...</span></div>
<input id="frameSlider" type="range" min="0" max="0" step="1" value="0" aria-label="当前 Episode 帧进度条">
<div class="frame-row">
<button id="prevFrame">◀ 上一帧 [</button>
<label>帧索引（从 0 开始） <input id="frameInput" type="number" min="0" value="0" step="1"></label>
<button id="seekFrame">查看该帧并暂停</button>
<button id="nextFrame">下一帧 ] ▶</button>
<span id="frameHint" class="info">拖动滑块后松开即选帧；选择不会自动播放</span>
</div></section>
<div class="toolbar">
<button data-cmd="b">◀ 上一条 b</button><button data-cmd="p">暂停/继续 p</button><button data-cmd="n">下一条 n ▶</button>
<button data-cmd="r">重播 r</button><button data-cmd="s">停止 s</button><button data-cmd="q">退出 q</button>
<label>Episode <input id="episode" type="number" min="0" placeholder="35"></label><button id="go">跳转 g</button>
<label>倍速 <input id="rate" type="number" step="0.25" min="0.1" value="1"></label><button id="setRate">应用 rate</button>
</div>
<div id="message" class="info">浏览器：拖动滑块或输入帧号即可暂停并查看对应图像；按 p 才继续播放。终端支持 f 64、[、]。服务器仅监听 127.0.0.1。</div>
</main><script>
const slider = document.getElementById('frameSlider');
const input = document.getElementById('frameInput');
const frameStatus = document.getElementById('frameStatus');
const message = document.getElementById('message');
let currentEpisode = null, currentFrame = 0, episodeLength = 1;
let editing = false, pendingFrame = null, pendingSince = 0;
async function send(cmd){
 try{const res=await fetch('/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command:cmd})});
 if(!res.ok)throw Error(await res.text());message.textContent='已发送命令：'+cmd;
 }catch(err){message.textContent='发送失败：'+err;}
}
function clampFrame(value){const n=Number(value);if(!Number.isInteger(n))return null;return Math.min(episodeLength-1,Math.max(0,n));}
function chooseFrame(value){
 const n=clampFrame(value);if(n===null){message.textContent='请输入整数帧索引';return;}
 input.value=n;slider.value=n;pendingFrame=n;pendingSince=Date.now();editing=false;
 frameStatus.textContent='正在定位第 '+n+' 帧（自动暂停）...';
 send('f '+n);
}
slider.addEventListener('input',()=>{editing=true;input.value=slider.value;frameStatus.textContent='待查看：frame_index='+slider.value;});
slider.addEventListener('change',()=>chooseFrame(slider.value));
input.addEventListener('focus',()=>{editing=true;});
input.addEventListener('keydown',event=>{if(event.key==='Enter'){event.preventDefault();chooseFrame(input.value);input.blur();}});
input.addEventListener('blur',()=>{if(pendingFrame===null)editing=false;});
document.getElementById('seekFrame').onclick=()=>chooseFrame(input.value);
document.getElementById('prevFrame').onclick=()=>chooseFrame(pendingFrame!==null?pendingFrame-1:currentFrame-1);
document.getElementById('nextFrame').onclick=()=>chooseFrame(pendingFrame!==null?pendingFrame+1:currentFrame+1);
document.querySelectorAll('[data-cmd]').forEach(b=>b.addEventListener('click',()=>send(b.dataset.cmd)));
document.getElementById('go').onclick=()=>{pendingFrame=null;editing=false;send('g '+document.getElementById('episode').value);};
document.getElementById('setRate').onclick=()=>send('rate '+document.getElementById('rate').value);
async function updateStatus(){
 try{
  const res=await fetch('/status',{cache:'no-store'});if(!res.ok)return;
  const v=await res.json();if(v.episode_index===undefined)return;
  const changed = v.episode_index!==currentEpisode;
  currentEpisode=v.episode_index;episodeLength=Math.max(1,v.length);
  currentFrame=v.frame_index;
  slider.max=String(episodeLength-1);input.max=String(episodeLength-1);
  if(changed){pendingFrame=null;editing=false;document.getElementById('episode').value=currentEpisode;}
  if(pendingFrame!==null&&(currentFrame===pendingFrame||Date.now()-pendingSince>20000)){
    pendingFrame=null;editing=false;
  }
  if(!editing&&pendingFrame===null){slider.value=String(Math.max(0,currentFrame));input.value=slider.value;}
  if(pendingFrame===null){frameStatus.textContent=`Episode ${currentEpisode} · frame_index=${currentFrame} / ${episodeLength-1} · ${v.status}`;}
 }catch(_err){}
}
setInterval(updateStatus,350);updateStatus();
document.addEventListener('keydown',event=>{
 if(['INPUT','TEXTAREA','SELECT'].includes(event.target.tagName))return;
 if(event.key==='ArrowLeft'||event.key==='['){event.preventDefault();chooseFrame(currentFrame-1);return;}
 if(event.key==='ArrowRight'||event.key===']'){event.preventDefault();chooseFrame(currentFrame+1);return;}
 const key=event.key.toLowerCase();if('hpnbrsq'.includes(key)&&key.length===1){event.preventDefault();send(key);}
});
</script></body></html>"""


class BrowserViewer:
    """Local-browser MJPEG display. Needs no cv2.namedWindow/Qt/GTK.

    Both the HTML and JPEG stream are bound to the local loopback address;
    HTTP commands share the same queue as the original terminal commands.
    """

    def __init__(self, args, commands: queue.Queue[str]):
        self.commands = commands
        self.condition = threading.Condition()
        self.latest_frame: bytes | None = None
        self.sequence = 0
        self.closed = False
        self.status: dict[str, Any] = {}
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *values) -> None:
                # Avoid logs on each JPEG update.
                return

            def do_GET(self) -> None:
                path = self.path.split('?', 1)[0]
                if path == '/':
                    body = _BROWSER_HTML.encode('utf-8')
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif path == '/status':
                    with parent.condition:
                        body = json.dumps(parent.status).encode('utf-8')
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json; charset=utf-8')
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif path == '/stream':
                    self.send_response(200)
                    self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                    self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
                    self.end_headers()
                    last_seq = -1
                    try:
                        while True:
                            with parent.condition:
                                parent.condition.wait_for(
                                    lambda: parent.closed or (parent.latest_frame is not None and parent.sequence != last_seq),
                                    timeout=5.0,
                                )
                                if parent.closed:
                                    break
                                if parent.latest_frame is None or parent.sequence == last_seq:
                                    continue
                                jpeg = parent.latest_frame
                                last_seq = parent.sequence
                            self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(jpeg)).encode() + b'\r\n\r\n')
                            self.wfile.write(jpeg + b'\r\n')
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass
                else:
                    self.send_error(404)

            def do_POST(self) -> None:
                if self.path != '/command':
                    self.send_error(404)
                    return
                if self.headers.get('Content-Type', '').split(';')[0].strip() != 'application/json':
                    self.send_error(415)
                    return
                try:
                    size = int(self.headers.get('Content-Length', '0'))
                    if not 0 < size <= 256:
                        raise ValueError('invalid command size')
                    data = json.loads(self.rfile.read(size).decode('utf-8'))
                    command = data['command']
                    if not isinstance(command, str) or not command.strip() or len(command) > 100:
                        raise ValueError('invalid command')
                    parent.commands.put(command.strip())
                    self.send_response(204)
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                except (ValueError, KeyError, UnicodeError, json.JSONDecodeError):
                    self.send_error(400, 'invalid command')

        self.server = ThreadingHTTPServer(('127.0.0.1', args.browser_port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}/'
        log(f'[INFO] Browser viewer: {self.url}')
        if not args.no_browser:
            try:
                if not webbrowser.open(self.url):
                    log('[INFO] Browser did not open automatically; open the URL above.')
            except Exception as exc:
                log(f'[INFO] Browser auto-open unavailable ({exc}); open URL above.')

    def show(self, frame: np.ndarray) -> None:
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 83])
        if not ok:
            raise RuntimeError('OpenCV JPEG encoding failed')
        with self.condition:
            self.latest_frame = buf.tobytes()
            self.sequence += 1
            self.condition.notify_all()

    def set_status(self, status: dict[str, Any]) -> None:
        with self.condition:
            self.status = dict(status)

    def poll(self, milliseconds: int) -> str | None:
        time.sleep(max(0, milliseconds) / 1000.0)
        return None  # Browser keystrokes are received through the HTTP command queue.

    def close(self) -> None:
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.server.shutdown()
        self.server.server_close()


def open_viewer(args, commands: queue.Queue[str]):
    if args.backend == 'browser':
        return BrowserViewer(args, commands)
    if args.backend == 'auto':
        gui_line = next((line for line in cv2.getBuildInformation().splitlines() if re.search(r'^\s*GUI\s*:', line)), '')
        if 'NONE' in gui_line.upper() or (os.name == 'posix' and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))):
            log('[INFO] HighGUI is unavailable; selecting browser backend.')
            return BrowserViewer(args, commands)
    try:
        return OpenCVViewer(args, commands)
    except cv2.error as exc:
        if args.backend == 'opencv':
            raise
        log(f'[WARN] OpenCV HighGUI unavailable ({exc}); falling back to browser backend.')
        return BrowserViewer(args, commands)


def display(args, eps: list[Episode], fps: float, cams: list[str], video_fps: float) -> None:
    current = find_episode_position(eps, args.episode_index)
    if current is None:
        log(f"[WARN] episode {args.episode_index} not found, using first episode")
        current = 0
    assert current is not None
    state: dict[str, Any] = {"paused": False, "stopped": False, "rate": args.rate, "frame": -1}
    commands: queue.Queue[str] = queue.Queue()
    threading.Thread(target=input_thread, args=(commands,), daemon=True).start()
    viewer = open_viewer(args, commands)
    log(f"[INFO] dataset episodes={len(eps)} fps={fps} cameras={cams}")
    log(HELP)
    first_episode = True
    try:
        while True:
            ep = eps[current]
            reader = EpisodeReader(ep, video_fps=video_fps)
            next_frame = 0   # next sequential frame to decode when playing
            state["frame"] = -1
            last_frames: dict[str, np.ndarray | None] = {cam: None for cam in cams}
            last_errors: dict[str, str] = {}
            next_action: tuple[str, int] | None = None
            last_status: str | None = None

            def publish(status: str) -> None:
                nonlocal last_status
                frame_index = max(0, state["frame"])
                viewer.show(compose_canvas(last_frames, last_errors, ep, frame_index,
                                           fps, state["rate"], status, cams))
                viewer.set_status({"episode_index": ep.episode_index,
                                   "list_index": ep.list_index,
                                   "length": ep.length,
                                   "frame_index": frame_index,
                                   "status": status,
                                   "paused": state["paused"],
                                   "rate": state["rate"]})
                last_status = status

            try:
                log(f"[PLAY] episode_{ep.episode_index:06d} frames={ep.length} rate={state['rate']:g}x")
                for cam, seg in ep.segments.items():
                    if seg.length != ep.length:
                        log(f"  [VIDEO LENGTH] {cam}: {seg.length}/{ep.length} (missing tail: {max(0, ep.length-seg.length)})")
                if first_episode and (args.start_paused or args.frame is not None):
                    target = args.frame if args.frame is not None else 0
                    if not 0 <= target < ep.length:
                        raise ValueError(f"--frame must be between 0 and {ep.length-1} for episode {ep.episode_index}")
                    last_frames, last_errors = reader.get(target)
                    state.update(paused=True, stopped=False, frame=target)
                    next_frame = target + 1
                    publish("PAUSED")
                first_episode = False

                while True:
                    next_action = poll_commands(commands, state, eps, current)
                    if next_action:
                        event, target = next_action
                        if event == "seek":
                            # Decode exactly the requested episode-local frame.
                            # The user explicitly requested no autoplay on seek.
                            last_frames, last_errors = reader.get(target)
                            next_frame = target + 1
                            state.update(frame=target, paused=True, stopped=False)
                            publish("PAUSED")
                            next_action = None
                            continue
                        break

                    if state["stopped"] or state["paused"] or next_frame >= ep.length:
                        status = ("STOPPED" if state["stopped"] else
                                  "PAUSED" if state["paused"] else "ENDED")
                        if status == "PAUSED" and state["frame"] < 0:
                            last_frames, last_errors = reader.get(0)
                            state["frame"] = 0
                            next_frame = 1
                        if status != last_status:
                            publish(status)
                        cmd = viewer.poll(30)
                        if cmd:
                            next_action = handle_command(cmd, state, eps, current)
                            if next_action:
                                if next_action[0] == "seek":
                                    # OpenCV backend shares the same queue/event logic.
                                    commands.put(f"f {next_action[1]}")
                                    next_action = None
                                else:
                                    break
                        if next_frame >= ep.length and not state["stopped"] and not state["paused"]:
                            next_action = ("ended", current)
                            break
                        continue

                    begin = time.monotonic()
                    last_frames, last_errors = reader.get(next_frame)
                    state["frame"] = next_frame
                    next_frame += 1
                    publish("PLAYING")
                    cmd = viewer.poll(1)
                    if cmd:
                        next_action = handle_command(cmd, state, eps, current)
                        if next_action:
                            if next_action[0] == "seek":
                                commands.put(f"f {next_action[1]}")
                                next_action = None
                            else:
                                break
                    remaining = max(0.0, (1.0 / (fps * state["rate"])) - (time.monotonic() - begin))
                    if remaining > 0:
                        time.sleep(remaining)
            finally:
                reader.close()

            if next_action is None:
                break
            event, target = next_action
            if event == "quit":
                break
            if event == "stop":
                state.update(stopped=True, paused=False)
                continue
            if event in ("goto", "replay"):
                current = target
                state.update(stopped=False, paused=False)
                continue
            if event == "ended":
                if current + 1 < len(eps):
                    current += 1
                    state.update(stopped=False, paused=False)
                elif args.loop:
                    current = 0
                    state.update(stopped=False, paused=False)
                else:
                    log("[INFO] reached last episode (stopped)")
                    state.update(stopped=True, paused=False)
    finally:
        viewer.close()
        log("[DONE] replay exited")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RoboTwin LeRobot v3.0 video keyboard replay (3 cameras)")
    parser.add_argument("--root", type=Path, required=True, help="LeRobot v3.0 dataset root")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame", type=int, default=None,
                        help="Start at this zero-based episode frame index and remain paused")
    parser.add_argument("--start-paused", action="store_true",
                        help="Show first frame but do not autoplay until pressing p")
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=None, help="Override playback FPS only; video indexing always uses metadata FPS")
    parser.add_argument("--loop", action="store_true", help="Repeat all episodes after last")
    parser.add_argument("--window-name", default="RoboTwin v3.0 LeRobot video replay")
    parser.add_argument("--window-scale", type=float, default=1.0)
    parser.add_argument("--cameras", nargs="*", default=None)
    parser.add_argument("--check-only", action="store_true", help="Print selected episode camera intervals, no GUI and no video decoding")
    parser.add_argument("--decode-check", action="store_true", help="Decode first/middle/last valid frame per camera using FFmpeg without opening a window")
    parser.add_argument("--backend", choices=("auto", "browser", "opencv"), default="auto", help="Viewer backend; browser avoids OpenCV HighGUI dependencies")
    parser.add_argument("--browser-port", type=int, default=0, help="Localhost browser server port, 0=automatic free port")
    parser.add_argument("--no-browser", action="store_true", help="Print the local URL without automatically opening the browser")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not math.isfinite(args.rate) or args.rate <= 0:
        raise ValueError("--rate must be a positive finite number")
    if not math.isfinite(args.window_scale) or args.window_scale <= 0:
        raise ValueError("--window-scale must be a positive finite number")
    root = args.root.expanduser().resolve()
    info = read_info(root)
    dataset_fps = float(info.get("fps", 15))
    fps = float(args.fps) if args.fps is not None else dataset_fps
    if not math.isfinite(dataset_fps) or dataset_fps <= 0:
        raise ValueError(f"Bad dataset metadata FPS: {dataset_fps}")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Bad playback FPS: {fps}")
    cams = camera_keys(info, args.cameras)
    episodes = load_episodes(root, cams, dataset_fps)
    if not episodes:
        raise RuntimeError("No episodes in metadata")
    if args.check_only:
        pos = find_episode_position(episodes, args.episode_index)
        if pos is None:
            raise ValueError(f"Episode {args.episode_index} not found")
        ep = episodes[pos]
        log(f"[CHECK] root={root}\n[CHECK] episode={ep.episode_index} T={ep.length} dataset_fps={dataset_fps} playback_fps={fps}")
        for cam, seg in ep.segments.items():
            log(f"  {cam}: file={seg.path} frames=[{seg.first},{seg.end}) valid={seg.length}/{ep.length} missing_tail={max(0, ep.length-seg.length)}")
        return
    if args.decode_check:
        pos = find_episode_position(episodes, args.episode_index)
        if pos is None:
            raise ValueError(f"Episode {args.episode_index} not found")
        ep = episodes[pos]
        failures = 0
        log(f"[DECODE CHECK] episode={ep.episode_index}, state_frames={ep.length}, decoder=ffmpeg")
        for cam, seg in ep.segments.items():
            if seg.length < 1:
                log(f"  [ERR] {cam}: no video frames in episode")
                failures += 1
                continue
            positions = sorted(set((0, seg.length // 2, seg.length - 1)))
            reader = SegmentReader(seg, fps=dataset_fps)
            try:
                for idx in positions:
                    frame, err = reader.get(idx)
                    if err:
                        failures += 1
                        log(f"  [ERR] {cam} frame={idx}/{ep.length}: {err}")
                    else:
                        log(f"  [OK]  {cam} frame={idx}/{ep.length} shape={frame.shape}")
                if seg.length < ep.length:
                    frame, err = reader.get(seg.length)
                    if frame is not None or err != "MISSING TAIL FRAME":
                        failures += 1
                        log(f"  [ERR] {cam}: missing tail-frame guard failed")
                    else:
                        log(f"  [OK]  {cam}: missing tail frame guarded ({ep.length-seg.length} frames)")
            finally:
                reader.close()
        if failures:
            raise SystemExit(f"[DECODE CHECK] FAILED: {failures} errors")
        log("[DECODE CHECK] PASSED")
        return
    display(args, episodes, fps, cams, dataset_fps)


if __name__ == "__main__":
    main()
