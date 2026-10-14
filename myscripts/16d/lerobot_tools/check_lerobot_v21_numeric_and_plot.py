#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LeRobot v2.1 numeric plotting tool only.

读取 LeRobot parquet
绘制 state/action overlay
绘制 action rows
绘制 state rows
绘制 frame-to-frame jump rows
生成静态 index.html
生成交互式 interactive_episode_groups.html
可选 xdg-open 自动打开网页

Example:
增加腰部：
（1）绘制全部图，较慢
cd /home/lixiang/PROJECT/LingBotVLA2.0

python /home/lixiang/PROJECT/LingBotVLA2.0/myscripts/16d/lerobot_tools/check_lerobot_v21_numeric_and_plot.py \
  --root /home/lixiang/PROJECT/LingBotVLA2.0/datasets/lerobot/lixiang/put_pen_and_tape_into_pen_holder \
  --out-dir /home/lixiang/PROJECT/LingBotVLA2.0/reports/lerobot_quality_reports/put_pen_and_tape_into_pen_holder_numeric_interactive \
  --plot-mode all \
  --interactive-groups \
  --interactive-default-group-size 1 \
  --interactive-group-sizes 1,5,10,20 \
  --open-html

（2）绘制网页交互图，较快
cd /home/lixiang/PROJECT/LingBotVLA2.0

python /home/lixiang/PROJECT/LingBotVLA2.0/myscripts/16d/lerobot_tools/check_lerobot_v21_numeric_and_plot.py \
  --root /home/lixiang/PROJECT/LingBotVLA2.0/datasets/lerobot/lixiang/put_pen_and_tape_into_pen_holder \
  --out-dir /home/lixiang/PROJECT/LingBotVLA2.0/reports/lerobot_quality_reports/put_pen_and_tape_into_pen_holder_numeric_interactive \
  --plot-mode all \
  --interactive-only \
  --interactive-default-group-size 1 \
  --interactive-group-sizes 1,5,10,20 \
  --open-html

看图：
xdg-open /home/lixiang/PROJECT/LingBotVLA2.0/reports/lerobot_quality_reports/hang_wire_stripper_on_wall_hook_16d_numeric_interactive/plots/interactive_episode_groups.html

  
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_DIMS = 16
# R1Pro 16D: left_arm7, left_gripper, right_arm7, right_gripper.
DEFAULT_ARM_DIMS = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]
DEFAULT_GRIPPER_DIMS = [7, 15]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot LeRobot v2.1 numeric state/action curves only.")
    p.add_argument("--root", required=True, type=Path, help="LeRobot dataset root")
    p.add_argument("--out-dir", required=True, type=Path, help="Output report directory")
    p.add_argument("--start", type=int, default=0, help="Start episode index, inclusive")
    p.add_argument("--end", type=int, default=None, help="End episode index, exclusive")
    p.add_argument("--episodes", default="", help="Comma-separated episode indices, e.g. 0,1,61. Overrides start/end when set.")
    p.add_argument("--plot-mode", choices=["none", "all", "range", "episodes"], default="all")
    p.add_argument("--plot-format", default="png", help="png or pdf, or comma list like png,pdf")
    p.add_argument("--dpi", type=int, default=140)
    p.add_argument("--expected-dim", type=int, default=DEFAULT_DIMS, help="Expected state/action dim; fixed to 16 for this toolchain.")
    p.add_argument("--arm-dims", default=",".join(map(str, DEFAULT_ARM_DIMS)))
    p.add_argument("--gripper-dims", default=",".join(map(str, DEFAULT_GRIPPER_DIMS)))
    p.add_argument("--jump-reference-lines", action="store_true", help="Draw optional reference lines on jump plots")
    p.add_argument("--arm-jump-reference", type=float, default=0.60)
    p.add_argument("--gripper-jump-reference", type=float, default=100.0)
    p.add_argument("--interactive-groups", action="store_true", help="Write plots/interactive_episode_groups.html")
    p.add_argument("--interactive-group-sizes", default="1,5,10,20")
    p.add_argument("--interactive-default-group-size", type=int, default=1)
    p.add_argument(
        "--interactive-max-points", type=int, default=2000,
        help="Maximum points retained per episode in interactive HTML; 0 keeps all points.",
    )
    p.add_argument("--open-html", action="store_true", help="Run xdg-open for generated HTML pages")
    p.add_argument(
        "--interactive-only",
        action="store_true",
        help=(
            "Only generate plots/interactive_episode_groups.html; "
            "skip static episode PNG/PDF plots and plots/index.html."
        ),
    )
    return p.parse_args()


def parse_int_list(s: str) -> list[int]:
    out: list[int] = []
    for x in s.split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    return out


def parse_formats(s: str) -> list[str]:
    fmts = [x.strip().lower() for x in s.split(",") if x.strip()]
    return fmts or ["png"]


def parse_group_sizes(s: str) -> list[int]:
    sizes = sorted(set(parse_int_list(s)))
    sizes = [x for x in sizes if x > 0]
    if not sizes:
        sizes = [1, 5, 10, 20]
    if 1 not in sizes:
        sizes.insert(0, 1)
    return sizes


def load_info(root: Path) -> dict:
    p = root / "meta" / "info.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    return json.loads(p.read_text())


def infer_feature_dim(info: dict, key: str) -> int | None:
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


def get_feature_names(info: dict, key: str, dim: int) -> str:
    try:
        names = info.get("features", {}).get(key, {}).get("names")
        if names and dim < len(names):
            return str(names[dim])
    except Exception:
        pass
    return f"{key}[{dim}]"


def to_array(series: pd.Series, expected_dim: int) -> np.ndarray:
    values = series.to_list()
    try:
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim == 1:
            arr = np.stack(values).astype(np.float32)
    except Exception:
        arr = np.stack([np.asarray(v, dtype=np.float32) for v in values]).astype(np.float32)
    if arr.ndim != 2 or arr.shape[-1] != expected_dim:
        raise ValueError(f"bad shape {arr.shape}, expected [T,{expected_dim}]")
    return arr


def collect_episode_groups(root: Path) -> Iterable[tuple[int, Path, pd.DataFrame]]:
    parquets = sorted((root / "data").glob("**/*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")
    for p in parquets:
        df = pd.read_parquet(p)
        if "episode_index" in df.columns:
            for ep_idx, g in df.groupby("episode_index", sort=True):
                yield int(ep_idx), p, g.copy()
        else:
            ep_idx = -1
            for token in p.stem.replace("-", "_").split("_"):
                if token.isdigit():
                    ep_idx = int(token)
            yield ep_idx, p, df.copy()


def sort_episode_df(df: pd.DataFrame) -> pd.DataFrame:
    if "frame_index" in df.columns:
        return df.sort_values("frame_index").reset_index(drop=True)
    return df.reset_index(drop=True)


def get_x_axis(df: pd.DataFrame, fps: float) -> tuple[np.ndarray, str]:
    if "timestamp" in df.columns:
        ts = np.asarray(df["timestamp"].to_list(), dtype=np.float64)
        if len(ts) > 0:
            return ts, "timestamp (s)"
    return np.arange(len(df), dtype=np.float64) / max(fps, 1e-9), "time from frame_index (s)"


def draw_single_series_rows(x: np.ndarray, arr: np.ndarray, names: list[str], title: str, out_path: Path, dpi: int) -> None:
    n_dims = arr.shape[1]
    fig_h = max(16, n_dims * 1.25)
    fig, axes = plt.subplots(n_dims, 1, figsize=(18, fig_h), sharex=True)
    if n_dims == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(x, arr[:, i], linewidth=1.2)
        ax.set_ylabel(f"{i}\n{names[i]}", rotation=0, labelpad=60, fontsize=8, va="center")
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.tick_params(axis="y", labelsize=8)
    axes[0].set_title(title)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def draw_overlay_state_action_rows(x: np.ndarray, state: np.ndarray, action: np.ndarray, state_names: list[str], action_names: list[str], title: str, out_path: Path, dpi: int) -> None:
    n_dims = state.shape[1]
    fig_h = max(16, n_dims * 1.35)
    fig, axes = plt.subplots(n_dims, 1, figsize=(18, fig_h), sharex=True)
    if n_dims == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(x, state[:, i], linewidth=1.1, label="state")
        ax.plot(x, action[:, i], linewidth=1.1, linestyle="--", label="action")
        label = f"{i}\nS:{state_names[i]}\nA:{action_names[i]}"
        ax.set_ylabel(label, rotation=0, labelpad=80, fontsize=7, va="center")
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.tick_params(axis="y", labelsize=8)
    axes[0].legend(loc="upper right")
    axes[0].set_title(title)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def draw_jump_rows(x_mid: np.ndarray, d_state: np.ndarray, d_action: np.ndarray, state_names: list[str], action_names: list[str], title: str, out_path: Path, dpi: int, *, draw_reference: bool, arm_reference: float, gripper_reference: float, arm_dims: list[int], gripper_dims: list[int]) -> None:
    n_dims = d_state.shape[1]
    fig_h = max(16, n_dims * 1.35)
    fig, axes = plt.subplots(n_dims, 1, figsize=(18, fig_h), sharex=True)
    if n_dims == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(x_mid, d_state[:, i], linewidth=1.1, label="abs diff state")
        ax.plot(x_mid, d_action[:, i], linewidth=1.1, linestyle="--", label="abs diff action")
        if draw_reference:
            if i in arm_dims:
                ax.axhline(arm_reference, linestyle="--", linewidth=0.8)
            elif i in gripper_dims:
                ax.axhline(gripper_reference, linestyle="--", linewidth=0.8)
        label = f"{i}\nS:{state_names[i]}\nA:{action_names[i]}"
        ax.set_ylabel(label, rotation=0, labelpad=80, fontsize=7, va="center")
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.tick_params(axis="y", labelsize=8)
    axes[0].legend(loc="upper right")
    axes[0].set_title(title)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def draw_episode_plots(ep_idx: int, df: pd.DataFrame, info: dict, plots_dir: Path, fps: float, expected_dim: int, dpi: int, formats: list[str], *, jump_reference_lines: bool, arm_reference: float, gripper_reference: float, arm_dims: list[int], gripper_dims: list[int]) -> list[Path]:
    df = sort_episode_df(df)
    for c in ["observation.state", "action"]:
        if c not in df.columns:
            raise ValueError(f"episode_{ep_idx:06d} missing column: {c}")
    state = to_array(df["observation.state"], expected_dim=expected_dim)
    action = to_array(df["action"], expected_dim=expected_dim)
    x, _ = get_x_axis(df, fps)
    state_names = [get_feature_names(info, "observation.state", i) for i in range(expected_dim)]
    action_names = [get_feature_names(info, "action", i) for i in range(expected_dim)]
    ep_dir = plots_dir / f"episode_{ep_idx:06d}"
    ep_dir.mkdir(parents=True, exist_ok=True)
    d_state = np.abs(np.diff(state, axis=0))
    d_action = np.abs(np.diff(action, axis=0))
    x_mid = (x[:-1] + x[1:]) / 2.0 if len(x) >= 2 else np.arange(len(d_state), dtype=np.float64) / max(fps, 1e-9)
    outputs: list[Path] = []
    for fmt in formats:
        overlay = ep_dir / f"episode_{ep_idx:06d}_state_action_overlay.{fmt}"
        action_rows = ep_dir / f"episode_{ep_idx:06d}_action_rows.{fmt}"
        state_rows = ep_dir / f"episode_{ep_idx:06d}_state_rows.{fmt}"
        jump_rows = ep_dir / f"episode_{ep_idx:06d}_jump_rows.{fmt}"
        draw_overlay_state_action_rows(x, state, action, state_names, action_names, f"episode_{ep_idx:06d} state/action overlay", overlay, dpi)
        draw_single_series_rows(x, action, action_names, f"episode_{ep_idx:06d} action rows", action_rows, dpi)
        draw_single_series_rows(x, state, state_names, f"episode_{ep_idx:06d} state rows", state_rows, dpi)
        draw_jump_rows(x_mid, d_state, d_action, state_names, action_names, f"episode_{ep_idx:06d} abs frame-to-frame jump rows", jump_rows, dpi, draw_reference=jump_reference_lines, arm_reference=arm_reference, gripper_reference=gripper_reference, arm_dims=arm_dims, gripper_dims=gripper_dims)
        outputs.extend([overlay, action_rows, state_rows, jump_rows])
    return outputs


def write_html_index(out_dir: Path, plotted_eps: list[int], formats: list[str]) -> Path:
    html = out_dir / "plots" / "index.html"
    html.parent.mkdir(parents=True, exist_ok=True)
    fmt = formats[0]
    lines = ["<!doctype html><html><head><meta charset='utf-8'><title>LeRobot numeric plots</title></head><body>",
             "<h1>LeRobot numeric plots</h1>",
             "<p>This page only visualizes existing LeRobot state/action curves. Missing-frame audit is handled by check_lerobot_v21_missing_audit.py.</p>",
             "<ul>"]
    for ep in plotted_eps:
        ep_name = f"episode_{ep:06d}"
        lines.append(f"<li><h2>{ep_name}</h2>")
        for suffix in ["state_action_overlay", "action_rows", "state_rows", "jump_rows"]:
            rel = f"{ep_name}/{ep_name}_{suffix}.{fmt}"
            lines.append(f"<p><a href='{rel}'>{suffix}</a></p>")
            lines.append(f"<img src='{rel}' style='width:1200px; max-width:100%; border:1px solid #ccc;'>")
        lines.append("</li>")
    lines.append("</ul></body></html>")
    html.write_text("\n".join(lines), encoding="utf-8")
    return html


def _round_float_list(arr: np.ndarray, ndigits: int = 6) -> list[float]:
    return np.round(np.asarray(arr, dtype=np.float64), ndigits).tolist()


def downsample_indices(length: int, max_points: int) -> np.ndarray:
    if length <= 0:
        return np.asarray([], dtype=np.int64)
    if max_points <= 0 or length <= max_points:
        return np.arange(length, dtype=np.int64)
    if max_points == 1:
        return np.asarray([0], dtype=np.int64)
    return np.unique(np.linspace(0, length - 1, max_points, dtype=np.int64))


def build_interactive_episode_payload(ep_idx: int, df: pd.DataFrame, fps: float, expected_dim: int, max_points: int) -> dict:
    df = sort_episode_df(df)
    state = to_array(df["observation.state"], expected_dim=expected_dim)
    action = to_array(df["action"], expected_dim=expected_dim)
    x, x_label = get_x_axis(df, fps)
    keep = downsample_indices(len(x), max_points)
    x = x[keep]
    state = state[keep]
    action = action[keep]
    return {
        "episode_index": int(ep_idx),
        "x": _round_float_list(x),
        "x_label": x_label,
        "state": [_round_float_list(state[:, i]) for i in range(expected_dim)],
        "action": [_round_float_list(action[:, i]) for i in range(expected_dim)],
    }


def write_interactive_group_html(out_dir: Path, payload_path: Path, info: dict, expected_dim: int, group_sizes: list[int], default_group_size: int) -> Path:
    html = out_dir / "plots" / "interactive_episode_groups.html"
    html.parent.mkdir(parents=True, exist_ok=True)
    state_names = [get_feature_names(info, "observation.state", i) for i in range(expected_dim)]
    action_names = [get_feature_names(info, "action", i) for i in range(expected_dim)]
    if default_group_size not in group_sizes:
        default_group_size = group_sizes[0]
    state_names_json = json.dumps(state_names, ensure_ascii=False)
    action_names_json = json.dumps(action_names, ensure_ascii=False)
    group_sizes_json = json.dumps(group_sizes)
    lines = ["<!doctype html>", "<html><head><meta charset='utf-8'>", "<title>Interactive Episode Groups</title>",
             "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>",
             "<style>body{font-family:Arial,sans-serif;margin:18px;background:#fafafa;color:#111}.toolbar{position:sticky;top:0;background:#fff;border:1px solid #ddd;padding:10px;z-index:10}button,select{margin:4px;padding:6px 9px}#plot{width:100%;height:760px;background:#fff;border:1px solid #ddd}.hint{font-size:13px;color:#444;line-height:1.45}</style></head><body>",
             "<h1>Interactive Episode Groups</h1>",
             "<div class='toolbar'><span>Group size: </span><span id='groupButtons'></span><button onclick='prevGroup()'>Prev group</button><button onclick='nextGroup()'>Next group</button><span> Dimension: </span><select id='dimSelect' onchange='setDim(parseInt(this.value))'></select><span> Mode: </span><select id='modeSelect' onchange='setMode(this.value)'><option value='overlay'>state + action</option><option value='state'>state</option><option value='action'>action</option></select><div id='status' class='hint'></div><div class='hint'>Keyboard: 1=single, 5=5 episodes, 0=10 episodes, 2=20 episodes, arrows switch group/dim, s/a/o switch mode.</div></div>",
             "<div id='plot'></div>", "<script>"]
    with html.open("w", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
        stream.write("\nconst EPISODES = [\n")
        first = True
        with payload_path.open("r", encoding="utf-8") as payload_stream:
            for line in payload_stream:
                line = line.strip()
                if not line:
                    continue
                if not first:
                    stream.write(",\n")
                stream.write(line)
                first = False
        stream.write("\n];\n")
        stream.write(f"const STATE_NAMES = {state_names_json};\n")
        stream.write(f"const ACTION_NAMES = {action_names_json};\n")
        stream.write(f"const GROUP_SIZES = {group_sizes_json};\n")
        stream.write(f"let groupSize = {int(default_group_size)};\n")
        stream.write("let groupStart = 0; let dim = 0; let mode = 'overlay';\n")
        stream.write(r'''
function initControls(){const dimSelect=document.getElementById('dimSelect');for(let i=0;i<STATE_NAMES.length;i++){const opt=document.createElement('option');opt.value=String(i);opt.textContent=`${i}: state=${STATE_NAMES[i]} / action=${ACTION_NAMES[i]}`;dimSelect.appendChild(opt);}const groupButtons=document.getElementById('groupButtons');GROUP_SIZES.forEach(gs=>{const b=document.createElement('button');b.textContent=String(gs);b.onclick=()=>setGroupSize(gs);groupButtons.appendChild(b);});}
function clampStart(){if(EPISODES.length===0){groupStart=0;return;}const maxStart=Math.max(0,EPISODES.length-groupSize);if(groupStart>maxStart)groupStart=maxStart;if(groupStart<0)groupStart=0;}
function currentGroup(){clampStart();return EPISODES.slice(groupStart,groupStart+groupSize);}
function setGroupSize(gs){groupSize=gs;groupStart=Math.floor(groupStart/groupSize)*groupSize;render();}
function setDim(d){dim=Math.max(0,Math.min(STATE_NAMES.length-1,d));document.getElementById('dimSelect').value=String(dim);render();}
function setMode(m){mode=m;document.getElementById('modeSelect').value=mode;render();}
function nextGroup(){groupStart+=groupSize;if(groupStart>=EPISODES.length)groupStart=0;render();}
function prevGroup(){groupStart-=groupSize;if(groupStart<0)groupStart=Math.max(0,Math.floor((EPISODES.length-1)/groupSize)*groupSize);render();}
function makeTrace(ep,kind){const y=ep[kind][dim];return {x:ep.x,y:y,mode:'lines',type:'scatter',name:`episode_${String(ep.episode_index).padStart(6,'0')} ${kind}`,line:{width:1.5,dash:kind==='action'?'dash':'solid'},customdata:ep.x.map((xv,idx)=>[ep.episode_index,idx,kind,dim,y[idx]]),hovertemplate:'episode=%{customdata[0]}<br>frame=%{customdata[1]}<br>kind=%{customdata[2]}<br>dim=%{customdata[3]}<br>time=%{x:.4f}s<br>value=%{y:.6f}<extra></extra>'};}
function render(){const group=currentGroup();const traces=[];group.forEach(ep=>{if(mode==='state'||mode==='overlay')traces.push(makeTrace(ep,'state'));if(mode==='action'||mode==='overlay')traces.push(makeTrace(ep,'action'));});const epNames=group.map(ep=>`episode_${String(ep.episode_index).padStart(6,'0')}`).join(', ');document.getElementById('status').textContent=`groupSize=${groupSize}, start=${groupStart}, dim=${dim}, mode=${mode}, episodes=${epNames}`;const layout={title:`Dimension ${dim}: state=${STATE_NAMES[dim]} / action=${ACTION_NAMES[dim]}`,hovermode:'closest',xaxis:{title:'time (s)',zeroline:false},yaxis:{title:'value',zeroline:false},legend:{orientation:'h'},margin:{l:70,r:30,t:70,b:80}};Plotly.react('plot',traces,layout,{responsive:true,scrollZoom:true,displaylogo:false});}
document.addEventListener('keydown',(e)=>{if(e.target&&['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName))return;if(e.key==='1')setGroupSize(1);else if(e.key==='5')setGroupSize(5);else if(e.key==='0')setGroupSize(10);else if(e.key==='2')setGroupSize(20);else if(e.key==='ArrowRight')nextGroup();else if(e.key==='ArrowLeft')prevGroup();else if(e.key==='ArrowDown')setDim(dim+1);else if(e.key==='ArrowUp')setDim(dim-1);else if(e.key.toLowerCase()==='s')setMode('state');else if(e.key.toLowerCase()==='a')setMode('action');else if(e.key.toLowerCase()==='o')setMode('overlay');});
initControls();render();
''')
        stream.write("\n</script></body></html>\n")
    return html


def write_manifest_csv(out_dir: Path, rows: list[dict]) -> Path:
    path = out_dir / "plot_manifest.csv"
    with path.open("w", newline="") as f:
        fieldnames = ["episode_index", "frames", "parquet", "plot_dir"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    return path


def xdg_open(path: Path) -> None:
    try:
        subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        print(f"[WARN] failed to open {path}: {exc}")


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    formats = parse_formats(args.plot_format)
    group_sizes = parse_group_sizes(args.interactive_group_sizes)
    episode_filter = set(parse_int_list(args.episodes)) if args.episodes.strip() else None
    arm_dims = parse_int_list(args.arm_dims)
    gripper_dims = parse_int_list(args.gripper_dims)

    info = load_info(root)
    fps = float(info.get("fps", 15))
    if args.expected_dim is None:
        state_dim = infer_feature_dim(info, "observation.state")
        action_dim = infer_feature_dim(info, "action")
        if state_dim is None or action_dim is None:
            raise ValueError("Cannot infer expected_dim from meta/info.json; pass --expected-dim explicitly.")
        if state_dim != action_dim:
            raise ValueError(f"state/action dim mismatch in meta/info.json: state={state_dim}, action={action_dim}")
        args.expected_dim = state_dim
    print(f"[INFO] expected_dim: {args.expected_dim}")
    print(f"[INFO] root: {root}")
    print(f"[INFO] codebase_version: {info.get('codebase_version')}")
    print(f"[INFO] fps: {fps}")
    print(f"[INFO] total_episodes: {info.get('total_episodes')}")
    print(f"[INFO] total_frames: {info.get('total_frames')}")
    print(f"[INFO] plot_mode: {args.plot_mode}")

    plotted_eps: list[int] = []
    manifest_rows: list[dict] = []
    interactive_enabled = args.interactive_groups or args.interactive_only
    payload_path = plots_dir / ".interactive_episode_payload.jsonl"
    payload_stream = payload_path.open("w", encoding="utf-8") if interactive_enabled else None
    try:
        for ep_idx, parquet_path, df in collect_episode_groups(root):
            if args.plot_mode == "none":
                continue
            if episode_filter is None and args.end is not None and ep_idx >= args.end:
                break
            if episode_filter is not None:
                if ep_idx not in episode_filter:
                    continue
            else:
                if ep_idx < args.start:
                    continue
                if args.end is not None and ep_idx >= args.end:
                    continue
                if args.plot_mode == "episodes" and episode_filter is None:
                    continue
            ep_idx = int(ep_idx)
            plotted_eps.append(ep_idx)
            manifest_rows.append({"episode_index": ep_idx, "frames": len(df), "parquet": str(parquet_path), "plot_dir": str(plots_dir / f"episode_{ep_idx:06d}")})
            if not args.interactive_only:
                print(f"[PLOT] episode_{ep_idx:06d}")
                draw_episode_plots(
                    ep_idx, df, info, plots_dir, fps, args.expected_dim,
                    args.dpi, formats,
                    jump_reference_lines=args.jump_reference_lines,
                    arm_reference=args.arm_jump_reference,
                    gripper_reference=args.gripper_jump_reference,
                    arm_dims=arm_dims, gripper_dims=gripper_dims,
                )
            if payload_stream is not None:
                payload = build_interactive_episode_payload(
                    ep_idx, df, fps, args.expected_dim, args.interactive_max_points
                )
                json.dump(payload, payload_stream, ensure_ascii=False, separators=(",", ":"))
                payload_stream.write("\n")
                payload_stream.flush()
            del df
            gc.collect()
    finally:
        if payload_stream is not None:
            payload_stream.close()

    plotted_eps.sort()
    html = write_html_index(out_dir, plotted_eps, formats) if plotted_eps and not args.interactive_only else None
    html_interactive = None
    if plotted_eps and interactive_enabled:
        html_interactive = write_interactive_group_html(
            out_dir, payload_path, info, args.expected_dim, group_sizes,
            args.interactive_default_group_size,
        )
    if payload_path.exists():
        payload_path.unlink()

    manifest_csv = write_manifest_csv(out_dir, manifest_rows)
    print("\n[DONE]")
    print("plot_manifest:", manifest_csv)
    if html:
        print("html:", html)
    if html_interactive:
        print("interactive_html:", html_interactive)
    print("episodes plotted:", len(plotted_eps))

    if args.open_html:
        if html:
            xdg_open(html)
        if html_interactive:
            xdg_open(html_interactive)


if __name__ == "__main__":
    main()
