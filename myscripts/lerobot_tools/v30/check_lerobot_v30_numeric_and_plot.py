#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Numeric plots for RoboTwin 2.0 LeRobot v3.0 (read-only dataset access).


cd /home/lixiang/PROJECT/LingBotVLA2.0
conda activate lingbotvla
python myscripts/lerobot_tools/v30/check_lerobot_v30_numeric_and_plot.py \
  --root "$PWD/RoboTwin2_data/lerobot_dataset/RoboTwin_lerobot_v30" \
  --out-dir "$PWD/myscripts/lerobot_tools/v30/results/robotwin_v30_numeric" \
  --interactive-only \
  --interactive-default-group-size 1 \
  --interactive-group-sizes 1,5,10,20 \
  --open-html


  
Preserves the LeRobot v2.1 plotting tool's workflows:
  - 14D state/action overlay, action rows, state rows, absolute frame jumps;
  - static HTML gallery;
  - standalone interactive Plotly HTML with group sizes 1/5/10/20;
  - episode/range selection, interactive-only, optional xdg-open.

Differences from the old 20D R1Pro script:
  - RoboTwin's 14D ordering: left arm 0:6, left gripper 6,
    right arm 7:13, right gripper 13;
  - LeRobot v3 data/chunk-*/file-*.parquet (episodes may cross parquet files);
  - LeRobot v3 nested feature names info['features'][...]['names'] = [[...]];
  - interactive episode samples are gzip-packed and decompressed on demand,
    avoiding allocating all 2500 episodes in the browser at once;
  - an optional numerical alignment summary (not an image alignment audit).

No write operations on the dataset path, and no video decoding.
"""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ROOT = Path(
    "/home/lixiang/PROJECT/LingBotVLA2.0/"
    "RoboTwin2_data/lerobot_dataset/RoboTwin_lerobot_v30"
)
DEFAULT_OUT = Path(
    "/home/lixiang/PROJECT/LingBotVLA2.0/"
    "myscripts/lerobot_tools/results/robotwin_v30_numeric"
)
DEFAULT_ARM_DIMS = tuple(list(range(6)) + list(range(7, 13)))
DEFAULT_GRIPPER_DIMS = (6, 13)


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_group_sizes(s: str) -> list[int]:
    sizes = sorted(set(x for x in parse_int_list(s) if x > 0))
    if not sizes:
        sizes = [1, 5, 10, 20]
    if 1 not in sizes:
        sizes.insert(0, 1)
    return sizes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot RoboTwin LeRobot v3.0 numeric state/action data (14D)."
    )
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--start", type=int, default=0, help="Inclusive episode index")
    p.add_argument("--end", type=int, default=None, help="Exclusive episode index")
    p.add_argument("--episodes", default="", help="Comma-separated episode indices; overrides start/end")
    p.add_argument("--plot-mode", choices=("none", "all", "range", "episodes"), default="all")
    p.add_argument("--plot-format", default="png", help="png, pdf, or png,pdf")
    p.add_argument("--dpi", type=int, default=140)
    p.add_argument("--expected-dim", type=int, default=None, help="Auto-detected from meta/info.json; must equal 14 for RoboTwin")
    p.add_argument("--arm-dims", default=",".join(map(str, DEFAULT_ARM_DIMS)))
    p.add_argument("--gripper-dims", default=",".join(map(str, DEFAULT_GRIPPER_DIMS)))
    p.add_argument("--jump-reference-lines", action="store_true")
    p.add_argument("--arm-jump-reference", type=float, default=0.60)
    p.add_argument("--gripper-jump-reference", type=float, default=1.0)
    p.add_argument("--interactive-groups", action="store_true")
    p.add_argument("--interactive-group-sizes", default="1,5,10,20")
    p.add_argument("--interactive-default-group-size", type=int, default=1)
    p.add_argument("--interactive-only", action="store_true",
                   help="Only generate interactive HTML; skip static PNG/PDF and index.html")
    p.add_argument("--offline-plotly", action="store_true",
                   help="Embed Plotly JS from installed plotly package; otherwise uses Plotly CDN")
    p.add_argument("--open-html", action="store_true", help="Open generated HTML via xdg-open")
    return p.parse_args(argv)


def get_feature_names(info: dict, key: str, dim: int) -> list[str]:
    feat = info.get("features", {}).get(key, {})
    names = feat.get("names", [])
    # LeRobot v3 uses nested names: [[left_joint_0, ..., right_joint_6]].
    if isinstance(names, list) and len(names) == 1 and isinstance(names[0], list):
        names = names[0]
    if isinstance(names, list) and len(names) == dim and all(isinstance(x, str) for x in names):
        return names
    fallback = ([f"left_joint_{i}" for i in range(7)] +
                [f"right_joint_{i}" for i in range(7)])
    return fallback if dim == 14 else [f"{key}[{i}]" for i in range(dim)]


def load_info(root: Path) -> tuple[dict, int, float]:
    path = root / "meta/info.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    info = json.loads(path.read_text(encoding="utf-8"))
    state_shape = info.get("features", {}).get("observation.state", {}).get("shape")
    action_shape = info.get("features", {}).get("action", {}).get("shape")
    if not state_shape or not action_shape:
        raise ValueError("meta/info.json has no state/action feature shapes")
    dim = int(state_shape[0])
    if int(action_shape[0]) != dim:
        raise ValueError(f"State/action dims mismatch: {state_shape}, {action_shape}")
    fps = float(info.get("fps", 15))
    if fps <= 0:
        raise ValueError("FPS must be > 0")
    return info, dim, fps


def load_selected_episodes(root: Path, allowed) -> tuple[dict[int, pd.DataFrame], dict[int, list[str]]]:
    """Read only numeric columns; gather episode parts across parquet files."""
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {root / 'data'}")
    columns = ["episode_index", "frame_index", "timestamp", "observation.state", "action"]
    parts: dict[int, list[pd.DataFrame]] = defaultdict(list)
    sources: dict[int, list[str]] = defaultdict(list)
    for file_no, p in enumerate(files, 1):
        df = pd.read_parquet(p, columns=columns)
        for ep_id, group in df.groupby("episode_index", sort=False):
            ep = int(ep_id)
            if allowed(ep):
                parts[ep].append(group.copy())
                sources[ep].append(str(p))
        print(f"[READ {file_no:2d}/{len(files)}] {p.relative_to(root)}  selected_episodes={len(parts)}", flush=True)
    out = {}
    for ep, segments in parts.items():
        df = pd.concat(segments, ignore_index=True)
        df = df.sort_values("frame_index", kind="stable").reset_index(drop=True)
        out[ep] = df
    return dict(sorted(out.items())), sources


def unpack(df: pd.DataFrame, expected_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state = np.asarray(df["observation.state"].to_list(), dtype=np.float64)
    action = np.asarray(df["action"].to_list(), dtype=np.float64)
    if state.ndim != 2 or action.ndim != 2 or state.shape != action.shape or state.shape[1] != expected_dim:
        raise ValueError(f"Unexpected state/action shape: state={state.shape}, action={action.shape}; dim={expected_dim}")
    ts = df["timestamp"].to_numpy(dtype=np.float64)
    frames = df["frame_index"].to_numpy(dtype=np.int64)
    return ts, frames, state, action


def summarize(ep: int, ts: np.ndarray, frames: np.ndarray, state: np.ndarray,
              action: np.ndarray, fps: float, sources: list[str]) -> dict:
    n = len(ts)
    df = np.diff(frames)
    dt = np.diff(ts)
    valid = df == 1
    err = np.abs(action[:-1][valid] - state[1:][valid]) if n > 1 else np.empty((0, state.shape[1]))
    jump = np.abs(np.diff(state, axis=0)) if n > 1 else np.empty((0, state.shape[1]))
    return {
        "episode_index": ep, "frames": n,
        "first_frame_index": int(frames[0]) if n else "",
        "last_frame_index": int(frames[-1]) if n else "",
        "nonconsecutive_frame_pairs": int(np.count_nonzero(~valid)),
        "timestamp_step_anomalies": int(np.count_nonzero(np.abs(dt[valid] - 1.0/fps) > max(0.002, 0.05/fps))),
        "action_t_vs_state_tplus1_mae": float(err.mean()) if err.size else "",
        "action_t_vs_state_tplus1_max_error": float(err.max()) if err.size else "",
        "max_abs_state_jump": float(jump.max()) if jump.size else "",
        "parquet_files": ";".join(dict.fromkeys(sources)),
    }


def use_plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def draw_rows(plt, x: np.ndarray, state: np.ndarray, action: np.ndarray,
              names: list[str], title: str, filename: Path,
              dpi: int, kind: str, reference_lines: bool, arms: set[int],
              grippers: set[int], arm_ref: float, gripper_ref: float) -> None:
    dim = state.shape[1]
    fig, axes = plt.subplots(dim, 1, figsize=(18, max(16, dim * 1.35)), sharex=True)
    if dim == 1:
        axes = [axes]
    if kind == "jump":
        ystate = np.abs(np.diff(state, axis=0))
        yaction = np.abs(np.diff(action, axis=0))
        xx = (x[:-1] + x[1:]) * 0.5
    else:
        ystate = state
        yaction = action
        xx = x
    for i, ax in enumerate(axes):
        if kind in ("overlay", "state", "jump"):
            ax.plot(xx, ystate[:, i], linewidth=1.05,
                    label="abs diff state" if kind == "jump" else "state")
        if kind in ("overlay", "action", "jump"):
            ax.plot(xx, yaction[:, i], linewidth=1.05, linestyle="--",
                    label="abs diff action" if kind == "jump" else "action")
        if kind == "jump" and reference_lines:
            if i in arms:
                ax.axhline(arm_ref, linestyle=":", linewidth=0.9)
            elif i in grippers:
                ax.axhline(gripper_ref, linestyle=":", linewidth=0.9)
        ax.set_ylabel(f"{i}\n{names[i]}", rotation=0, labelpad=63, fontsize=8, va="center")
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.tick_params(axis="y", labelsize=8)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(title)
    axes[-1].set_xlabel("timestamp (seconds)")
    fig.tight_layout()
    fig.subplots_adjust(left=0.16, right=0.985)
    fig.savefig(filename, dpi=dpi)
    plt.close(fig)


def make_static_plots(ep: int, x: np.ndarray, state: np.ndarray, action: np.ndarray,
                      names: list[str], plots_dir: Path, args, plt, arms: set[int], grippers: set[int]) -> None:
    dirname = f"episode_{ep:06d}"
    d = plots_dir / dirname
    d.mkdir(parents=True, exist_ok=True)
    for fmt in [x.strip() for x in args.plot_format.split(",") if x.strip()]:
        if fmt not in {"png", "pdf", "svg"}:
            raise ValueError(f"Unsupported --plot-format: {fmt}")
        for kind, suffix in [
            ("overlay", "state_action_overlay"), ("action", "action_rows"),
            ("state", "state_rows"), ("jump", "jump_rows")
        ]:
            path = d / f"{dirname}_{suffix}.{fmt}"
            draw_rows(plt, x, state, action, names,
                      f"{dirname} {suffix}", path, args.dpi,
                      kind, args.jump_reference_lines,
                      arms, grippers, args.arm_jump_reference, args.gripper_jump_reference)


def static_html(out_dir: Path, plotted: list[int], fmt: str) -> Path:
    target = out_dir / "plots/index.html"
    lines = ["<!doctype html><html><head><meta charset='utf-8'><title>RoboTwin v3 numeric plots</title>",
             "<style>body{font:14px system-ui;margin:25px;color:#222;background:#fafafa}img{max-width:100%;width:1250px;border:1px solid #ccc;margin-bottom:18px}h2{margin-top:40px}</style></head><body>",
             "<h1>RoboTwin LeRobot v3.0 numeric plots</h1><p>Raw state and action; timestamp is the x axis.</p>"]
    for ep in plotted:
        e = f"episode_{ep:06d}"
        lines.append(f"<h2>{e}</h2>")
        for suffix in ["state_action_overlay", "action_rows", "state_rows", "jump_rows"]:
            rel = f"{e}/{e}_{suffix}.{fmt}"
            lines.append(f"<p><a href='{rel}'>{suffix}</a></p>")
            if fmt != "pdf":
                lines.append(f"<img loading='lazy' src='{rel}'>")
    lines.append("</body></html>")
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def pack_payload(ep: int, x: np.ndarray, state: np.ndarray, action: np.ndarray) -> str:
    # Round JSON data, not the original Parquet values. Works on 2500 episodes.
    packed = {"episode_index": int(ep), "x": np.round(x, 6).tolist(),
              "state": [np.round(state[:, i], 6).tolist() for i in range(state.shape[1])],
              "action": [np.round(action[:, i], 6).tolist() for i in range(action.shape[1])]}
    js = json.dumps(packed, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return base64.b64encode(gzip.compress(js, compresslevel=6)).decode("ascii")


def plotly_script_tag(offline: bool) -> str:
    if offline:
        try:
            from plotly.offline import get_plotlyjs
        except ImportError as exc:
            raise RuntimeError("--offline-plotly requires the 'plotly' Python package") from exc
        js = get_plotlyjs().replace("</script", "<\\/script")
        return f"<script>{js}</script>"
    return "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>"


INTERACTIVE_JS = r"""
const RECORDS = Object.freeze(JSON.parse(document.getElementById('payload').textContent));
const NAMES = JSON.parse(document.getElementById('names').textContent);
const GROUP_SIZES = JSON.parse(document.getElementById('group-sizes').textContent);
let groupSize = Number(document.getElementById('default-size').textContent);
let groupStart = 0, dim = 0, mode = 'overlay', renderId = 0;
const cache = new Map();

async function unpack(record) {
  if (cache.has(record.episode_index)) return cache.get(record.episode_index);
  if (!('DecompressionStream' in window)) throw new Error('Browser needs DecompressionStream (modern Chromium/Firefox).');
  const binary = atob(record.data);
  const arr = Uint8Array.from(binary, c => c.charCodeAt(0));
  const decompressed = new Blob([arr]).stream().pipeThrough(new DecompressionStream('gzip'));
  const obj = JSON.parse(await new Response(decompressed).text());
  cache.set(record.episode_index, obj);
  if (cache.size > 32) cache.delete(cache.keys().next().value);
  return obj;
}
function currentGroup(){
  const end = Math.max(0, RECORDS.length-groupSize);
  groupStart = Math.max(0, Math.min(end, groupStart));
  return RECORDS.slice(groupStart, groupStart+groupSize);
}
function setGroupSize(s){groupSize=s;groupStart=Math.floor(groupStart/s)*s;render();}
function nextGroup(){groupStart+=groupSize;if(groupStart>=RECORDS.length)groupStart=0;render();}
function prevGroup(){groupStart-=groupSize;if(groupStart<0)groupStart=Math.max(0,Math.floor((RECORDS.length-1)/groupSize)*groupSize);render();}
function setDim(v){dim=Math.max(0,Math.min(NAMES.length-1,v));document.getElementById('dim').value=String(dim);render();}
function setMode(v){mode=v;document.getElementById('mode').value=v;render();}
function makeTrace(ep, typ) {
  const x = ep.x, y = ep[typ][dim];
  return {x, y, type:'scattergl', mode:'lines', name:`ep_${String(ep.episode_index).padStart(6,'0')} ${typ}`,
    line:{width:1.5,dash:typ==='action'?'dash':'solid'},
    customdata:ep.x.map((t,i)=>[ep.episode_index,i]),
    hovertemplate:'episode=%{customdata[0]}<br>frame=%{customdata[1]}<br>time=%{x:.5f}s<br>value=%{y:.6f}<extra></extra>'};
}
async function render(){
  const serial=++renderId;
  const group=currentGroup();
  document.getElementById('status').textContent=`Loading ${group.length} episodes (group=${groupSize}, dim=${dim}, mode=${mode})...`;
  try {
    if (!window.Plotly) throw new Error('Plotly not found. Connect to internet or regenerate with --offline-plotly.');
    const payload=await Promise.all(group.map(unpack));
    if (serial!==renderId) return;
    const traces=[];
    payload.forEach(ep=>{
      if(mode==='state'||mode==='overlay') traces.push(makeTrace(ep,'state'));
      if(mode==='action'||mode==='overlay') traces.push(makeTrace(ep,'action'));
    });
    const ids=payload.map(ep=>ep.episode_index).join(', ');
    document.getElementById('status').textContent=`${RECORDS.length} episodes loaded in index | displaying ${ids} | dim=${dim} ${NAMES[dim]} | ${mode}`;
    const layout={title:`Dimension ${dim}: ${NAMES[dim]}`, hovermode:'closest',
      xaxis:{title:'timestamp (s)'},yaxis:{title:'value'},legend:{orientation:'h'},
      margin:{l:70,r:20,t:60,b:70},paper_bgcolor:'#fff',plot_bgcolor:'#fff'};
    await Plotly.react('plot',traces,layout,{responsive:true,scrollZoom:true,displaylogo:false});
  } catch(e) {
    if(serial===renderId)document.getElementById('status').textContent=`ERROR: ${e.message}`;
    console.error(e);
  }
}
function init(){
  for(let i=0;i<NAMES.length;i++){
    const opt=document.createElement('option');opt.value=String(i);opt.textContent=`${i}: ${NAMES[i]}`;
    document.getElementById('dim').appendChild(opt);
  }
  GROUP_SIZES.forEach(gs=>{
    const b=document.createElement('button');b.textContent=String(gs);
    b.onclick=()=>setGroupSize(gs);document.getElementById('groups').appendChild(b);
  });
  document.getElementById('dim').onchange=e=>setDim(Number(e.target.value));
  document.getElementById('mode').onchange=e=>setMode(e.target.value);
  document.getElementById('prev').onclick=prevGroup;
  document.getElementById('next').onclick=nextGroup;
  document.addEventListener('keydown',e=>{
    if(e.target && ['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName))return;
    if(e.key==='1')setGroupSize(1);else if(e.key==='5')setGroupSize(5);
    else if(e.key==='0')setGroupSize(10);else if(e.key==='2')setGroupSize(20);
    else if(e.key==='ArrowRight')nextGroup();else if(e.key==='ArrowLeft')prevGroup();
    else if(e.key==='ArrowDown')setDim(dim+1);else if(e.key==='ArrowUp')setDim(dim-1);
    else if(e.key.toLowerCase()==='s')setMode('state');
    else if(e.key.toLowerCase()==='a')setMode('action');
    else if(e.key.toLowerCase()==='o')setMode('overlay');
  });
  render();
}
init();
"""


def interactive_html(out_dir: Path, records: list[dict], names: list[str],
                     sizes: list[int], default_size: int, offline: bool) -> Path:
    target = out_dir / "plots/interactive_episode_groups.html"
    if default_size not in sizes:
        default_size = sizes[0]
    payload = json.dumps(records, separators=(",", ":"))
    script_tag = plotly_script_tag(offline)
    html_text = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>RoboTwin v3 Interactive Episode Groups</title>
<style>
body {{font:14px system-ui,Arial,sans-serif;background:#f8fafc;color:#111827;margin:18px}}
h1 {{font-size:21px}} .bar {{position:sticky;top:0;background:white;border:1px solid #d3dbe6;
 padding:10px;z-index:5;border-radius:8px;box-shadow:0 1px 3px #0001}}
button,select {{padding:6px 10px;margin:3px;border:1px solid #b8c1cf;border-radius:5px;background:white}}
#plot {{height:750px;width:100%;border:1px solid #d3dbe6;background:white;margin-top:10px}}
#status {{margin-top:6px;color:#475569}} .hint {{color:#64748b;font-size:12px}}
</style>{script_tag}</head><body>
<h1>RoboTwin 2.0 / LeRobot v3.0 — 14D State &amp; Action</h1>
<div class="bar"><span>Group size </span><span id="groups"></span>
<button id="prev">Previous</button><button id="next">Next</button>
<label> Dimension <select id="dim"></select></label>
<label> Mode <select id="mode"><option value="overlay">state + action</option>
<option value="state">state</option><option value="action">action</option></select></label>
<div id="status"></div><div class="hint">Keyboard: 1/5/0/2 = group 1/5/10/20;
 ←/→ = previous/next; ↑/↓ = dimension; s/a/o = mode.
 Raw absolute actions; the dataset may use action[t] = state[t+1].
 This page does not inspect video synchronization.</div></div>
<div id="plot"></div>
<script id="payload" type="application/json">{payload}</script>
<script id="names" type="application/json">{json.dumps(names, ensure_ascii=False)}</script>
<script id="group-sizes" type="application/json">{json.dumps(sizes)}</script>
<script id="default-size" type="application/json">{int(default_size)}</script>
<script>{INTERACTIVE_JS}</script></body></html>"""
    target.write_text(html_text, encoding="utf-8")
    return target


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for row in rows:
            wr.writerow({k: row.get(k, "") for k in fields})


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset missing: {root}")
    if root == out_dir or root in out_dir.parents:
        raise ValueError("--out-dir must not be inside the dataset directory")

    info, inferred_dim, fps = load_info(root)
    dim = inferred_dim if args.expected_dim is None else args.expected_dim
    if dim != inferred_dim:
        raise ValueError(f"Expected dim {dim}, dataset has {inferred_dim}")
    if dim != 14:
        raise ValueError(f"This RoboTwin script expects 14D, got {dim}")
    if args.plot_mode == "episodes" and not args.episodes.strip():
        raise ValueError("--plot-mode episodes requires --episodes")
    if args.plot_mode == "none" and not args.interactive_only:
        print("[INFO] plot-mode none: no plots requested")
        return 0

    selection = set(parse_int_list(args.episodes)) if args.episodes.strip() else None
    def allowed(ep: int) -> bool:
        if selection is not None:
            return ep in selection
        return ep >= args.start and (args.end is None or ep < args.end)

    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    arms, grippers = set(parse_int_list(args.arm_dims)), set(parse_int_list(args.gripper_dims))
    if arms & grippers or not arms | grippers <= set(range(dim)):
        raise ValueError("Invalid overlapping/out-of-range arm/gripper dims")

    print(f"[INFO] root={root}\n[INFO] out_dir={out_dir}")
    print(f"[INFO] fps={fps}, dimensions={dim}, total_episodes={info.get('total_episodes')}")
    print(f"[INFO] selected={args.episodes if selection is not None else f'{args.start}:{args.end}'}")
    eps, sources = load_selected_episodes(root, allowed)
    if not eps:
        raise ValueError("No matching episodes were found")
    names = get_feature_names(info, "observation.state", dim)
    for gripper_idx in sorted(grippers):
        names[gripper_idx] += " [gripper]"
    print(f"[INFO] feature order: {list(enumerate(names))}")

    do_static = not args.interactive_only and args.plot_mode != "none"
    do_interactive = args.interactive_only or args.interactive_groups
    plt = use_plotting() if do_static else None
    packed = []
    manifest, summary = [], []
    for ix, (ep, df) in enumerate(eps.items(), 1):
        x, frames, state, action = unpack(df, dim)
        summary.append(summarize(ep, x, frames, state, action, fps, sources.get(ep, [])))
        manifest.append({"episode_index":ep,"frames":len(df),
                         "parquet":";".join(dict.fromkeys(sources.get(ep, []))),
                         "plot_dir":str(plots_dir / f'episode_{ep:06d}')})
        if do_static:
            make_static_plots(ep, x, state, action, names, plots_dir, args, plt, arms, grippers)
        if do_interactive:
            packed.append({"episode_index":ep,"data":pack_payload(ep, x, state, action)})
        if ix % 50 == 0 or ix == len(eps):
            print(f"[PROCESS] {ix}/{len(eps)} episodes", flush=True)

    manifest_path = out_dir / "plot_manifest.csv"
    write_csv(manifest_path, manifest, ["episode_index","frames","parquet","plot_dir"])
    summary_path = out_dir / "numeric_alignment_summary.csv"
    write_csv(summary_path, summary, list(summary[0].keys()))
    index = None
    if do_static:
        fmt = next(s.strip() for s in args.plot_format.split(",") if s.strip())
        index = static_html(out_dir, list(eps.keys()), fmt)
    interactive = None
    if do_interactive:
        sizes = parse_group_sizes(args.interactive_group_sizes)
        interactive = interactive_html(out_dir, packed, names, sizes,
                                       args.interactive_default_group_size,
                                       args.offline_plotly)
    print(f"\n[DONE] episodes={len(eps)}\n[CSV] {manifest_path}\n[CSV] {summary_path}")
    if index:
        print(f"[STATIC] {index}")
    if interactive:
        print(f"[INTERACTIVE] {interactive} ({interactive.stat().st_size / (1024**2):.1f} MiB)")
    if args.open_html and (interactive or index):
        dest = interactive or index
        print(f"[OPEN] {dest}")
        try:
            subprocess.Popen(["xdg-open", str(dest)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (FileNotFoundError, OSError) as exc:
            print(f"[WARN] xdg-open unavailable: {exc}")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
