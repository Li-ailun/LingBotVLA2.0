"""Command-line entrypoint used by isolated evaluation launchers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import subprocess
import yaml

from .recorder import EvaluationRecorder


def _git_commit(path: str | None) -> str | None:
    if not path:
        return None
    try:
        return subprocess.run(
            ["git", "-C", path, "rev-parse", "HEAD"],
            check=True, text=True, capture_output=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _integer_env(name: str) -> int | None:
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else None


def _float_env(name: str) -> float | None:
    value = os.environ.get(name)
    return float(value) if value not in (None, "") else None


def _model_fingerprint(model_path: str | None) -> dict:
    result = {}
    if not model_path:
        return result
    root = Path(model_path)
    for name in ("config.json", "model.safetensors.index.json"):
        path = root / name
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result[name] = {"sha256": digest.hexdigest(), "size": path.stat().st_size}
    return result


def build_metadata(args: argparse.Namespace) -> dict:
    mode = os.environ.get("LINGBOT_EVAL_MODE", "non_rtc")
    model_path = os.environ.get("LINGBOT_MODEL_PATH", args.model_path)
    metadata = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "resume": bool(args.resume),
        "mode": mode,
        "model": {
            "id": os.environ.get("LINGBOT_MODEL_ID"),
            "path": model_path,
            "asset_root": os.environ.get("LINGBOT_MODEL_ASSET_ROOT"),
            "files": _model_fingerprint(model_path),
        },
        "precision": os.environ.get(
            "LINGBOT_PRECISION", "bf16" if args.use_bf16 == "True" else "fp32"
        ),
        "evaluation": {
            "task_config": args.task_config,
            "episodes_per_task": args.episodes,
            "start_seed": args.start_seed,
            "initial_episode_count": args.initial_episode_count,
            "initial_success_count": args.initial_success_count,
            "video_enabled": args.video_enabled == "True",
        },
        "inference": {
            "initial_seed": 42,
            "action_chunk": args.action_chunk,
            "torch_compile": args.use_compile == "True",
            "num_gpus": args.num_gpus,
            "servers_per_gpu": args.num_servers,
            "physical_gpu_ids": [int(value) for value in args.gpu_ids.split(",")],
        },
        "code": {
            "project_commit": _git_commit(os.environ.get("LINGBOT_PROJECT_ROOT")),
            "robotwin_commit": _git_commit(args.eval_workdir),
        },
    }
    if mode == "rtc":
        rtc_yaml = {}
        rtc_config_path = os.environ.get("LINGBOT_RTC_CONFIG")
        if rtc_config_path and Path(rtc_config_path).is_file():
            rtc_yaml = yaml.safe_load(Path(rtc_config_path).read_text(encoding="utf-8")) or {}
        request = _integer_env("LINGBOT_RTC_REQUEST_INDEX")
        request = int(rtc_yaml.get("request_index")) if request is None else request
        delay = _integer_env("LINGBOT_RTC_FIXED_DELAY_STEPS")
        delay = int(rtc_yaml.get("fixed_delay_steps")) if delay is None else delay
        horizon = _integer_env("LINGBOT_RTC_EXECUTION_HORIZON")
        horizon = int(rtc_yaml.get("execution_horizon")) if horizon is None else horizon
        crossfade = _integer_env("LINGBOT_RTC_CROSSFADE_STEPS")
        if crossfade is None:
            crossfade = int((rtc_yaml.get("smoothing") or {}).get("crossfade_steps"))
        ema = _float_env("LINGBOT_RTC_EMA_ALPHA")
        if ema is None:
            ema = float((rtc_yaml.get("smoothing") or {}).get("ema_alpha"))
        metadata["rtc"] = {
            "request_index": request,
            "fixed_delay_steps": delay,
            "switch_index": request + delay if request is not None and delay is not None else None,
            "execution_horizon": horizon,
            "crossfade_steps": crossfade,
            "ema_alpha": ema,
        }
    else:
        metadata["rtc"] = None
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    init = subparsers.add_parser("init-run")
    init.add_argument("--run-dir", required=True)
    init.add_argument("--task-config", required=True)
    init.add_argument("--episodes", type=int, required=True)
    init.add_argument("--action-chunk", type=int, required=True)
    init.add_argument("--use-compile", required=True)
    init.add_argument("--use-bf16", default="False")
    init.add_argument("--model-path", default=None)
    init.add_argument("--num-gpus", type=int, required=True)
    init.add_argument("--num-servers", type=int, required=True)
    init.add_argument("--gpu-ids", default="0")
    init.add_argument("--eval-workdir", default=None)
    init.add_argument("--start-seed", type=int, default=None)
    init.add_argument("--initial-episode-count", type=int, default=0)
    init.add_argument("--initial-success-count", type=int, default=0)
    init.add_argument("--video-enabled", default="False")
    init.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "init-run":
        recorder = EvaluationRecorder(Path(args.run_dir), task="__run__")
        recorder.record_session(build_metadata(args))


if __name__ == "__main__":
    main()
