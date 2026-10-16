"""Build a validated per-task plan for parallel evaluation resume."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable


class ResumePlanError(ValueError):
    """Raised when an existing run cannot be resumed safely."""


@dataclass(frozen=True)
class ResumeTask:
    task: str
    episode_count: int
    success_count: int
    next_seed: int


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResumePlanError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ResumePlanError(f"expected a JSON object in {path}")
    return value


def _manifest_value(manifest: dict[str, Any], key: str) -> Any:
    paths = {
        "mode": ("mode",),
        "model_id": ("model", "id"),
        "precision": ("precision",),
        "task_config": ("evaluation", "task_config"),
        "instruction_type": ("evaluation", "instruction_type"),
        "episodes": ("evaluation", "episodes_per_task"),
        "request_index": ("rtc", "request_index"),
        "fixed_delay_steps": ("rtc", "fixed_delay_steps"),
        "execution_horizon": ("rtc", "execution_horizon"),
        "crossfade_steps": ("rtc", "crossfade_steps"),
        "ema_alpha": ("rtc", "ema_alpha"),
    }
    value: Any = manifest
    for component in paths[key]:
        if not isinstance(value, dict) or component not in value:
            return None
        value = value[component]
    return value


def _validate_manifest(run_dir: Path, expected: dict[str, Any]) -> None:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise ResumePlanError(f"missing run_manifest.json: {manifest_path}")
    manifest = _read_json(manifest_path)
    mismatches = []
    for key, wanted in expected.items():
        if wanted is None:
            continue
        actual = _manifest_value(manifest, key)
        equal = abs(float(actual) - float(wanted)) < 1e-12 if key == "ema_alpha" and actual is not None else actual == wanted
        if not equal:
            mismatches.append(f"{key}: existing={actual!r}, requested={wanted!r}")
    if mismatches:
        raise ResumePlanError("resume configuration mismatch: " + "; ".join(mismatches))


def _read_task_rows(run_dir: Path, task: str) -> list[dict[str, Any]]:
    path = run_dir / "eval_results" / task / "episodes.jsonl"
    if not path.is_file():
        log_path = run_dir / "eval_logs" / f"{task}.log"
        if log_path.is_file() and "Success rate:" in log_path.read_text(encoding="utf-8", errors="replace"):
            raise ResumePlanError(
                f"{task} has progress in {log_path} but no {path}; use manual single-task resume"
            )
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ResumePlanError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ResumePlanError(f"expected JSON object in {path}:{line_number}")
        rows.append(row)
    return rows


def build_resume_plan(
    run_dir: str | Path,
    tasks: Iterable[str],
    expected: dict[str, Any],
) -> list[ResumeTask]:
    root = Path(run_dir).resolve()
    _validate_manifest(root, expected)
    target = int(expected["episodes"])
    plan = []
    for task in tasks:
        rows = _read_task_rows(root, task)
        indices = [row.get("episode_index") for row in rows]
        if indices != list(range(len(rows))):
            raise ResumePlanError(
                f"{task} episode_index values must be unique and contiguous from 0; got {indices}"
            )
        if len(rows) > target:
            raise ResumePlanError(f"{task} has {len(rows)} episodes, exceeding requested target {target}")
        for row in rows:
            if row.get("task") != task:
                raise ResumePlanError(f"{task} contains a row recorded for task {row.get('task')!r}")
            if not isinstance(row.get("environment_seed"), int):
                raise ResumePlanError(f"{task} contains a non-integer environment_seed")
        seeds = [row["environment_seed"] for row in rows]
        if any(current <= previous for previous, current in zip(seeds, seeds[1:])):
            raise ResumePlanError(f"{task} environment_seed values must be strictly increasing")
        if len(rows) == target:
            continue
        plan.append(
            ResumeTask(
                task=task,
                episode_count=len(rows),
                success_count=sum(row.get("success") is True for row in rows),
                next_seed=(rows[-1]["environment_seed"] + 1) if rows else 100000,
            )
        )
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--instruction-type", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--request-index", type=int)
    parser.add_argument("--fixed-delay-steps", type=int)
    parser.add_argument("--execution-horizon", type=int)
    parser.add_argument("--crossfade-steps", type=int)
    parser.add_argument("--ema-alpha", type=float)
    args = parser.parse_args()
    expected = vars(args).copy()
    run_dir = expected.pop("run_dir")
    tasks = expected.pop("task")
    try:
        plan = build_resume_plan(run_dir, tasks, expected)
    except ResumePlanError as exc:
        parser.error(str(exc))
    for item in plan:
        print(f"{item.task}\t{item.episode_count}\t{item.success_count}\t{item.next_seed}")


if __name__ == "__main__":
    main()
