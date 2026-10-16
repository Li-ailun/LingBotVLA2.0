#!/usr/bin/env python3
"""Portable environment and full30k runtime reproduction helpers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import sys
import tempfile


_FORBIDDEN = (
    re.compile(r"file://", re.IGNORECASE),
    re.compile(r"(?:^|\s)-e\s+", re.IGNORECASE),
    re.compile(r"/home/(?:lixiang|hddData)(?:/|$)"),
)


def find_nonportable_references(text: str) -> list[str]:
    findings: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if any(pattern.search(line) for pattern in _FORBIDDEN):
            finding = f"line {number}: {line.strip()}"
            if finding not in findings:
                findings.append(finding)
    return findings


def sanitize_conda_export(text: str, name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError(f"invalid environment name: {name!r}")
    lines = text.splitlines()
    portable = [line for line in lines if not line.startswith("prefix:")]
    if portable and portable[0].startswith("name:"):
        portable[0] = f"name: {name}"
    else:
        portable.insert(0, f"name: {name}")
    result = "\n".join(portable).rstrip() + "\n"
    findings = find_nonportable_references(result)
    if findings:
        raise ValueError(
            "non-portable references found:\n  " + "\n  ".join(findings)
        )
    return result


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _sanitize_conda_command(args: argparse.Namespace) -> int:
    raw = args.input.read_text(encoding="utf-8")
    result = sanitize_conda_export(raw, args.name)
    _atomic_write(args.output, result)
    return 0


def build_create_plan(
    *,
    root: Path,
    conda_exe: Path,
    environment_dir: Path,
    lingbot_prefix: Path,
    robotwin_prefix: Path,
    update: bool,
) -> list[list[str]]:
    root = root.resolve()
    environment_dir = environment_dir.resolve()
    definitions = (
        ("lingbotvla", environment_dir / "lingbotvla.environment.yml", lingbot_prefix),
        ("RoboTwin", environment_dir / "RoboTwin.environment.yml", robotwin_prefix),
    )
    commands: list[list[str]] = []
    for name, definition, prefix in definitions:
        if not definition.is_file():
            raise FileNotFoundError(
                f"missing verified environment definition: {definition}; "
                "run scripts/export_verified_environments.sh on DRLServer"
            )
        exists = (prefix / "bin/python").exists()
        if exists and not update:
            raise FileExistsError(
                f"environment already exists: {prefix}; pass --update to modify it"
            )
        action = "update" if exists else "create"
        command = [
            str(conda_exe), "env", action, "--prefix", str(prefix),
            "--file", str(definition),
        ]
        if action == "update":
            command.append("--prune")
        commands.append(command)

    commands.extend(
        [
            [
                str(lingbot_prefix / "bin/python"), "-m", "pip", "install", "-e",
                str(root / "lingbot-vla-v2"),
            ],
            [
                str(robotwin_prefix / "bin/python"), "-m", "pip", "install", "-e",
                str(root / "RoboTwin_lingbot/envs/curobo"), "--no-build-isolation",
            ],
        ]
    )
    return commands


def _plan_create_command(args: argparse.Namespace) -> int:
    commands = build_create_plan(
        root=args.root,
        conda_exe=args.conda,
        environment_dir=args.environment_dir,
        lingbot_prefix=args.lingbot_prefix,
        robotwin_prefix=args.robotwin_prefix,
        update=args.update,
    )
    for command in commands:
        print(shlex.join(command))
    return 0


def _asset_base(root: Path, asset: dict) -> Path:
    options = asset.get("path_options") or [asset["path"]]
    candidates = [root / option for option in options]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def validate_assets(root: Path, manifest: dict) -> list[str]:
    root = Path(root).resolve()
    errors: list[str] = []
    for asset in manifest.get("assets", []):
        asset_id = asset.get("id", "unknown")
        base = _asset_base(root, asset)
        if not base.exists():
            errors.append(f"{asset_id}: missing path: {base}")
            continue
        for relative in asset.get("required", []):
            required = base / relative
            if not required.exists():
                errors.append(f"{asset_id}: missing required file: {required}")
        if asset.get("type") == "hf_indexed_checkpoint":
            incomplete = sorted(base.glob("*.incomplete"))
            if incomplete:
                errors.append(f"{asset_id}: incomplete download: {incomplete[0]}")
            index_path = base / "model.safetensors.index.json"
            if index_path.is_file():
                try:
                    index = json.loads(index_path.read_text(encoding="utf-8"))
                    shards = sorted(set(index["weight_map"].values()))
                except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    errors.append(f"{asset_id}: invalid weight index {index_path}: {exc}")
                else:
                    for shard in shards:
                        if not (base / shard).is_file():
                            errors.append(f"{asset_id}: indexed shard missing: {base / shard}")
    return errors


def build_runtime_plan(gpu_ids: str, base_port: int, min_free_mib: int) -> dict:
    parts = gpu_ids.split(",")
    if not parts or any(not re.fullmatch(r"\d+", part) for part in parts):
        raise ValueError("--gpu-ids must be a comma-separated list of non-negative integers")
    ids = [int(part) for part in parts]
    if len(ids) != len(set(ids)):
        raise ValueError("--gpu-ids contains duplicates")
    if not 1 <= base_port <= 65535 or base_port + len(ids) - 1 > 65535:
        raise ValueError("requested inference port range is invalid")
    if min_free_mib <= 0:
        raise ValueError("--min-free-mib must be positive")
    return {
        "gpu_ids": ids,
        "ports": list(range(base_port, base_port + len(ids))),
        "min_free_mib": int(min_free_mib),
    }


def _validate_assets_command(args: argparse.Namespace) -> int:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    errors = validate_assets(args.root, manifest)
    if errors:
        raise ValueError("asset validation failed:\n  " + "\n  ".join(errors))
    print("FULL30K ASSETS VALID")
    return 0


def _runtime_plan_command(args: argparse.Namespace) -> int:
    print(json.dumps(build_runtime_plan(args.gpu_ids, args.base_port, args.min_free_mib)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sanitize = commands.add_parser("sanitize-conda")
    sanitize.add_argument("--input", type=Path, required=True)
    sanitize.add_argument("--output", type=Path, required=True)
    sanitize.add_argument("--name", required=True)
    sanitize.set_defaults(handler=_sanitize_conda_command)
    plan = commands.add_parser("plan-create")
    plan.add_argument("--root", type=Path, required=True)
    plan.add_argument("--conda", type=Path, required=True)
    plan.add_argument("--environment-dir", type=Path, required=True)
    plan.add_argument("--lingbot-prefix", type=Path, required=True)
    plan.add_argument("--robotwin-prefix", type=Path, required=True)
    plan.add_argument("--update", action="store_true")
    plan.set_defaults(handler=_plan_create_command)
    assets = commands.add_parser("validate-assets")
    assets.add_argument("--root", type=Path, required=True)
    assets.add_argument("--manifest", type=Path, required=True)
    assets.set_defaults(handler=_validate_assets_command)
    runtime = commands.add_parser("runtime-plan")
    runtime.add_argument("--gpu-ids", required=True)
    runtime.add_argument("--base-port", type=int, required=True)
    runtime.add_argument("--min-free-mib", type=int, required=True)
    runtime.set_defaults(handler=_runtime_plan_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.handler(args))
    except (OSError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
