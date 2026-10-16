#!/usr/bin/env python3
"""Portable environment and full30k runtime reproduction helpers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sanitize = commands.add_parser("sanitize-conda")
    sanitize.add_argument("--input", type=Path, required=True)
    sanitize.add_argument("--output", type=Path, required=True)
    sanitize.add_argument("--name", required=True)
    sanitize.set_defaults(handler=_sanitize_conda_command)
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
