#!/usr/bin/env python3
"""Verify provenance for copied RLinf and official LingBot source files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_provenance(manifest_path: Path) -> list[str]:
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for entry in manifest.get("sources", []):
        source = Path(entry["source"])
        copied = manifest_path.parent / entry["copied"]
        expected = entry["source_sha256"]
        if not source.is_file():
            errors.append(f"missing source: {source}")
            continue
        if sha256_file(source) != expected:
            errors.append(f"source hash mismatch: {source}")
        if not copied.is_file():
            errors.append(f"missing copy: {copied}")
        elif entry.get("copy_sha256") and sha256_file(copied) != entry["copy_sha256"]:
            errors.append(f"copy hash mismatch: {copied}")
    return errors


if __name__ == "__main__":
    path = Path(__file__).with_name("provenance.json")
    problems = verify_provenance(path)
    if problems:
        raise SystemExit("\n".join(problems))
    print("PROVENANCE OK")
