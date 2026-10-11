#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml


FORBIDDEN = "lingbot-vla-v2-6b-robotwin"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    args = parser.parse_args()
    adapter = args.adapter.resolve()
    manifest_path = adapter / "manifest.json"
    config_path = adapter / "lingbotvla_cli.yaml"
    if not manifest_path.is_file() or not config_path.is_file():
        fail("adapter manifest or lingbotvla_cli.yaml is missing")
    manifest = json.loads(manifest_path.read_text())
    if FORBIDDEN in manifest.get("source_model", "").lower():
        fail("forbidden post-trained source in manifest")

    checkpoint = adapter / "checkpoints/global_step_0/hf_ckpt"
    shards = sorted(checkpoint.glob("model-*-of-00006.safetensors"))
    if len(shards) != 6:
        fail(f"expected exactly 6 model shards, found {len(shards)}")
    hashes = {item["name"]: item["sha256"] for item in manifest.get("weights", [])}
    for shard in shards:
        if not shard.is_symlink() or shard.readlink().is_absolute():
            fail(f"weight must be a relative symbolic link: {shard}")
        if not shard.resolve().is_file():
            fail(f"broken weight link: {shard}")
        if hashes.get(shard.name) != sha256(shard.resolve()):
            fail(f"weight hash mismatch: {shard.name}")

    config = yaml.safe_load(config_path.read_text())
    for key, raw in (
        ("model.model_path", config["model"]["model_path"]),
        ("model.tokenizer_path", config["model"]["tokenizer_path"]),
        ("data.norm_stats_file", config["data"]["norm_stats_file"]),
    ):
        if not Path(raw).exists():
            fail(f"configured path does not exist ({key}): {raw}")
    print("ADAPTER VALID")


if __name__ == "__main__":
    main()
