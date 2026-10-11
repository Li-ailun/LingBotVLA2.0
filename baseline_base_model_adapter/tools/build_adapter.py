#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    raise SystemExit(f"Error: {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--qwen-path", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clean-dataset", type=Path)
    args = parser.parse_args()

    source_input = str(args.source_model.absolute())
    if FORBIDDEN in source_input.lower():
        fail(f"forbidden post-trained source: {source_input}")

    source = args.source_model.resolve()
    qwen = args.qwen_path.resolve()
    norm = args.norm_stats.resolve()
    output = args.output.resolve()
    for label, path in (("source model", source), ("Qwen3-VL", qwen), ("norm stats", norm)):
        if not path.exists():
            fail(f"{label} does not exist: {path}")

    shards = sorted(source.glob("model-*-of-00006.safetensors"))
    if len(shards) != 6:
        fail(f"expected exactly 6 model shards, found {len(shards)}")
    metadata = [source / "config.json", source / "model.safetensors.index.json"]
    for path in metadata:
        if not path.is_file():
            fail(f"missing checkpoint metadata: {path}")

    checkpoint = output / "checkpoints/global_step_0/hf_ckpt"
    checkpoint.mkdir(parents=True, exist_ok=True)
    weights = []
    for source_file in [*shards, *metadata]:
        destination = checkpoint / source_file.name
        relative_target = Path(os.path.relpath(source_file, destination.parent))
        if destination.is_symlink() or destination.exists():
            if destination.is_symlink() and destination.readlink() == relative_target:
                pass
            else:
                fail(f"refusing to overwrite adapter artifact: {destination}")
        else:
            destination.symlink_to(relative_target)
        if source_file.suffix == ".safetensors":
            weights.append({
                "name": source_file.name,
                "sha256": sha256(source_file),
                "source": str(source_file.resolve()),
            })

    repo_root = Path(__file__).resolve().parents[2]
    template = repo_root / "lingbot-vla-v2/configs/vla/robotwin/robotwin.yaml"
    if not template.is_file():
        fail(f"official RoboTwin config template missing: {template}")
    config = yaml.safe_load(template.read_text())
    config["model"]["model_path"] = str(source)
    config["model"]["tokenizer_path"] = str(qwen)
    config["data"]["data_name"] = "robotwin"
    config["data"]["joints"] = [str(item) for item in config["data"]["joints"]]
    config["data"]["norm_type"] = [str(item) for item in config["data"]["norm_type"]]
    if args.clean_dataset:
        config["data"]["train_path"] = str(args.clean_dataset.resolve())
    config["data"]["norm_stats_file"] = str(norm)
    config["train"]["output_dir"] = str(output)
    (output / "lingbotvla_cli.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    manifest = {
        "adapter_kind": "zero-shot-foundation-model",
        "forbidden_checkpoint": FORBIDDEN,
        "source_model": str(source),
        "qwen_path": str(qwen),
        "norm_stats": str(norm),
        "clean_dataset": str(args.clean_dataset.resolve()) if args.clean_dataset else None,
        "weights": weights,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(checkpoint)


if __name__ == "__main__":
    main()
