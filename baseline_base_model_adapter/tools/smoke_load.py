#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def fail(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    adapter = args.adapter.resolve()
    checkpoint = adapter / "checkpoints/global_step_0/hf_ckpt"
    norm = (args.norm_stats or adapter / "assets/norm_stats/robotwin_clean_50x50.json").resolve()
    if not norm.is_file():
        fail(f"norm stats file does not exist: {norm}")
    if not (adapter / "lingbotvla_cli.yaml").is_file():
        fail("lingbotvla_cli.yaml does not exist")
    if len(list(checkpoint.glob("model-*-of-00006.safetensors"))) != 6:
        fail("checkpoint does not expose exactly 6 model shards")
    if args.check_only:
        print("SMOKE INPUTS VALID")
        return

    repo = adapter.parent / "lingbot-vla-v2"
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server

    server = LingbotVLAv2Server(
        str(checkpoint), use_length=50, chunk_ret=True,
        use_bf16=True, use_fp32=False, use_compile=False,
    )
    parameter = next(server.vla.parameters())
    print(f"STRICT MODEL LOAD OK device={parameter.device} dtype={parameter.dtype}")


if __name__ == "__main__":
    main()
