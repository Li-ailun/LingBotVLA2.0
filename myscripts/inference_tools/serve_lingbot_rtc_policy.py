#!/usr/bin/env python3
"""Entry point for the isolated LingBot-VLA RTC service."""

import argparse
import hashlib
from pathlib import Path

from .lingbot_rtc.lingbot_vla_v2_rtc_policy import LingBotVlaV2RTCPolicy
from .websocket_policy_server import WebsocketPolicyServer


def build_server_metadata(checkpoint_id, normalization_id):
    return {"model_backend": "lingbot-vla-v2", "action_horizon": 50,
            "physical_action_dim": 20, "rtc_capable": True,
            "checkpoint_id": str(checkpoint_id), "normalization_id": str(normalization_id)}


def validate_server_metadata(metadata):
    if metadata.get("model_backend") != "lingbot-vla-v2":
        raise ValueError("backend must be lingbot-vla-v2")
    if metadata.get("action_horizon") != 50 or metadata.get("physical_action_dim") != 20:
        raise ValueError("server action shape must be [50, 20]")
    if metadata.get("rtc_capable") is not True:
        raise ValueError("server must advertise RTC support")


def _identity(path):
    resolved = Path(path).resolve()
    digest = hashlib.sha256(str(resolved).encode()).hexdigest()[:12]
    return f"{resolved.name}:{digest}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--qwen-path", required=True)
    parser.add_argument("--robot-config", required=True)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    required = [Path(args.model_path), Path(args.qwen_path), Path(args.robot_config), Path(args.norm_stats)]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing required production assets:\n" + "\n".join(missing))
    # The real backend is deliberately required from a checkpoint-specific adapter.
    # This prevents a simulator transform or an OpenPI backend from being selected silently.
    raise SystemExit(
        "R1Pro LingBot backend adapter is not sealed yet. Supply the trained checkpoint-specific "
        "robot config/norm mapping before enabling the WebSocket service; protocol and RTC tests remain available."
    )


if __name__ == "__main__":
    main()
