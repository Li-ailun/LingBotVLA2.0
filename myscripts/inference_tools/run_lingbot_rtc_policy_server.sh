#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../.." && pwd)"
: "${LINGBOT_MODEL_PATH:?Set LINGBOT_MODEL_PATH to the trained LingBot checkpoint}"
: "${QWEN3VL_PATH:?Set QWEN3VL_PATH to Qwen3-VL}"
: "${R1PRO_ROBOT_CONFIG:?Set R1PRO_ROBOT_CONFIG to the sealed R1Pro config}"
: "${R1PRO_NORM_STATS:?Set R1PRO_NORM_STATS to its matching norm_stats.json}"

export PYTHONPATH="$ROOT/lingbot-vla-v2:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec /home/lixiang/miniconda3/envs/lingbotvla/bin/python -m myscripts.inference_tools.serve_lingbot_rtc_policy \
  --model-path "$LINGBOT_MODEL_PATH" --qwen-path "$QWEN3VL_PATH" \
  --robot-config "$R1PRO_ROBOT_CONFIG" --norm-stats "$R1PRO_NORM_STATS" "$@"
