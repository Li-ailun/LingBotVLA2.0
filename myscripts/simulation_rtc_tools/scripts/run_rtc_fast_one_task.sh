#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TOOL="$(cd -- "$HERE/.." && pwd)"
ROOT="$(cd -- "$TOOL/../.." && pwd)"
source "$ROOT/evaluation/runtime_paths.sh"
resolve_lingbot_runtime_paths
TASK=""
EPISODES=1
DRY_RUN=false
MODEL_NAME=step2000
PRECISION=bf16
RESUME_RUN_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --model) MODEL_NAME="$2"; shift 2 ;;
    --precision) PRECISION="$2"; shift 2 ;;
    --resume-run-dir) RESUME_RUN_DIR="$2"; shift 2 ;;
    --request-index) export LINGBOT_RTC_REQUEST_INDEX="$2"; shift 2 ;;
    --execution-horizon) export LINGBOT_RTC_EXECUTION_HORIZON="$2"; shift 2 ;;
    --fixed-delay-steps) export LINGBOT_RTC_FIXED_DELAY_STEPS="$2"; shift 2 ;;
    --crossfade-steps) export LINGBOT_RTC_CROSSFADE_STEPS="$2"; shift 2 ;;
    --ema-alpha) export LINGBOT_RTC_EMA_ALPHA="$2"; shift 2 ;;
    --*) echo "Unknown option: $1" >&2; exit 2 ;;
    *)
      [[ -z "$TASK" ]] || { echo "Unexpected positional argument: $1" >&2; exit 2; }
      TASK="$1"; shift ;;
  esac
done
[[ -n "$TASK" ]] || { echo "usage: $0 TASK|--task TASK [options]" >&2; exit 2; }

source "$ROOT/evaluation/model_registry.sh"
resolve_lingbot_model "$MODEL_NAME" "$ROOT"
resolve_lingbot_precision "$PRECISION"

export QWEN3VL_PATH="$ROOT/models/Qwen3-VL-4B-Instruct"
export LINGBOT_RTC_TOOL_ROOT="$TOOL"
export LINGBOT_RTC_PROJECT_ROOT="$ROOT"
export LINGBOT_PROJECT_ROOT="$ROOT"
export LINGBOT_EVAL_MODE=rtc
export LINGBOT_PROVENANCE_ENABLED=1
export LINGBOT_EVAL_CLIENT_SOURCE="$TOOL/copied_eval/eval_policy_client_lingbotvla_rtc.py"
export LINGBOT_EVAL_CLIENT_NAME=eval_policy_client_lingbotvla_rtc.py
export LINGBOT_RTC_CONFIG="$TOOL/configs/lingbot_action_step2000_rtc_fast.yaml"

RTC_CONFIG_PY="${LINGBOT_RTC_CONFIG_PY:-$ROBOTWIN_PYTHON}"
RTC_SUMMARY=$(PYTHONPATH="$TOOL:$ROOT" "$RTC_CONFIG_PY" -c '
import os
from simulation_rtc.config import RTCConfig
c = RTCConfig.from_yaml(os.environ["LINGBOT_RTC_CONFIG"])
print(f"RTC effective config: request_index={c.request_index} fixed_delay_steps={c.fixed_delay_steps} execution_horizon={c.execution_horizon} crossfade_steps={c.crossfade_steps} ema_alpha={c.ema_alpha}")
')
echo "$RTC_SUMMARY"

EXTRA=()
if [[ -n "$RESUME_RUN_DIR" ]]; then
  EXTRA+=(--resume_run_dir "$RESUME_RUN_DIR")
fi

COMMAND=(bash "$HERE/start_robotwin_rtc_infer_and_eval.sh"
  --model_path "$LINGBOT_MODEL_PATH"
  --inference_script myscripts/simulation_rtc_tools/server/policy_server_rtc.py
  --inference_workdir "$ROOT/"
  --eval_workdir "$ROOT/RoboTwin_lingbot"
  --output_base "$TOOL/results/$LINGBOT_MODEL_ID/$LINGBOT_PRECISION/clean"
  --conda_sh "$LINGBOT_CONDA_SH"
  --inference_env lingbotvla --sim_env RoboTwin
  --task_config demo_clean --task_name "$TASK"
  --num_tasks 1 --num_episodes "$EPISODES"
  --num_gpus 1 --num_per_gpu 1 --use_length 50
  --use_bf16 "$LINGBOT_USE_BF16" --use_fp32 "$LINGBOT_USE_FP32" --use_compile True --no_video
  "${EXTRA[@]}")

if $DRY_RUN; then
  printf 'LINGBOT_MODEL_ASSET_ROOT=%q LINGBOT_NORM_STATS=%q LINGBOT_ROBOT_CONFIG=%q LINGBOT_TRAINING_CONFIG=%q ' \
    "$LINGBOT_MODEL_ASSET_ROOT" "$LINGBOT_NORM_STATS" "$LINGBOT_ROBOT_CONFIG" "$LINGBOT_TRAINING_CONFIG"
  for name in REQUEST_INDEX EXECUTION_HORIZON FIXED_DELAY_STEPS CROSSFADE_STEPS EMA_ALPHA; do
    var="LINGBOT_RTC_${name}"
    [[ -v "$var" ]] && printf '%s=%q ' "$var" "${!var}"
  done
  printf 'LINGBOT_RTC_EVAL_CLIENT=%q ' "$TOOL/copied_eval/eval_policy_client_lingbotvla_rtc.py"
  printf 'LINGBOT_EVAL_MODE=%q LINGBOT_PROVENANCE_ENABLED=%q ' "$LINGBOT_EVAL_MODE" "$LINGBOT_PROVENANCE_ENABLED"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi
cd "$ROOT"
exec "${COMMAND[@]}"
