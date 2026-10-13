#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TOOL="$(cd -- "$HERE/.." && pwd)"
ROOT="$(cd -- "$TOOL/../.." && pwd)"
source "$ROOT/evaluation/runtime_paths.sh"
resolve_lingbot_runtime_paths

SETTING=demo_clean
TASK=""
EPISODES=50
VIDEO_FPS=10
DRY_RUN=false
MODEL_NAME=step2000
PRECISION=bf16
RESUME_RUN_DIR=""
START_SEED=""
GPU_IDS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --setting) SETTING="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --video-fps) VIDEO_FPS="$2"; shift 2 ;;
    --model) MODEL_NAME="$2"; shift 2 ;;
    --precision) PRECISION="$2"; shift 2 ;;
    --resume-run-dir) RESUME_RUN_DIR="$2"; shift 2 ;;
    --start-seed) START_SEED="$2"; shift 2 ;;
    --gpu|--gpu-id|--gpu_id) GPU_IDS="$2"; shift 2 ;;
    --gpu-ids|--gpu_ids) GPU_IDS="$2"; shift 2 ;;
    --request-index) export LINGBOT_RTC_REQUEST_INDEX="$2"; shift 2 ;;
    --execution-horizon) export LINGBOT_RTC_EXECUTION_HORIZON="$2"; shift 2 ;;
    --fixed-delay-steps) export LINGBOT_RTC_FIXED_DELAY_STEPS="$2"; shift 2 ;;
    --crossfade-steps) export LINGBOT_RTC_CROSSFADE_STEPS="$2"; shift 2 ;;
    --ema-alpha) export LINGBOT_RTC_EMA_ALPHA="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -z "$GPU_IDS" || "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
  echo "GPU ids must be comma-separated non-negative integers: $GPU_IDS" >&2; exit 2;
}

source "$ROOT/evaluation/model_registry.sh"
resolve_lingbot_model "$MODEL_NAME" "$ROOT"
resolve_lingbot_precision "$PRECISION"

case "$SETTING" in
  demo_clean|demo_randomized) ;;
  *) echo "setting must be demo_clean or demo_randomized" >&2; exit 2 ;;
esac
[[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || { echo "episodes must be a positive integer" >&2; exit 2; }
[[ "$VIDEO_FPS" =~ ^[1-9][0-9]*$ ]] || { echo "video-fps must be a positive integer" >&2; exit 2; }
[[ -z "$START_SEED" || "$START_SEED" =~ ^[0-9]+$ ]] || {
  echo "start-seed must be a non-negative integer: $START_SEED" >&2; exit 2;
}

NUM_TASKS=50
EXTRA=()
TASK_DIR=all_tasks
if [[ -n "$TASK" ]]; then
  NUM_TASKS=1
  TASK_DIR="$TASK"
  EXTRA+=(--task_name "$TASK")
fi
if [[ -n "$RESUME_RUN_DIR" ]]; then
  EXTRA+=(--resume_run_dir "$RESUME_RUN_DIR")
fi
if [[ -n "$START_SEED" ]]; then
  EXTRA+=(--start_seed "$START_SEED")
fi
if [[ -n "$GPU_IDS" ]]; then
  EXTRA+=(--gpu_ids "$GPU_IDS")
  export LINGBOT_GPU_IDS="$GPU_IDS"
fi

unset ROBOTWIN_REALTIME_VISUALIZATION
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
RTC_DETAILS=$(PYTHONPATH="$TOOL:$ROOT" "$RTC_CONFIG_PY" -c '
import os
from simulation_rtc.config import RTCConfig
c = RTCConfig.from_yaml(os.environ["LINGBOT_RTC_CONFIG"])
print(f"RTC effective config: request_index={c.request_index} fixed_delay_steps={c.fixed_delay_steps} execution_horizon={c.execution_horizon} crossfade_steps={c.crossfade_steps} ema_alpha={c.ema_alpha}")
ema = format(c.ema_alpha, "g").replace(".", "p")
switch = c.request_index + c.fixed_delay_steps
print(f"request_{c.request_index}-delay_{c.fixed_delay_steps}-switch_{switch}-horizon_{c.execution_horizon}-xfade_{c.crossfade_steps}-ema_{ema}")
')
RTC_SUMMARY=$(printf '%s\n' "$RTC_DETAILS" | sed -n '1p')
RTC_CONFIG_TAG=$(printf '%s\n' "$RTC_DETAILS" | sed -n '2p')
echo "$RTC_SUMMARY"
echo "RTC result config directory: $RTC_CONFIG_TAG"

COMMAND=(bash "$HERE/start_robotwin_rtc_infer_and_eval.sh"
  --model_path "$LINGBOT_MODEL_PATH"
  --inference_script myscripts/simulation_rtc_tools/server/policy_server_rtc.py
  --inference_workdir "$ROOT/"
  --eval_workdir "$ROOT/RoboTwin_lingbot"
  --output_base "$ROOT/evaluation/video_comparison/$LINGBOT_MODEL_ID/$LINGBOT_PRECISION/rtc/${SETTING#demo_}/$RTC_CONFIG_TAG/$TASK_DIR"
  --conda_sh "$LINGBOT_CONDA_SH"
  --inference_env lingbotvla --sim_env RoboTwin
  --task_config "$SETTING" --num_tasks "$NUM_TASKS" --num_episodes "$EPISODES"
  --num_gpus 1 --num_per_gpu 1 --use_length 50
  --use_bf16 "$LINGBOT_USE_BF16" --use_fp32 "$LINGBOT_USE_FP32" --use_compile True
  --render_freq 0 --video_fps "$VIDEO_FPS" --video
  "${EXTRA[@]}")

if $DRY_RUN; then
  printf 'LINGBOT_MODEL_ASSET_ROOT=%q LINGBOT_NORM_STATS=%q LINGBOT_ROBOT_CONFIG=%q LINGBOT_TRAINING_CONFIG=%q ' \
    "$LINGBOT_MODEL_ASSET_ROOT" "$LINGBOT_NORM_STATS" "$LINGBOT_ROBOT_CONFIG" "$LINGBOT_TRAINING_CONFIG"
  for name in REQUEST_INDEX EXECUTION_HORIZON FIXED_DELAY_STEPS CROSSFADE_STEPS EMA_ALPHA; do
    var="LINGBOT_RTC_${name}"
    [[ -v "$var" ]] && printf '%s=%q ' "$var" "${!var}"
  done
  printf 'LINGBOT_RTC_EVAL_CLIENT=%q ' "$TOOL/copied_eval/eval_policy_client_lingbotvla_rtc.py"
  [[ -n "$GPU_IDS" ]] && printf 'LINGBOT_GPU_IDS=%q ' "$GPU_IDS"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

cd "$ROOT"
exec "${COMMAND[@]}"
