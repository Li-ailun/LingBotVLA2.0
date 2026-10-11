#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../.." && pwd)"
SETTING="${1:?demo_clean or demo_randomized required}"
shift

TASK=""
EPISODES=100
REALTIME=false
DRY_RUN=false
MODEL_NAME=step2000
PRECISION=bf16
RESUME_RUN_DIR=""
RESUME_TASK=""
INITIAL_EPISODE_COUNT=0
INITIAL_SUCCESS_COUNT=0
START_SEED=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --model) MODEL_NAME="$2"; shift 2 ;;
    --precision) PRECISION="$2"; shift 2 ;;
    --resume-run-dir) RESUME_RUN_DIR="$2"; shift 2 ;;
    --resume-task) RESUME_TASK="$2"; shift 2 ;;
    --initial-episode-count) INITIAL_EPISODE_COUNT="$2"; shift 2 ;;
    --initial-success-count) INITIAL_SUCCESS_COUNT="$2"; shift 2 ;;
    --start-seed) START_SEED="$2"; shift 2 ;;
    --realtime) REALTIME=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

case "$SETTING" in demo_clean|demo_randomized) ;; *) exit 2 ;; esac
source "$ROOT/evaluation/model_registry.sh"
resolve_lingbot_model "$MODEL_NAME" "$ROOT"
resolve_lingbot_precision "$PRECISION"
MODEL="$LINGBOT_MODEL_PATH"

NUM_TASKS=50
EXTRA=()
OUTPUT="$HERE/results/$LINGBOT_MODEL_ID/$LINGBOT_PRECISION/${SETTING#demo_}"
if [[ -n "$TASK" ]]; then
  NUM_TASKS=1
  EXTRA+=(--task_name "$TASK")
fi
if $REALTIME; then
  [[ -n "$TASK" ]] || { echo "--realtime requires --task" >&2; exit 2; }
  EXTRA+=(--render_freq 1)
  OUTPUT="$HERE/results/$LINGBOT_MODEL_ID/$LINGBOT_PRECISION/realtime_${SETTING#demo_}"
  export ROBOTWIN_REALTIME_VISUALIZATION=1
  export PYTHONPATH="$HERE/realtime_live:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
fi
if [[ -n "$RESUME_RUN_DIR" ]]; then
  EXTRA+=(--resume_run_dir "$RESUME_RUN_DIR")
  [[ -n "$RESUME_TASK" ]] && EXTRA+=(--task_name "$RESUME_TASK")
  EXTRA+=(--initial_episode_count "$INITIAL_EPISODE_COUNT")
  EXTRA+=(--initial_success_count "$INITIAL_SUCCESS_COUNT")
  [[ -n "$START_SEED" ]] && EXTRA+=(--start_seed "$START_SEED")
fi

COMMAND=(bash "$ROOT/lingbot-vla-v2/experiment/robotwin/start_robotwin_infer_and_eval.sh"
  --model_path "$MODEL"
  --inference_script evaluation/lingbot_action_step2000/policy_server.py
  --inference_workdir "$ROOT/lingbot-vla-v2/"
  --eval_workdir "$ROOT/RoboTwin_lingbot"
  --output_base "$OUTPUT"
  --conda_sh /home/lixiang/miniconda3/etc/profile.d/conda.sh
  --inference_env lingbotvla --sim_env RoboTwin
  --task_config "$SETTING" --num_tasks "$NUM_TASKS" --num_episodes "$EPISODES"
  --num_gpus 1 --num_per_gpu 1 --use_length 50
  --use_bf16 "$LINGBOT_USE_BF16" --use_fp32 "$LINGBOT_USE_FP32" --use_compile True --no_video
  "${EXTRA[@]}")

export QWEN3VL_PATH="$ROOT/models/Qwen3-VL-4B-Instruct"
if $DRY_RUN; then
  printf 'LINGBOT_MODEL_ASSET_ROOT=%q LINGBOT_NORM_STATS=%q LINGBOT_ROBOT_CONFIG=%q LINGBOT_TRAINING_CONFIG=%q ' \
    "$LINGBOT_MODEL_ASSET_ROOT" "$LINGBOT_NORM_STATS" "$LINGBOT_ROBOT_CONFIG" "$LINGBOT_TRAINING_CONFIG"
  $REALTIME && printf 'ROBOTWIN_REALTIME_VISUALIZATION=1 '
  printf 'QWEN3VL_PATH=%q ' "$QWEN3VL_PATH"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi
cd "$ROOT"
exec "${COMMAND[@]}"
