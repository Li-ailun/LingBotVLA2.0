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
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --realtime) REALTIME=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

case "$SETTING" in demo_clean|demo_randomized) ;; *) exit 2 ;; esac
MODEL="$ROOT/models/lingbot_action_step2000"
[[ -f "$MODEL/model.safetensors.index.json" ]] || { echo "Missing checkpoint: $MODEL" >&2; exit 2; }

NUM_TASKS=50
EXTRA=()
OUTPUT="$HERE/results/${SETTING#demo_}"
if [[ -n "$TASK" ]]; then
  NUM_TASKS=1
  EXTRA+=(--task_name "$TASK")
fi
if $REALTIME; then
  [[ -n "$TASK" ]] || { echo "--realtime requires --task" >&2; exit 2; }
  EXTRA+=(--render_freq 1)
  OUTPUT="$HERE/results/realtime_${SETTING#demo_}"
  export ROBOTWIN_REALTIME_VISUALIZATION=1
  export PYTHONPATH="$HERE/realtime_live:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
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
  --use_bf16 True --use_fp32 False --use_compile True --no_video
  "${EXTRA[@]}")

export QWEN3VL_PATH="$ROOT/models/Qwen3-VL-4B-Instruct"
if $DRY_RUN; then
  $REALTIME && printf 'ROBOTWIN_REALTIME_VISUALIZATION=1 '
  printf 'QWEN3VL_PATH=%q ' "$QWEN3VL_PATH"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi
cd "$ROOT"
exec "${COMMAND[@]}"
