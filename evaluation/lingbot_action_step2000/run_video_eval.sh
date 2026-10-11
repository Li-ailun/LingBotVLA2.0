#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../.." && pwd)"

SETTING="${1:?usage: $0 demo_clean|demo_randomized [--task TASK] [--episodes N] [--video-fps N] [--dry-run]}"
shift
TASK=""
EPISODES=50
VIDEO_FPS=10
DRY_RUN=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --video-fps) VIDEO_FPS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

case "$SETTING" in
  demo_clean|demo_randomized) ;;
  *) echo "setting must be demo_clean or demo_randomized" >&2; exit 2 ;;
esac
[[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || { echo "episodes must be a positive integer" >&2; exit 2; }
[[ "$VIDEO_FPS" =~ ^[1-9][0-9]*$ ]] || { echo "video-fps must be a positive integer" >&2; exit 2; }

NUM_TASKS=50
EXTRA=()
if [[ -n "$TASK" ]]; then
  NUM_TASKS=1
  EXTRA+=(--task_name "$TASK")
fi

unset ROBOTWIN_REALTIME_VISUALIZATION
export QWEN3VL_PATH="$ROOT/models/Qwen3-VL-4B-Instruct"
export PYTHONPATH="$ROOT:$ROOT/lingbot-vla-v2${PYTHONPATH:+:$PYTHONPATH}"

COMMAND=(bash "$HERE/video_eval_launcher/start_robotwin_video_infer_and_eval.sh"
  --model_path "$ROOT/models/lingbot_action_step2000"
  --inference_script evaluation/lingbot_action_step2000/policy_server.py
  --inference_workdir "$ROOT/lingbot-vla-v2/"
  --eval_workdir "$ROOT/RoboTwin_lingbot"
  --output_base "$ROOT/evaluation/video_comparison/non_rtc/${SETTING#demo_}"
  --conda_sh /home/lixiang/miniconda3/etc/profile.d/conda.sh
  --inference_env lingbotvla --sim_env RoboTwin
  --task_config "$SETTING" --num_tasks "$NUM_TASKS" --num_episodes "$EPISODES"
  --num_gpus 1 --num_per_gpu 1 --use_length 50
  --use_bf16 True --use_fp32 False --use_compile True
  --render_freq 0 --video_fps "$VIDEO_FPS" --video
  "${EXTRA[@]}")

if $DRY_RUN; then
  printf 'QWEN3VL_PATH=%q ' "$QWEN3VL_PATH"
  printf 'PYTHONPATH=%q ' "$PYTHONPATH"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

cd "$ROOT"
exec "${COMMAND[@]}"
