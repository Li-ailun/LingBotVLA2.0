#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <demo_clean|demo_randomized> <output-subdir> [options]" >&2
  exit 2
fi

task_config=$1
output_subdir=$2
shift 2
dry_run=false
visualize=false
task_name=""
num_episodes=50
resume_run=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) dry_run=true; shift ;;
    --visualize) visualize=true; shift ;;
    --task) task_name="$2"; shift 2 ;;
    --num-episodes) num_episodes="$2"; shift 2 ;;
    --resume-run) resume_run="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
case "$task_config" in
  demo_clean|demo_randomized) ;;
  *) echo "Unsupported task config: $task_config" >&2; exit 2 ;;
esac

workspace=/home/lixiang/PROJECT/LingBotVLA2.0
repo=${workspace}/lingbot-vla-v2
adapter=${workspace}/baseline_base_model_adapter
launcher=${repo}/experiment/robotwin/start_robotwin_infer_and_eval.sh
model=${adapter}/checkpoints/global_step_0/hf_ckpt
output=${adapter}/results/${output_subdir}

extra_args=()
if $visualize; then
  if [[ -z "$task_name" ]]; then
    echo "--visualize requires --task <task_name>" >&2
    exit 2
  fi
  extra_args+=(--task_name "$task_name" --render_freq 1)
  output=${adapter}/results/visualization_${task_config#demo_}
fi

if [[ -n "$resume_run" ]]; then
  if $visualize; then
    echo "--resume-run cannot be combined with --visualize" >&2
    exit 2
  fi
  if [[ ! -d "$resume_run/eval_logs" ]]; then
    echo "Resume run has no eval_logs directory: $resume_run" >&2
    exit 2
  fi
  latest_log=""
  progress=""
  while IFS= read -r candidate; do
    candidate_progress=$(sed -r 's/\x1B\[[0-9;]*[mK]//g' "$candidate" | grep -aoE 'Success rate: [0-9]+/[0-9]+ => [0-9.]+%, current seed: [0-9]+' | tail -1 || true)
    if [[ -n "$candidate_progress" ]]; then
      latest_log="$candidate"
      progress="$candidate_progress"
      break
    fi
  done < <(find "$resume_run/eval_logs" -maxdepth 1 -type f -name '*.log' -printf '%T@ %p\n' | sort -nr | cut -d' ' -f2-)
  if [[ -z "$latest_log" ]]; then
    echo "No completed episode checkpoint found in: $resume_run/eval_logs" >&2
    exit 2
  fi
  task_name=$(basename "$latest_log" .log)
  initial_success=$(sed -E 's/.*Success rate: ([0-9]+)\/.*/\1/' <<<"$progress")
  initial_episodes=$(sed -E 's/.*Success rate: [0-9]+\/([0-9]+).*/\1/' <<<"$progress")
  last_seed=$(sed -E 's/.*current seed: ([0-9]+).*/\1/' <<<"$progress")
  start_seed=$((last_seed + 1))
  extra_args+=(
    --resume_run_dir "$resume_run"
    --task_name "$task_name"
    --initial_episode_count "$initial_episodes"
    --initial_success_count "$initial_success"
    --start_seed "$start_seed"
  )
fi

num_tasks=50
if $visualize; then
  num_tasks=1
fi

command=(bash "$launcher"
  --model_path "$model"
  --inference_workdir "${repo}/"
  --eval_workdir "${workspace}/RoboTwin_lingbot"
  --output_base "$output"
  --conda_sh /home/lixiang/miniconda3/etc/profile.d/conda.sh
  --inference_env lingbotvla
  --sim_env RoboTwin
  --task_config "$task_config"
  --num_tasks "$num_tasks"
  --num_episodes "$num_episodes"
  --num_gpus 1
  --num_per_gpu 1
  --use_bf16 True
  --use_fp32 False
  --use_compile True
  --no_video
  "${extra_args[@]}")

if $dry_run; then
  printf 'QWEN3VL_PATH=%q ' "${workspace}/models/Qwen3-VL-4B-Instruct"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

cd "$repo"
export QWEN3VL_PATH="${workspace}/models/Qwen3-VL-4B-Instruct"
exec "${command[@]}"
