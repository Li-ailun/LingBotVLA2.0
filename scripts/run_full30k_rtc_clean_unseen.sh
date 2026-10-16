#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/.." && pwd)"
GPU_IDS=""
BASE_PORT=9330
EPISODES=50
MIN_FREE_MIB=55000
DRY_RUN=0
LINGBOT_PREFIX=""
ROBOTWIN_PREFIX=""

usage() {
  cat <<'EOF'
Usage: bash scripts/run_full30k_rtc_clean_unseen.sh --gpu-ids CSV [options]
  --gpu-ids CSV          Physical GPU IDs, for example 1,2,3,5 (required)
  --base-port PORT       First inference port (default: 9330)
  --episodes N           Episodes per task (default: 50)
  --min-free-mib MIB     FP32 free-memory gate per GPU (default: 55000)
  --lingbot-prefix PATH  lingbotvla Conda environment prefix
  --robotwin-prefix PATH RoboTwin Conda environment prefix
  --dry-run              Print and validate the launcher command only
EOF
}
while (($#)); do
  case "$1" in
    --gpu-ids) GPU_IDS="${2:?--gpu-ids requires a value}"; shift 2 ;;
    --base-port) BASE_PORT="${2:?--base-port requires a value}"; shift 2 ;;
    --episodes) EPISODES="${2:?--episodes requires a value}"; shift 2 ;;
    --min-free-mib) MIN_FREE_MIB="${2:?--min-free-mib requires a value}"; shift 2 ;;
    --lingbot-prefix) LINGBOT_PREFIX="${2:?--lingbot-prefix requires a path}"; shift 2 ;;
    --robotwin-prefix) ROBOTWIN_PREFIX="${2:?--robotwin-prefix requires a path}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "$GPU_IDS" ]] || { echo "[ERROR] --gpu-ids is required." >&2; exit 2; }
[[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] --episodes must be positive." >&2; exit 2; }

if [[ -n "$LINGBOT_PREFIX" ]]; then export LINGBOTVLA_PYTHON="$LINGBOT_PREFIX/bin/python"; fi
if [[ -n "$ROBOTWIN_PREFIX" ]]; then export ROBOTWIN_PYTHON="$ROBOTWIN_PREFIX/bin/python"; fi
unset LD_LIBRARY_PATH
unset ROBOTWIN_REALTIME_VISUALIZATION

launcher=(
  bash "$ROOT/myscripts/simulation_rtc_tools/scripts/run_rtc_fast_50x50.sh"
  --setting demo_clean
  --base-port "$BASE_PORT"
  --model full30k
  --precision fp32
  --episodes "$EPISODES"
  --gpu-ids "$GPU_IDS"
  --instruction-type unseen
  --request-index 20
  --fixed-delay-steps 10
  --execution-horizon 20
  --crossfade-steps 10
  --ema-alpha 0.10
)
((DRY_RUN)) && launcher+=(--dry-run)

printf '[CANONICAL]'
printf ' %s' "${launcher[@]:2}"
printf '\n'

if ((DRY_RUN == 0)); then
  bash "$ROOT/scripts/verify_full30k_rtc_environment.sh" \
    --gpu-ids "$GPU_IDS" --base-port "$BASE_PORT" --min-free-mib "$MIN_FREE_MIB"
fi
exec "${launcher[@]}"
