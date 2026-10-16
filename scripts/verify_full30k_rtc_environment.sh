#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/.." && pwd)"
GPU_IDS="${LINGBOT_GPU_IDS:-0}"
BASE_PORT="${LINGBOT_BASE_PORT:-9330}"
MIN_FREE_MIB="${LINGBOT_FP32_MIN_FREE_MIB:-55000}"

usage() {
  cat <<'EOF'
Usage: bash scripts/verify_full30k_rtc_environment.sh [options]
  --gpu-ids CSV         Physical GPUs to validate (default: 0)
  --base-port PORT      First inference port (default: 9330)
  --min-free-mib MIB    Required free memory per GPU for FP32 (default: 55000)
EOF
}
while (($#)); do
  case "$1" in
    --gpu-ids) GPU_IDS="${2:?--gpu-ids requires a value}"; shift 2 ;;
    --base-port) BASE_PORT="${2:?--base-port requires a value}"; shift 2 ;;
    --min-free-mib) MIN_FREE_MIB="${2:?--min-free-mib requires a value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ":${LD_LIBRARY_PATH-}:" == *":/usr/local/cuda-12.0/lib64:"* ]]; then
  echo "[ERROR] LD_LIBRARY_PATH contains /usr/local/cuda-12.0/lib64; unset it before using PyTorch CUDA 12.1." >&2
  exit 2
fi
LINGBOTVLA_PYTHON="${LINGBOTVLA_PYTHON:?set LINGBOTVLA_PYTHON to the lingbotvla interpreter}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:?set ROBOTWIN_PYTHON to the RoboTwin interpreter}"
[[ -x "$LINGBOTVLA_PYTHON" ]] || { echo "[ERROR] Not executable: $LINGBOTVLA_PYTHON" >&2; exit 2; }
[[ -x "$ROBOTWIN_PYTHON" ]] || { echo "[ERROR] Not executable: $ROBOTWIN_PYTHON" >&2; exit 2; }

plan="$(python3 "$ROOT/evaluation/environment_reproduction.py" runtime-plan \
  --gpu-ids "$GPU_IDS" --base-port "$BASE_PORT" --min-free-mib "$MIN_FREE_MIB")"
python3 "$ROOT/evaluation/environment_reproduction.py" validate-assets \
  --root "$ROOT" --manifest "$ROOT/environment/assets-full30k.json"

mapfile -t gpu_ids < <(python3 -c 'import json,sys; print(*json.loads(sys.argv[1])["gpu_ids"], sep="\n")' "$plan")
mapfile -t ports < <(python3 -c 'import json,sys; print(*json.loads(sys.argv[1])["ports"], sep="\n")' "$plan")
for gpu in "${gpu_ids[@]}"; do
  free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
  [[ "$free" =~ ^[0-9]+$ ]] || { echo "[ERROR] Cannot read GPU $gpu free memory." >&2; exit 2; }
  ((free >= MIN_FREE_MIB)) || {
    echo "[ERROR] GPU $gpu has ${free} MiB free; FP32 threshold is ${MIN_FREE_MIB} MiB." >&2
    exit 2
  }
  echo "GPU $gpu MEMORY VALID: ${free} MiB free"
done
for port in "${ports[@]}"; do
  if ss -lnt | awk '{print $4}' | grep -Eq ":${port}$"; then
    echo "[ERROR] Inference port is already in use: $port" >&2
    exit 2
  fi
done

"$LINGBOTVLA_PYTHON" -c 'import torch, transformers, websockets; assert torch.cuda.is_available(); print("LINGBOTVLA IMPORTS VALID", torch.__version__, torch.version.cuda)'
"$ROBOTWIN_PYTHON" -c 'import torch, sapien, mplib, curobo, pytorch3d; assert torch.cuda.is_available(); print("ROBOTWIN IMPORTS VALID", torch.__version__, torch.version.cuda)'
bash "$ROOT/evaluation/prepare_runtime.sh" --check-only
bash "$ROOT/myscripts/simulation_rtc_tools/scripts/preflight.sh" --model full30k --deep
echo "FULL30K FP32 RTC ENVIRONMENT VALID: GPUs=$GPU_IDS ports=${ports[*]}"
