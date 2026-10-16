#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/.." && pwd)"
ENVIRONMENT_DIR="$ROOT/environment"
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
DRY_RUN=0
UPDATE=0
LINGBOT_PREFIX=""
ROBOTWIN_PREFIX=""

usage() {
  cat <<'EOF'
Usage: bash scripts/create_eval_environments.sh [options]
  --conda PATH             Conda executable
  --environment-dir PATH   Exported definitions (default: ./environment)
  --lingbot-prefix PATH    Target lingbotvla environment prefix
  --robotwin-prefix PATH   Target RoboTwin environment prefix
  --update                 Update existing prefixes instead of refusing
  --dry-run                Print commands without executing them
EOF
}

while (($#)); do
  case "$1" in
    --conda) CONDA_BIN="${2:?--conda requires a path}"; shift 2 ;;
    --environment-dir) ENVIRONMENT_DIR="${2:?--environment-dir requires a path}"; shift 2 ;;
    --lingbot-prefix) LINGBOT_PREFIX="${2:?--lingbot-prefix requires a path}"; shift 2 ;;
    --robotwin-prefix) ROBOTWIN_PREFIX="${2:?--robotwin-prefix requires a path}"; shift 2 ;;
    --update) UPDATE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$CONDA_BIN" && -x "$CONDA_BIN" ]] || {
  echo "[ERROR] Conda executable not found; pass --conda PATH." >&2
  exit 2
}
if [[ -z "$LINGBOT_PREFIX" || -z "$ROBOTWIN_PREFIX" ]]; then
  env_root="${CONDA_ENVS_PATH%%:*}"
  if [[ -z "$env_root" ]]; then
    env_root="$($CONDA_BIN info --base)/envs"
  fi
  LINGBOT_PREFIX="${LINGBOT_PREFIX:-$env_root/lingbotvla}"
  ROBOTWIN_PREFIX="${ROBOTWIN_PREFIX:-$env_root/RoboTwin}"
fi

plan_args=(
  "$ROOT/evaluation/environment_reproduction.py" plan-create
  --root "$ROOT" --conda "$CONDA_BIN" --environment-dir "$ENVIRONMENT_DIR"
  --lingbot-prefix "$LINGBOT_PREFIX" --robotwin-prefix "$ROBOTWIN_PREFIX"
)
((UPDATE)) && plan_args+=(--update)
mapfile -t commands < <(python3 "${plan_args[@]}")
((${#commands[@]} > 0)) || { echo "[ERROR] Empty creation plan." >&2; exit 2; }

for command in "${commands[@]}"; do
  echo "+ $command"
  if ((DRY_RUN == 0)); then
    eval "$command"
  fi
done

if ((DRY_RUN)); then
  echo "[DRY-RUN] No environments were created or updated."
  exit 0
fi

export LINGBOTVLA_PYTHON="$LINGBOT_PREFIX/bin/python"
export ROBOTWIN_PYTHON="$ROBOTWIN_PREFIX/bin/python"
bash "$ROOT/scripts/verify_full30k_rtc_environment.sh"
