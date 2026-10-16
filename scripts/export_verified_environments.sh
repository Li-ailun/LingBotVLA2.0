#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/.." && pwd)"
DEST="${LINGBOT_ENV_EXPORT_DIR:-$ROOT/environment}"
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
LINGBOTVLA_ENV_PREFIX="${LINGBOTVLA_ENV_PREFIX:-/home/hddData/User/lixiang/conda-envs/lingbotvla}"
ROBOTWIN_ENV_PREFIX="${ROBOTWIN_ENV_PREFIX:-/home/hddData/User/lixiang/conda-envs/RoboTwin}"

[[ -n "$CONDA_BIN" && -x "$CONDA_BIN" ]] || {
  echo "[ERROR] Conda executable not found; set CONDA_EXE." >&2
  exit 2
}
for prefix in "$LINGBOTVLA_ENV_PREFIX" "$ROBOTWIN_ENV_PREFIX"; do
  [[ -x "$prefix/bin/python" ]] || {
    echo "[ERROR] Conda environment Python not found: $prefix/bin/python" >&2
    exit 2
  }
done

mkdir -p "$DEST/snapshots"
stage="$(mktemp -d "$DEST/.verified-export.XXXXXX")"
trap 'rm -rf -- "$stage"' EXIT
mkdir -p "$stage/snapshots"

export_one() {
  local name="$1" prefix="$2" raw="$stage/$name.raw.yml"
  "$CONDA_BIN" env export --prefix "$prefix" --no-builds > "$raw"
  "$CONDA_BIN" list --prefix "$prefix" --explicit > "$stage/snapshots/$name.conda-explicit.txt"
  "$prefix/bin/python" -m pip freeze --exclude-editable > "$stage/snapshots/$name.pip-freeze.txt"
  "$prefix/bin/python" "$ROOT/evaluation/environment_reproduction.py" sanitize-conda \
    --input "$raw" --output "$stage/$name.environment.yml" --name "$name" \
    --drop-editable
  for snapshot in \
    "$stage/$name.environment.yml" \
    "$stage/snapshots/$name.conda-explicit.txt" \
    "$stage/snapshots/$name.pip-freeze.txt"; do
    "$prefix/bin/python" "$ROOT/evaluation/environment_reproduction.py" \
      check-portable --input "$snapshot"
  done
}

export_one lingbotvla "$LINGBOTVLA_ENV_PREFIX"
export_one RoboTwin "$ROBOTWIN_ENV_PREFIX"

{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "git_commit=$(git -C "$ROOT" rev-parse HEAD)"
  echo "kernel=$(uname -srmo)"
  if [[ -f /etc/os-release ]]; then grep -E '^(NAME|VERSION|ID|VERSION_ID)=' /etc/os-release; fi
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
  for entry in "lingbotvla:$LINGBOTVLA_ENV_PREFIX" "RoboTwin:$ROBOTWIN_ENV_PREFIX"; do
    name="${entry%%:*}"; prefix="${entry#*:}"
    "$prefix/bin/python" - "$name" <<'PY'
import importlib, platform, sys
name = sys.argv[1]
print(f"[{name}]")
print("python=" + platform.python_version())
for package in ("torch", "torchvision", "transformers", "websockets", "sapien", "mplib", "pytorch3d", "curobo"):
    try:
        module = importlib.import_module(package)
    except Exception as exc:
        print(f"{package}=UNAVAILABLE:{type(exc).__name__}")
    else:
        print(f"{package}={getattr(module, '__version__', 'installed')}")
if name == "RoboTwin":
    import torch
    print(f"torch_cuda={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
PY
  done
} > "$stage/snapshots/server-runtime.txt"

for file in \
  lingbotvla.environment.yml RoboTwin.environment.yml \
  snapshots/lingbotvla.conda-explicit.txt snapshots/lingbotvla.pip-freeze.txt \
  snapshots/RoboTwin.conda-explicit.txt snapshots/RoboTwin.pip-freeze.txt \
  snapshots/server-runtime.txt; do
  mv -f -- "$stage/$file" "$DEST/$file"
done
echo "[OK] Verified environment artifacts exported to: $DEST"
