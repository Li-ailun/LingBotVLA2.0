#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/.." && pwd)"
TASK_SOURCE="$HERE/runtime_assets/robotwin_task_config"
ROBOTWIN_DIR="$ROOT/RoboTwin_lingbot"
ASSETS_SOURCE=""
CHECK_ONLY=0
TASK_CONFIG_ONLY=0
INSTALL_CUROBO=0
status=0

ASSET_URL="https://huggingface.co/datasets/TianxingChen/RoboTwin2.0"
CUROBO_REPO="https://github.com/NVlabs/curobo.git"
CUROBO_COMMIT="d64c4b005459db10c5dd867d8b30a87d5bda9bdb"

usage() {
  cat <<'EOF'
Usage: bash evaluation/prepare_runtime.sh [options]
  --robotwin-dir PATH   RoboTwin checkout (default: ./RoboTwin_lingbot)
  --assets-source PATH  Link objects/embodiments/background_texture from PATH
  --check-only          Validate without modifying the checkout
  --task-config-only    Install tracked task configs, then exit
  --install-curobo      Clone pinned CuRobo and install it into RoboTwin env
  -h, --help            Show this help
EOF
}

while (($#)); do
  case "$1" in
    --robotwin-dir) ROBOTWIN_DIR="${2:?--robotwin-dir requires a path}"; shift 2 ;;
    --assets-source) ASSETS_SOURCE="${2:?--assets-source requires a path}"; shift 2 ;;
    --check-only) CHECK_ONLY=1; shift ;;
    --task-config-only) TASK_CONFIG_ONLY=1; shift ;;
    --install-curobo) INSTALL_CUROBO=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -d "$ROBOTWIN_DIR" ]] || {
  echo "RoboTwin directory not found: $ROBOTWIN_DIR" >&2
  exit 2
}

install_task_configs() {
  local target="$ROBOTWIN_DIR/task_config"
  if ((CHECK_ONLY)); then
    local name
    for name in _camera_config.yml _config_template.yml _embodiment_config.yml \
      _eval_step_limit.yml create_task_config.sh demo_clean.yml demo_randomized.yml; do
      if [[ ! -f "$target/$name" ]]; then
        echo "Missing task config: $target/$name" >&2
        status=1
      fi
    done
    return
  fi
  mkdir -p "$target"
  cp -f "$TASK_SOURCE"/* "$target/"
  chmod +x "$target/create_task_config.sh"
  echo "TASK CONFIG READY: $target"
}

verify_task_config_hashes() {
  local target="$ROBOTWIN_DIR/task_config"
  local name expected actual
  while IFS=$'\t' read -r name expected; do
    actual="$(sha256sum "$TASK_SOURCE/$name" | awk '{print $1}')"
    if [[ "$actual" != "$expected" ]]; then
      echo "Tracked task config hash mismatch: $name" >&2
      status=1
    fi
    if [[ -f "$target/$name" ]]; then
      actual="$(sha256sum "$target/$name" | awk '{print $1}')"
      if [[ "$actual" != "$expected" ]]; then
        echo "Installed task config differs from manifest: $target/$name" >&2
        status=1
      fi
    fi
  done < <(python3 - "$HERE/runtime_assets/manifest.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    hashes = json.load(handle)["robotwin"]["task_config_sha256"]
for name, digest in hashes.items():
    print(f"{name}\t{digest}")
PY
  )
  if ((status == 0)); then
    echo "TASK CONFIG VALID: $target"
  fi
}

link_asset_group() {
  local group="$1"
  local source="$ASSETS_SOURCE/$group"
  local target="$ROBOTWIN_DIR/assets/$group"
  [[ -e "$source" ]] || { echo "Asset source missing: $source" >&2; return 1; }
  if [[ -e "$target" || -L "$target" ]]; then
    [[ "$(readlink -f "$target")" == "$(readlink -f "$source")" ]] || {
      echo "Refusing to replace existing asset path: $target" >&2
      return 1
    }
    return
  fi
  mkdir -p "$ROBOTWIN_DIR/assets"
  ln -s "$source" "$target"
}

install_task_configs
verify_task_config_hashes
if ((TASK_CONFIG_ONLY)); then
  exit "$status"
fi

if [[ -d "$ROBOTWIN_DIR/assets/assets" ]]; then
  echo "Invalid nested asset layout: $ROBOTWIN_DIR/assets/assets; contents must live directly under assets/." >&2
  status=1
fi

if [[ -n "$ASSETS_SOURCE" && $CHECK_ONLY -eq 0 ]]; then
  [[ ! -d "$ASSETS_SOURCE/assets" ]] || {
    echo "Invalid --assets-source: use the directory containing objects/, embodiments/ and background_texture/, not assets/assets." >&2
    exit 2
  }
  link_asset_group objects
  link_asset_group embodiments
  link_asset_group background_texture
fi

for required in \
  "$ROBOTWIN_DIR/assets/objects/objaverse/list.json" \
  "$ROBOTWIN_DIR/assets/embodiments/aloha-agilex" \
  "$ROBOTWIN_DIR/assets/background_texture"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing RoboTwin asset: $required" >&2
    status=1
  fi
done
if ((status)); then
  echo "Official RoboTwin 2.0 assets: $ASSET_URL" >&2
else
  echo "ASSET LAYOUT VALID: $ROBOTWIN_DIR/assets"
fi

CUROBO_DIR="$ROBOTWIN_DIR/envs/curobo"
if [[ ! -d "$CUROBO_DIR/.git" ]]; then
  if ((INSTALL_CUROBO)); then
    source "$ROOT/evaluation/runtime_paths.sh"
    resolve_conda_env_python RoboTwin ROBOTWIN_PYTHON
    mkdir -p "$ROBOTWIN_DIR/envs"
    git clone "$CUROBO_REPO" "$CUROBO_DIR"
    git -C "$CUROBO_DIR" checkout --detach "$CUROBO_COMMIT"
    "$ROBOTWIN_PYTHON" -m pip install -e "$CUROBO_DIR" --no-build-isolation
  else
    echo "Missing pinned CuRobo checkout: $CUROBO_DIR" >&2
    echo "Source: $CUROBO_REPO" >&2
    echo "Required commit: $CUROBO_COMMIT" >&2
    echo "Run again with --install-curobo after confirming CUDA/build prerequisites." >&2
    status=1
  fi
fi

if [[ -d "$CUROBO_DIR/.git" ]]; then
  actual_commit="$(git -C "$CUROBO_DIR" rev-parse HEAD)"
  if [[ "$actual_commit" != "$CUROBO_COMMIT" ]]; then
    echo "CuRobo commit mismatch: expected $CUROBO_COMMIT, got $actual_commit" >&2
    status=1
  else
    echo "CUROBO SOURCE VALID: $actual_commit"
  fi
fi

exit "$status"
