#!/usr/bin/env bash
# Shared model/precision resolver for synchronous and RTC evaluation wrappers.

resolve_lingbot_model() {
  local requested="${1:-step2000}"
  local root="${2:?project root is required}"
  case "$requested" in
    step2000|lingbot_action_step2000)
      LINGBOT_MODEL_ID=step2000
      LINGBOT_MODEL_PATH="$root/models/lingbot_action_step2000"
      LINGBOT_MODEL_ASSET_ROOT="$root/models/lingbot_action_step2000"
      LINGBOT_TRAINING_CONFIG="$LINGBOT_MODEL_ASSET_ROOT/training/robotwin_clean_head.yaml"
      LINGBOT_ROBOT_CONFIG="$LINGBOT_MODEL_ASSET_ROOT/configs/robot_configs/robotwin.yaml" ;;
    full5k|lingbot_full_5k_6_60)
      LINGBOT_MODEL_ID=full5k
      LINGBOT_MODEL_ASSET_ROOT="$root/models/lingbot_full_5k_6_60"
      LINGBOT_MODEL_PATH="$LINGBOT_MODEL_ASSET_ROOT/checkpoints/global_step_5000/hf_ckpt"
      LINGBOT_TRAINING_CONFIG="$LINGBOT_MODEL_ASSET_ROOT/lingbotvla_cli.yaml"
      LINGBOT_ROBOT_CONFIG="$LINGBOT_MODEL_ASSET_ROOT/configs/robot_configs_clean/robotwin.yaml" ;;
    full15k|15k|lingbot_15K)
      LINGBOT_MODEL_ID=full15k
      LINGBOT_MODEL_PATH="$root/models/lingbot_15K"
      # The 15k export contains only the continued HF weights. It uses the
      # unchanged full5k normalization, robot, and training configuration.
      LINGBOT_MODEL_ASSET_ROOT="$root/models/lingbot_full_5k_6_60"
      LINGBOT_TRAINING_CONFIG="$LINGBOT_MODEL_ASSET_ROOT/lingbotvla_cli.yaml"
      LINGBOT_ROBOT_CONFIG="$LINGBOT_MODEL_ASSET_ROOT/configs/robot_configs_clean/robotwin.yaml" ;;
    full30k|30k|lingbot_30K)
      LINGBOT_MODEL_ID=full30k
      local full30k_root="$root/models/lingbot_30K"
      local full30k_nested="$full30k_root/checkpoints/global_step_30000/hf_ckpt"
      if [[ -f "$full30k_root/config.json" && -f "$full30k_root/model.safetensors.index.json" ]]; then
        LINGBOT_MODEL_PATH="$full30k_root"
      elif [[ -f "$full30k_nested/config.json" && -f "$full30k_nested/model.safetensors.index.json" ]]; then
        LINGBOT_MODEL_PATH="$full30k_nested"
      else
        echo "30K model download is incomplete; checked both supported layouts:" >&2
        echo "  $full30k_root/config.json" >&2
        echo "  $full30k_root/model.safetensors.index.json" >&2
        echo "  $full30k_nested/config.json" >&2
        echo "  $full30k_nested/model.safetensors.index.json" >&2
        return 2
      fi
      local incomplete_shard
      incomplete_shard="$(compgen -G "$LINGBOT_MODEL_PATH/*.incomplete" | head -n 1 || true)"
      if [[ -n "$incomplete_shard" ]]; then
        echo "30K model download is incomplete: $incomplete_shard" >&2
        return 2
      fi
      local indexed_shards shard
      if ! indexed_shards="$(python3 -c '
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    index = json.load(stream)
for name in sorted(set(index["weight_map"].values())):
    print(name)
' "$LINGBOT_MODEL_PATH/model.safetensors.index.json")"; then
        echo "Invalid 30K model weight index: $LINGBOT_MODEL_PATH/model.safetensors.index.json" >&2
        return 2
      fi
      while IFS= read -r shard; do
        [[ -z "$shard" ]] && continue
        if [[ ! -f "$LINGBOT_MODEL_PATH/$shard" ]]; then
          echo "30K indexed weight shard is missing: $LINGBOT_MODEL_PATH/$shard" >&2
          return 2
        fi
      done <<< "$indexed_shards"
      # The 30k checkpoint continues the same full-weight training run as
      # full5k/full15k and therefore shares their normalization and configs.
      LINGBOT_MODEL_ASSET_ROOT="$root/models/lingbot_full_5k_6_60"
      LINGBOT_TRAINING_CONFIG="$LINGBOT_MODEL_ASSET_ROOT/lingbotvla_cli.yaml"
      LINGBOT_ROBOT_CONFIG="$LINGBOT_MODEL_ASSET_ROOT/configs/robot_configs_clean/robotwin.yaml" ;;
    *) echo "Unknown model '$requested' (expected: step2000, full5k, full15k, or full30k)" >&2; return 2 ;;
  esac
  LINGBOT_NORM_STATS="$LINGBOT_MODEL_ASSET_ROOT/assets/norm_stats/robotwin_clean.json"
  LINGBOT_MODELS_ROOT="$root/models"
  local required
  for required in "$LINGBOT_MODEL_PATH/model.safetensors.index.json" "$LINGBOT_MODEL_PATH/config.json" \
    "$LINGBOT_NORM_STATS" "$LINGBOT_ROBOT_CONFIG" "$LINGBOT_TRAINING_CONFIG"; do
    [[ -f "$required" ]] || { echo "Missing model adapter input: $required" >&2; return 2; }
  done
  export LINGBOT_MODEL_ID LINGBOT_MODEL_PATH LINGBOT_MODEL_ASSET_ROOT
  export LINGBOT_NORM_STATS LINGBOT_ROBOT_CONFIG LINGBOT_TRAINING_CONFIG LINGBOT_MODELS_ROOT
}

resolve_lingbot_precision() {
  case "${1:-bf16}" in
    bf16|bf) LINGBOT_PRECISION=bf16; LINGBOT_USE_BF16=True; LINGBOT_USE_FP32=False ;;
    fp32|fp) LINGBOT_PRECISION=fp32; LINGBOT_USE_BF16=False; LINGBOT_USE_FP32=True ;;
    *) echo "Unknown precision '$1' (expected: bf16 or fp32)" >&2; return 2 ;;
  esac
  export LINGBOT_PRECISION LINGBOT_USE_BF16 LINGBOT_USE_FP32
}
