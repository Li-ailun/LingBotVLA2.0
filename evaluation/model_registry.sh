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
    *) echo "Unknown model '$requested' (expected: step2000 or full5k)" >&2; return 2 ;;
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
