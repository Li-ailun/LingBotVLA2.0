#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"

exec /usr/bin/python3 "$HERE/r1pro_ros_client_node.py" \
  --rate-hz 15 --async-prefetch-steps 16 --requery-steps 20 \
  --rtc-execution-horizon 10 --rtc-max-guidance-weight 10 \
  --rtc-prefix-attention-schedule exp --rtc-initial-inference-delay-steps 3 \
  --rtc-max-inference-delay-steps 4 --smooth-transition-steps 4 \
  --publish-smooth-alpha 0.36 --enable-client-smoothing \
  --enable-gripper-stabilizer --gripper-debounce-steps 7 \
  --gripper-reversal-lock-steps 7 --gripper-low-threshold 20 \
  --gripper-high-threshold 60 "$@"
