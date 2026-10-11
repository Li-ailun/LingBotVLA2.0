#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK="${1:-lift_pot}"
EPISODES="${2:-1}"

export ROBOTWIN_REALTIME_VISUALIZATION=1
export PYTHONPATH="$HERE/realtime_live${PYTHONPATH:+:$PYTHONPATH}"

echo "Live 1x visualization: task=$TASK episodes=$EPISODES physics=250Hz viewer=30FPS"
exec bash "$HERE/run_clean_50x50.sh" \
  --visualize --task "$TASK" --num-episodes "$EPISODES"
