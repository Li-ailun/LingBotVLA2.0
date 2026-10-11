#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK="${1:-lift_pot}"
EPISODES="${2:-1}"
shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))
exec bash "$HERE/run_eval.sh" demo_clean --task "$TASK" --episodes "$EPISODES" --realtime "$@"
