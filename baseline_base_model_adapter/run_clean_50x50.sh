#!/usr/bin/env bash
set -euo pipefail
module_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
exec bash "${module_dir}/run_eval.sh" demo_clean zero_shot_clean "$@"
