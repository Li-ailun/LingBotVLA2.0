#!/usr/bin/env bash
set -euo pipefail

workspace=/home/lixiang/PROJECT/LingBotVLA2.0
adapter=${workspace}/baseline_base_model_adapter
repo=${workspace}/lingbot-vla-v2
py=/home/lixiang/miniconda3/envs/lingbotvla/bin/python

"$py" "${adapter}/tools/validate_adapter.py" --adapter "$adapter"
"$py" "${adapter}/tools/smoke_load.py" --adapter "$adapter" --check-only
bash -n "${repo}/experiment/robotwin/start_robotwin_infer_and_eval.sh"
bash "${repo}/experiment/robotwin/start_robotwin_infer_and_eval.sh" --validate_only --num_tasks 50 --num_episodes 50

test -d "${workspace}/RoboTwin_lingbot/assets"
test -f "${workspace}/RoboTwin_lingbot/task_config/demo_clean.yml"
test -f "${workspace}/RoboTwin_lingbot/task_config/demo_randomized.yml"
test -f /home/lixiang/miniconda3/etc/profile.d/conda.sh

source /home/lixiang/miniconda3/etc/profile.d/conda.sh
conda env list | grep -qE '^lingbotvla[[:space:]]'
conda env list | grep -qE '^RoboTwin[[:space:]]'
(
  cd "${workspace}/RoboTwin_lingbot"
  PYTHONNOUSERSITE=1 conda run -n RoboTwin python -c \
    'import websockets; import websockets.sync.client; import script.deploy.websocket_client_policy; print("RoboTwin websockets", websockets.__version__)'
)
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader

echo "PREFLIGHT OK: no simulator episodes were started"
