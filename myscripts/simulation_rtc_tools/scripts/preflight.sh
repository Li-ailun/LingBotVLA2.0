#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TOOL="$(cd -- "$HERE/.." && pwd)"
ROOT="$(cd -- "$TOOL/../.." && pwd)"
PYTEST_PY=/home/lixiang/miniconda3/envs/point-flow-sim/bin/python

test -f "$ROOT/models/lingbot_action_step2000/model.safetensors.index.json"
test -f "$ROOT/models/lingbot_action_step2000/configs/robot_configs/robotwin.yaml"
test -f "$TOOL/copied_eval/eval_policy_client_lingbotvla_rtc.py"
test -f "$ROOT/evaluation/lingbot_action_step2000/copied_eval/eval_policy_client_lingbotvla_recording.py"
test -f "$TOOL/server/policy_server_rtc.py"
test -x /home/lixiang/miniconda3/envs/lingbotvla/bin/python
test -x /home/lixiang/miniconda3/envs/RoboTwin/bin/python

PYTHONPATH="$TOOL:$ROOT" "$PYTEST_PY" -c "from pathlib import Path; from simulation_rtc.config import RTCConfig, RobotActionLayout; c=RTCConfig.from_yaml(Path('$TOOL/configs/lingbot_action_step2000_rtc_fast.yaml')); l=RobotActionLayout.from_yaml(Path('$ROOT/models/lingbot_action_step2000/configs/robot_configs/robotwin.yaml')); assert (c.request_index,c.fixed_delay_steps,l.action_dim,l.gripper_indices)==(10,10,14,(6,13)); print('RTC CONFIG VALID')"
python3 -m py_compile "$TOOL"/simulation_rtc/*.py "$TOOL/server/policy_server_rtc.py" "$TOOL/copied_eval/eval_policy_client_lingbotvla_rtc.py" "$ROOT/evaluation/lingbot_action_step2000/copied_eval/eval_policy_client_lingbotvla_recording.py"

printf '%s  %s\n' \
  2fb48570fab1a0c5d7667215966bcd468cd24d7ad556a92303e68cc88f87ee8e "$ROOT/RoboTwin_lingbot/script/eval_policy_client_lingbotvla.py" \
  ff74c9726e722416c9c7a57d88f7be855e848b4258f8828108e8c64f8e38b032 "$ROOT/evaluation/lingbot_action_step2000/policy_server.py" \
  2fb48570fab1a0c5d7667215966bcd468cd24d7ad556a92303e68cc88f87ee8e "$ROOT/lingbot-vla-v2/experiment/robotwin/eval_policy_client_lingbotvla.py" | sha256sum -c -

bash "$HERE/run_rtc_fast_one_task.sh" lift_pot --dry-run >/dev/null
bash "$HERE/run_rtc_fast_50x50.sh" --dry-run >/dev/null
echo "RTC PREFLIGHT OK: no simulator episodes were started"
