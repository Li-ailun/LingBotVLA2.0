#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TOOL="$(cd -- "$HERE/.." && pwd)"
ROOT="$(cd -- "$TOOL/../.." && pwd)"
source "$ROOT/evaluation/runtime_paths.sh"
source "$ROOT/evaluation/model_registry.sh"

MODEL="step2000"
DEEP=0
while (($#)); do
  case "$1" in
    --model) MODEL="${2:?--model requires step2000, full5k, full15k, or full30k}"; shift 2 ;;
    --deep) DEEP=1; shift ;;
    -h|--help)
      echo "Usage: bash preflight.sh [--model step2000|full5k|full15k|full30k] [--deep]"
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

resolve_lingbot_runtime_paths
CONFIG_PY="${LINGBOT_RTC_CONFIG_PY:-$ROBOTWIN_PYTHON}"

resolve_lingbot_model "$MODEL" "$ROOT"
echo "MODEL ADAPTER VALID: $LINGBOT_MODEL_ID ($LINGBOT_MODEL_PATH)"

test -f "$TOOL/copied_eval/eval_policy_client_lingbotvla_rtc.py"
test -f "$ROOT/evaluation/lingbot_action_step2000/copied_eval/eval_policy_client_lingbotvla_recording.py"
test -f "$TOOL/server/policy_server_rtc.py"
test -x "$LINGBOTVLA_PYTHON"
test -x "$ROBOTWIN_PYTHON"

bash "$ROOT/evaluation/prepare_runtime.sh" --check-only

mkdir -p /tmp/lingbot-matplotlib
ROBOTWIN_DIR="$ROOT/RoboTwin_lingbot" DEEP_CHECK="$DEEP" MPLCONFIGDIR=/tmp/lingbot-matplotlib "$ROBOTWIN_PYTHON" - <<'PY'
import importlib
import os
from pathlib import Path

robotwin = Path(os.environ["ROBOTWIN_DIR"]).resolve()
modules = {}
for name in ("torch", "sapien", "mplib", "curobo", "pytorch3d"):
    modules[name] = importlib.import_module(name)

torch = modules["torch"]
cuda_available = torch.cuda.is_available()
if not cuda_available and os.environ.get("DEEP_CHECK") == "1":
    raise RuntimeError("torch.cuda.is_available() is false; --deep requires an accessible GPU")

curobo_path = Path(modules["curobo"].__file__).resolve()
expected_curobo = (robotwin / "envs/curobo").resolve()
if expected_curobo not in curobo_path.parents:
    raise RuntimeError(
        f"curobo imports from {curobo_path}, expected editable source under {expected_curobo}"
    )

if os.environ.get("DEEP_CHECK") == "1":
    from pytorch3d.ops import sample_farthest_points
    points = torch.zeros((1, 8, 3), device="cuda")
    sample_farthest_points(points, K=4)

print(
    "RUNTIME IMPORTS VALID: "
    f"torch={torch.__version__} torch_cuda={torch.version.cuda} "
    f"cuda_available={cuda_available} "
    f"gpu={torch.cuda.get_device_name(0) if cuda_available else 'not-visible'} "
    f"curobo={curobo_path}"
)
PY

PYTHONPATH="$TOOL:$ROOT" "$CONFIG_PY" -c "from pathlib import Path; from simulation_rtc.config import RTCConfig, RobotActionLayout; c=RTCConfig.from_yaml(Path('$TOOL/configs/lingbot_action_step2000_rtc_fast.yaml')); l=RobotActionLayout.from_yaml(Path('$LINGBOT_ROBOT_CONFIG')); assert (c.request_index,c.fixed_delay_steps,l.action_dim,l.gripper_indices)==(10,10,14,(6,13)); print('RTC CONFIG VALID')"
python3 -m py_compile "$TOOL"/simulation_rtc/*.py "$TOOL/server/policy_server_rtc.py" "$TOOL/copied_eval/eval_policy_client_lingbotvla_rtc.py" "$ROOT/evaluation/lingbot_action_step2000/copied_eval/eval_policy_client_lingbotvla_recording.py"

printf '%s  %s\n' \
  2fb48570fab1a0c5d7667215966bcd468cd24d7ad556a92303e68cc88f87ee8e "$ROOT/RoboTwin_lingbot/script/eval_policy_client_lingbotvla.py" \
  e1acfb9f710020211a59897fbdba6fcdd8eccea1f90abe93e6dcef60f7cc4f14 "$ROOT/evaluation/lingbot_action_step2000/policy_server.py" \
  2fb48570fab1a0c5d7667215966bcd468cd24d7ad556a92303e68cc88f87ee8e "$ROOT/lingbot-vla-v2/experiment/robotwin/eval_policy_client_lingbotvla.py" | sha256sum -c -

bash "$HERE/run_rtc_fast_one_task.sh" lift_pot --dry-run >/dev/null
bash "$HERE/run_rtc_fast_50x50.sh" --dry-run >/dev/null
echo "RTC PREFLIGHT OK: no simulator episodes were started"
