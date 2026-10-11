#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../.." && pwd)"
MODEL="$ROOT/models/lingbot_action_step2000"

for path in \
  "$MODEL/model.safetensors.index.json" \
  "$MODEL/model-00001-of-00003.safetensors" \
  "$MODEL/model-00002-of-00003.safetensors" \
  "$MODEL/model-00003-of-00003.safetensors" \
  "$MODEL/training/robotwin_clean_head.yaml" \
  "$MODEL/assets/norm_stats/robotwin_clean.json" \
  "$MODEL/configs/robot_configs/robotwin.yaml" \
  "$ROOT/models/Qwen3-VL-4B-Instruct/config.json"; do
  [[ -f "$path" ]] || { echo "MISSING: $path" >&2; exit 1; }
done

ROOT="$ROOT" /usr/bin/python3 - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ["ROOT"])
model = root / "models/lingbot_action_step2000"
index = json.loads((model / "model.safetensors.index.json").read_text())
shards = set(index["weight_map"].values())
assert shards == {f"model-0000{i}-of-00003.safetensors" for i in range(1, 4)}, shards
norm = json.loads((model / "assets/norm_stats/robotwin_clean.json").read_text())
required = {"action.arm.position", "action.effector.position", "observation.state.arm.position", "observation.state.effector.position"}
assert required <= set(norm["norm_stats"]), set(norm["norm_stats"])
print(f"CHECKPOINT OK: {len(index['weight_map'])} tensors, 3 shards")
print(f"NORM STATS OK: count={norm.get('count')}")
PY

bash "$HERE/run_formal_clean.sh" --dry-run >/dev/null
bash "$HERE/run_realtime_visualization.sh" lift_pot 1 --dry-run >/dev/null
echo "PREFLIGHT OK: no model server or simulator episode was started"
