from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "convert_tools" / "build_lerobot_from_r1pro_intermediate.py"
PYTHON = "/home/lixiang/PROJECT/openpi/.venv/bin/python"


def test_builder_declares_left_first_16d_features() -> None:
    code = (
        "import importlib.util,json;"
        f"s=importlib.util.spec_from_file_location('builder',{str(SCRIPT)!r});"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "f=m.make_features();"
        "print(json.dumps({'state':f['observation.state'],'action':f['action']}))"
    )
    result = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    features = json.loads(result.stdout)
    assert features["state"]["shape"] == [16]
    assert features["action"]["shape"] == [16]
    assert features["state"]["names"][0] == "left_arm_j0"
    assert features["state"]["names"][7] == "left_gripper"
    assert features["state"]["names"][8] == "right_arm_j0"
    assert features["state"]["names"][15] == "right_gripper"
    assert not any("torso" in name for name in features["state"]["names"])


def test_builder_rejects_legacy_20d_intermediate() -> None:
    code = (
        "import importlib.util,numpy as np;"
        f"s=importlib.util.spec_from_file_location('builder',{str(SCRIPT)!r});"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "m.validate_vector_array(np.zeros((2,20)),name='state',episode='episode_legacy')"
    )
    result = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True)
    assert result.returncode != 0
    assert "expected final dimension 16, got 20" in result.stderr
