from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "convert_tools" / "export_r1pro_mcap_to_intermediate.py"


def test_exporter_builds_left_first_16d_state_and_action() -> None:
    code = f"""
import importlib.util,json,numpy as np
s=importlib.util.spec_from_file_location('exporter',{str(SCRIPT)!r})
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class One:
    def __init__(self,v): self.v=v
    def nearest(self,t,d): return self.v
series={{
 'state_left_arm':One(np.arange(1,8)), 'state_left_gripper':One(8),
 'state_right_arm':One(np.arange(9,16)), 'state_right_gripper':One(16),
 'action_left_arm':One(np.arange(101,108)), 'action_left_gripper':One(108),
 'action_right_arm':One(np.arange(109,116)), 'action_right_gripper':One(116),
}}
print(json.dumps({{'state':m.build_state(series,0,1).tolist(),'action':m.build_action(series,0,1).tolist(),'topics':dict(m.TOPICS)}}))
"""
    result = subprocess.run(["/usr/bin/python3", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["state"] == [float(i) for i in range(1, 17)]
    assert payload["action"] == [float(i) for i in range(101, 117)]
    assert not any("torso" in value for value in payload["topics"].values())


def test_exporter_returns_none_when_required_arm_is_missing() -> None:
    code = f"""
import importlib.util,numpy as np
s=importlib.util.spec_from_file_location('exporter',{str(SCRIPT)!r})
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class One:
    def __init__(self,v): self.v=v
    def nearest(self,t,d): return self.v
series={{'state_left_arm':One(None),'state_left_gripper':One(1),'state_right_arm':One(np.zeros(7)),'state_right_gripper':One(1)}}
print(m.build_state(series,0,1))
"""
    result = subprocess.run(["/usr/bin/python3", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "None"
