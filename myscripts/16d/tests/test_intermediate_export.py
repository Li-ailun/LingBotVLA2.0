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


def test_exporter_preserves_instruction_metadata(tmp_path: Path) -> None:
    episode = tmp_path / "episode_000001"
    episode.mkdir()
    instruction = (
        "Stack the blue plate and then the pink plate on the left. "
        "Stack the blue, red, and yellow cups on the right. "
        "Finally, place the cup stack on the plates."
    )
    (episode / "task.txt").write_text(instruction + "\n", encoding="utf-8")
    (episode / "record_config.json").write_text(json.dumps({
        "task_id": "stack_plates_and_cups",
        "instruction_id": "stack_plates_cups_01",
        "instruction_text": instruction,
        "language": "en",
    }), encoding="utf-8")
    code = f"""
import importlib.util,json
s=importlib.util.spec_from_file_location('exporter',{str(SCRIPT)!r})
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
print(json.dumps(m.read_episode_instruction_metadata(m.Path({str(episode)!r}))))
"""
    result = subprocess.run(["/usr/bin/python3", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["task"] == instruction
    assert payload["task_id"] == "stack_plates_and_cups"
    assert payload["instruction_id"] == "stack_plates_cups_01"
    assert payload["language"] == "en"


def test_exporter_discovers_nested_instruction_episodes(tmp_path: Path) -> None:
    for instruction_id in ("stack_plates_cups_01", "stack_plates_cups_02"):
        episode = tmp_path / "stack_plates_and_cups" / instruction_id / "episode_000000"
        (episode / "bag").mkdir(parents=True)
    code = f"""
import importlib.util,json
s=importlib.util.spec_from_file_location('exporter',{str(SCRIPT)!r})
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
paths=m.find_episode_dirs(m.Path({str(tmp_path)!r}))
print(json.dumps([str(p.relative_to(m.Path({str(tmp_path)!r}))) for p in paths]))
"""
    result = subprocess.run(["/usr/bin/python3", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [
        "stack_plates_and_cups/stack_plates_cups_01/episode_000000",
        "stack_plates_and_cups/stack_plates_cups_02/episode_000000",
    ]
