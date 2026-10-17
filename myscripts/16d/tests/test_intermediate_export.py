from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "convert_tools" / "export_r1pro_mcap_to_intermediate.py"


def _raw_episode(root: Path, relative: str) -> Path:
    episode = root / relative
    (episode / "bag").mkdir(parents=True)
    return episode


def _intermediate_episode(root: Path, relative: str, *, complete: bool = True) -> Path:
    episode = root / relative
    episode.mkdir(parents=True)
    if complete:
        (episode / "episode_meta.json").write_text(json.dumps({
            "source_episode_relative_path": relative,
            "source_mcap_episode_name": episode.name,
        }))
        for name in ("timestamps.npy", "state.npy", "action.npy", "task.txt"):
            (episode / name).write_text("test")
        for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            (episode / "images" / camera).mkdir(parents=True)
    return episode


def _run_resume_plan(input_root: Path, output_root: Path) -> subprocess.CompletedProcess:
    code = f"""
import importlib.util,json
s=importlib.util.spec_from_file_location('exporter',{str(SCRIPT)!r})
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
i=m.Path({str(input_root)!r});o=m.Path({str(output_root)!r})
e=m.find_episode_dirs(i)
print(json.dumps([str(p.relative_to(i)) for p in m.plan_resume(e,i,o)]))
"""
    return subprocess.run(
        ["/usr/bin/python3", "-c", code], capture_output=True, text=True
    )


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


def test_exporter_preserves_operator_and_source_episode_metadata(tmp_path: Path) -> None:
    episode = tmp_path / "stack_plates_and_cups" / "yu" / "stack_plates_cups_03" / "episode_000012_20261013123456789"
    episode.mkdir(parents=True)
    (episode / "task.txt").write_text("instruction\n", encoding="utf-8")
    (episode / "record_config.json").write_text(json.dumps({
        "task_id": "stack_plates_and_cups",
        "instruction_id": "stack_plates_cups_03",
        "instruction_text": "instruction",
        "language": "en",
        "operator_id": "yu",
        "operator_order": 2,
        "episode_index": 12,
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
    assert payload["operator_id"] == "yu"
    assert payload["operator_order"] == 2
    assert payload["source_mcap_episode_index"] == 12
    assert payload["source_mcap_episode_name"] == episode.name


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


def test_resume_plan_returns_only_new_source_episodes(tmp_path: Path) -> None:
    input_root = tmp_path / "raw"
    output_root = tmp_path / "intermediate"
    first = "stack_plates_and_cups/li/stack_plates_cups_01/episode_000001_stamp"
    second = "stack_plates_and_cups/li/stack_plates_cups_01/episode_000002_stamp"
    _raw_episode(input_root, first)
    _raw_episode(input_root, second)
    _intermediate_episode(output_root, first)

    result = _run_resume_plan(input_root, output_root)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [second]


def test_resume_plan_rejects_intermediate_whose_source_is_gone(tmp_path: Path) -> None:
    input_root = tmp_path / "raw"
    output_root = tmp_path / "intermediate"
    current = "stack_plates_and_cups/li/stack_plates_cups_01/episode_000001_stamp"
    stale = "stack_plates_and_cups/li/stack_plates_cups_01/episode_000002_stamp"
    _raw_episode(input_root, current)
    _intermediate_episode(output_root, current)
    _intermediate_episode(output_root, stale)

    result = _run_resume_plan(input_root, output_root)

    assert result.returncode != 0
    assert "stale source episodes" in result.stderr
    assert stale in result.stderr
    assert "--overwrite" in result.stderr


def test_resume_plan_rejects_incomplete_existing_episode(tmp_path: Path) -> None:
    input_root = tmp_path / "raw"
    output_root = tmp_path / "intermediate"
    relative = "stack_plates_and_cups/li/stack_plates_cups_01/episode_000001_stamp"
    _raw_episode(input_root, relative)
    _intermediate_episode(output_root, relative, complete=False)

    result = _run_resume_plan(input_root, output_root)

    assert result.returncode != 0
    assert "incomplete intermediate episode" in result.stderr
    assert relative in result.stderr
    assert "--overwrite" in result.stderr


def test_resume_plan_rejects_source_mapping_mismatch(tmp_path: Path) -> None:
    input_root = tmp_path / "raw"
    output_root = tmp_path / "intermediate"
    relative = "stack_plates_and_cups/li/stack_plates_cups_01/episode_000001_stamp"
    _raw_episode(input_root, relative)
    episode = _intermediate_episode(output_root, relative)
    (episode / "episode_meta.json").write_text(json.dumps({
        "source_episode_relative_path": (
            "stack_plates_and_cups/liu/stack_plates_cups_01/episode_000001_stamp"
        ),
        "source_mcap_episode_name": episode.name,
    }))

    result = _run_resume_plan(input_root, output_root)

    assert result.returncode != 0
    assert "intermediate source mapping mismatch" in result.stderr
    assert relative in result.stderr
    assert "--overwrite" in result.stderr


def test_resume_and_overwrite_are_mutually_exclusive(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            "/usr/bin/python3", str(SCRIPT),
            "--input", str(tmp_path / "raw"),
            "--output", str(tmp_path / "intermediate"),
            "--resume", "--overwrite",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "not allowed with argument" in result.stderr
