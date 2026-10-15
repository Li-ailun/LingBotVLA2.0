from __future__ import annotations

import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory


SCRIPT = Path(__file__).parents[1] / "record_tools" / "record_r1pro_mcap_episode.py"


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/usr/bin/python3", "-c", code], capture_output=True, text=True, check=False
    )


def test_recorder_exposes_only_torso_free_16d_core_topics() -> None:
    result = _run_python(
        "import importlib.util,json;"
        f"s=importlib.util.spec_from_file_location('recorder',{str(SCRIPT)!r});"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "print(json.dumps(m.r1pro_16d_core_topics()))"
    )
    assert result.returncode == 0, result.stderr
    topics = json.loads(result.stdout)
    assert len(topics) == 11
    assert not any("torso" in topic.lower() for topic in topics)


def test_recorder_cli_has_no_torso_option() -> None:
    result = subprocess.run(
        ["/usr/bin/python3", str(SCRIPT), "--help"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--record-all-topics" not in result.stdout
    assert "--record-minimal-with-ee" not in result.stdout
    assert "torso" not in result.stdout.lower()


def test_plate_cup_instruction_catalog_and_directory_layout() -> None:
    result = _run_python(
        "import importlib.util,json;"
        f"s=importlib.util.spec_from_file_location('recorder',{str(SCRIPT)!r});"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "spec=m.resolve_instruction('2');"
        "print(json.dumps({'id':spec.instruction_id,'task_id':spec.task_id,'text':spec.text,'parts':m.instruction_path_parts(spec)}))"
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["id"] == "stack_plates_cups_02"
    assert payload["task_id"] == "stack_plates_and_cups"
    assert payload["parts"] == ["stack_plates_and_cups", "stack_plates_cups_02"]
    assert payload["text"] == (
        "Arrange the blue plate at the bottom with the pink plate on top. "
        "Arrange the cups from bottom to top as red, yellow, and blue, "
        "with the cup stack on the plate stack."
    )


def test_plate_cup_instructions_constrain_final_state_without_fixed_regions() -> None:
    result = _run_python(
        "import importlib.util,json;"
        f"s=importlib.util.spec_from_file_location('recorder',{str(SCRIPT)!r});"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "print(json.dumps([m.resolve_instruction(str(i)).text for i in (1,2,3)]))"
    )
    assert result.returncode == 0, result.stderr
    instructions = json.loads(result.stdout)
    assert instructions == [
        (
            "Arrange the blue plate at the bottom with the pink plate on top. "
            "Arrange the cups from bottom to top as blue, red, and yellow, "
            "with the cup stack on the plate stack."
        ),
        (
            "Arrange the blue plate at the bottom with the pink plate on top. "
            "Arrange the cups from bottom to top as red, yellow, and blue, "
            "with the cup stack on the plate stack."
        ),
        (
            "Arrange the pink plate at the bottom with the blue plate on top. "
            "Arrange the cups from bottom to top as yellow, blue, and red, "
            "with the cup stack on the plate stack."
        ),
    ]
    assert not any(
        region in instruction.lower()
        for instruction in instructions
        for region in ("left", "right", "center", "middle")
    )


def test_recorder_cli_accepts_instruction_id_without_generic_task() -> None:
    result = subprocess.run(
        ["/usr/bin/python3", str(SCRIPT), "--help"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--instruction-id" in result.stdout
    assert "--task" in result.stdout


def test_recorder_operator_catalog_and_directory_layout() -> None:
    result = _run_python(
        "import importlib.util,json,types;"
        f"s=importlib.util.spec_from_file_location('recorder',{str(SCRIPT)!r});"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "args=types.SimpleNamespace(instruction_id='stack_plates_cups_02',operator_id='liu');"
        "print(json.dumps({'operators':m.OPERATOR_IDS,'resolved':m.resolve_operator(' LIU '),'path':str(m.task_directory(m.Path('/raw'),args))}))"
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload == {
        "operators": ["li", "liu", "yu", "xing"],
        "resolved": "liu",
        "path": "/raw/stack_plates_and_cups/liu/stack_plates_cups_02",
    }


def test_recorder_rejects_unknown_operator() -> None:
    result = _run_python(
        "import importlib.util;"
        f"s=importlib.util.spec_from_file_location('recorder',{str(SCRIPT)!r});"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "m.resolve_operator('other')"
    )
    assert result.returncode != 0
    assert "choose one of: li, liu, yu, xing" in result.stderr


def test_recorder_cli_exposes_operator_option() -> None:
    result = subprocess.run(
        ["/usr/bin/python3", str(SCRIPT), "--help"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--operator" in result.stdout


def test_direct_recording_cli_disables_timestamp_monitor() -> None:
    result = _run_python(
        "import importlib.util,json,types;"
        f"s=importlib.util.spec_from_file_location('recorder',{str(SCRIPT)!r});"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "args=types.SimpleNamespace(enable_timestamp_monitor=True,direct_recording=True);"
        "m.apply_recording_mode(args);"
        "print(json.dumps({'monitor':args.enable_timestamp_monitor}))"
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"monitor": False}


def test_recorder_cli_exposes_direct_recording_option() -> None:
    result = subprocess.run(
        ["/usr/bin/python3", str(SCRIPT), "--help"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--direct-recording" in result.stdout


def test_episode_index_collision_matches_timestamped_episode_directories() -> None:
    with TemporaryDirectory() as tmp:
        task_dir = Path(tmp)
        (task_dir / "episode_000125_20261015120000000").mkdir()
        (task_dir / "episode_000126_20261015120100000").mkdir()
        result = _run_python(
            "import importlib.util,json,pathlib;"
            f"s=importlib.util.spec_from_file_location('recorder',{str(SCRIPT)!r});"
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
            f"d=pathlib.Path({str(task_dir)!r});"
            "print(json.dumps({'used':m.episode_directories_for_index(d,125),"
            "'free':m.episode_directories_for_index(d,127)}))"
        )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert len(payload["used"]) == 1
    assert payload["used"][0].endswith("episode_000125_20261015120000000")
    assert payload["free"] == []


def test_start_episode_prompt_rejects_duplicate_then_accepts_free_index() -> None:
    with TemporaryDirectory() as tmp:
        task_dir = Path(tmp)
        (task_dir / "episode_000125_20261015120000000").mkdir()
        result = _run_python(
            "import importlib.util,json,pathlib;"
            f"s=importlib.util.spec_from_file_location('recorder',{str(SCRIPT)!r});"
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
            f"d=pathlib.Path({str(task_dir)!r});"
            "answers=iter(['125','bad','127']);messages=[];"
            "chosen=m.prompt_start_episode_index(d,input_fn=lambda _:next(answers),print_fn=messages.append);"
            "print(json.dumps({'chosen':chosen,'messages':messages}))"
        )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["chosen"] == 127
    assert any("already exists" in message for message in payload["messages"])
    assert any("non-negative integer" in message for message in payload["messages"])
