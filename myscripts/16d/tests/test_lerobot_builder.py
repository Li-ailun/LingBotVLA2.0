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


def test_builder_uses_full_natural_language_instruction(tmp_path: Path) -> None:
    episode = tmp_path / "episode_000001"
    episode.mkdir()
    instruction = "Stack the pink plate and then the blue plate on the left."
    (episode / "task.txt").write_text(instruction + "\n", encoding="utf-8")
    code = (
        "import importlib.util;"
        f"s=importlib.util.spec_from_file_location('builder',{str(SCRIPT)!r});"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        f"print(m.read_task(m.Path({str(episode)!r})))"
    )
    result = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == instruction


def test_builder_discovers_nested_instruction_episodes(tmp_path: Path) -> None:
    for instruction_id in ("stack_plates_cups_01", "stack_plates_cups_03"):
        episode = tmp_path / instruction_id / "episode_000000"
        (episode / "images").mkdir(parents=True)
        (episode / "state.npy").write_bytes(b"placeholder")
        (episode / "action.npy").write_bytes(b"placeholder")
    code = (
        "import importlib.util,json;"
        f"s=importlib.util.spec_from_file_location('builder',{str(SCRIPT)!r});"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        f"r=m.Path({str(tmp_path)!r});"
        "print(json.dumps([str(p.relative_to(r)) for p in m.find_episode_dirs(r)]))"
    )
    result = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [
        "stack_plates_cups_01/episode_000000",
        "stack_plates_cups_03/episode_000000",
    ]


def test_builder_resizes_camera_image_to_declared_schema(tmp_path: Path) -> None:
    image_path = tmp_path / "cam_high.jpg"
    code = (
        "import importlib.util,json;"
        "from PIL import Image;"
        f"Image.new('RGB',(1920,1080),(10,20,30)).save({str(image_path)!r});"
        f"s=importlib.util.spec_from_file_location('builder',{str(SCRIPT)!r});"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        f"a=m.read_rgb_image(m.Path({str(image_path)!r}),m.IMAGE_SHAPES['cam_high']);"
        "print(json.dumps({'shape':list(a.shape),'dtype':str(a.dtype)}))"
    )
    result = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    image = json.loads(result.stdout)
    assert image == {"shape": [360, 640, 3], "dtype": "uint8"}


def test_builder_orders_by_instruction_operator_and_source_episode(tmp_path: Path) -> None:
    specs = [
        ("stack_plates_cups_02", "li", 0),
        ("stack_plates_cups_01", "yu", 0),
        ("stack_plates_cups_01", "li", 3),
        ("stack_plates_cups_01", "li", 1),
        ("stack_plates_cups_01", "liu", 0),
    ]
    for instruction, operator, source_index in specs:
        episode = tmp_path / operator / instruction / f"episode_{source_index:06d}_stamp"
        (episode / "images").mkdir(parents=True)
        (episode / "state.npy").write_bytes(b"placeholder")
        (episode / "action.npy").write_bytes(b"placeholder")
        (episode / "episode_meta.json").write_text(json.dumps({
            "task_id": "stack_plates_and_cups",
            "instruction_id": instruction,
            "operator_id": operator,
            "source_mcap_episode_index": source_index,
            "source_mcap_episode_name": episode.name,
        }), encoding="utf-8")
    code = (
        "import importlib.util,json;"
        f"s=importlib.util.spec_from_file_location('builder',{str(SCRIPT)!r});"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        f"r=m.Path({str(tmp_path)!r});"
        "eps=m.order_episode_dirs(m.find_episode_dirs(r));"
        "print(json.dumps([str(p.relative_to(r)) for p in eps]))"
    )
    result = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [
        "li/stack_plates_cups_01/episode_000001_stamp",
        "li/stack_plates_cups_01/episode_000003_stamp",
        "liu/stack_plates_cups_01/episode_000000_stamp",
        "yu/stack_plates_cups_01/episode_000000_stamp",
        "li/stack_plates_cups_02/episode_000000_stamp",
    ]


def test_builder_source_map_has_zero_and_one_based_numbers(tmp_path: Path) -> None:
    episode = tmp_path / "liu" / "stack_plates_cups_01" / "episode_000007_stamp"
    episode.mkdir(parents=True)
    (episode / "episode_meta.json").write_text(json.dumps({
        "task_id": "stack_plates_and_cups",
        "instruction_id": "stack_plates_cups_01",
        "instruction_text": "instruction one",
        "operator_id": "liu",
        "source_mcap_episode_index": 7,
        "source_mcap_episode_name": "episode_000007_stamp",
        "source_episode_dir": "/raw/stack_plates_and_cups/liu/stack_plates_cups_01/episode_000007_stamp",
        "source_episode_relative_path": "liu/stack_plates_cups_01/episode_000007_stamp",
        "frame_count": 123,
    }), encoding="utf-8")
    code = (
        "import importlib.util,json;"
        f"s=importlib.util.spec_from_file_location('builder',{str(SCRIPT)!r});"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        f"ep=m.Path({str(episode)!r});root=m.Path({str(tmp_path)!r});"
        "print(json.dumps(m.build_episode_source_record(ep,3,root)))"
    )
    result = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["lerobot_episode_index"] == 3
    assert payload["lerobot_sequence_number"] == 4
    assert payload["operator_id"] == "liu"
    assert payload["source_mcap_episode_index"] == 7
    assert payload["source_mcap_relative_path"].endswith(
        "liu/stack_plates_cups_01/episode_000007_stamp"
    )


def test_builder_source_map_uses_actual_lerobot_chunk(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    parquet = dataset_root / "data" / "chunk-001" / "episode_001234.parquet"
    parquet.parent.mkdir(parents=True)
    parquet.write_bytes(b"parquet")
    record = {
        "lerobot_episode_index": 1234,
        "lerobot_sequence_number": 1235,
        "lerobot_episode_file": "episode_001234.parquet",
    }
    code = (
        "import importlib.util,json;"
        f"s=importlib.util.spec_from_file_location('builder',{str(SCRIPT)!r});"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        f"root=m.Path({str(dataset_root)!r});records={json.dumps([record])!r};"
        "m.write_episode_source_maps(root,json.loads(records));"
        "print((root/'meta'/'episode_source_map.jsonl').read_text().strip())"
    )
    result = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["lerobot_episode_file"] == "data/chunk-001/episode_001234.parquet"


def test_builder_plans_bounded_worker_ranges() -> None:
    code = (
        "import importlib.util,json;"
        f"s=importlib.util.spec_from_file_location('builder',{str(SCRIPT)!r});"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "print(json.dumps(m.plan_worker_ranges(25,7,10)))"
    )
    result = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [[7, 17], [17, 25]]


def test_builder_rejects_non_positive_worker_batch_size() -> None:
    code = (
        "import importlib.util;"
        f"s=importlib.util.spec_from_file_location('builder',{str(SCRIPT)!r});"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "m.plan_worker_ranges(10,0,0)"
    )
    result = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True)
    assert result.returncode != 0
    assert "episodes_per_worker must be positive" in result.stderr
