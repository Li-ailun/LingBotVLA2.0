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
