from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1] / "lerobot_tools"


def _load(name: str):
    path = ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _valid_info() -> dict:
    return {
        "features": {
            "observation.state": {
                "shape": [16],
                "names": [
                    *[f"left_arm_j{i}" for i in range(7)], "left_gripper",
                    *[f"right_arm_j{i}" for i in range(7)], "right_gripper",
                ],
            },
            "action": {
                "shape": [16],
                "names": [
                    *[f"target_left_arm_j{i}" for i in range(7)], "target_left_gripper",
                    *[f"target_right_arm_j{i}" for i in range(7)], "target_right_gripper",
                ],
            },
        }
    }


def test_keyboard_replay_rejects_legacy_20d_metadata() -> None:
    module = _load("replay_lerobot_v21_parquet_images_keyboard")
    info = _valid_info()
    info["features"]["observation.state"]["shape"] = [20]
    with pytest.raises(ValueError, match=r"expected metadata shape \[16\]"):
        module.validate_16d_info(info)


def test_web_replay_uses_four_left_first_groups() -> None:
    module = _load("replay_lerobot_v21_web")
    module.validate_16d_info(_valid_info())
    assert list(module.GROUPS) == ["left_arm", "left_gripper", "right_arm", "right_gripper"]
    indices = [index for group in module.GROUPS.values() for index in group["indices"]]
    assert indices == list(range(16))
