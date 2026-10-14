from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1] / "lerobot_tools"


def _load(name: str):
    path = ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_numeric_plot_defaults_cover_left_first_16d() -> None:
    module = _load("check_lerobot_v21_numeric_and_plot")
    assert module.DEFAULT_DIMS == 16
    assert module.DEFAULT_ARM_DIMS == list(range(7)) + list(range(8, 15))
    assert module.DEFAULT_GRIPPER_DIMS == [7, 15]


def test_interactive_downsampling_is_bounded_and_keeps_endpoints() -> None:
    module = _load("check_lerobot_v21_numeric_and_plot")
    indices = module.downsample_indices(10001, 2000)
    assert len(indices) <= 2000
    assert indices[0] == 0
    assert indices[-1] == 10000
    assert np.all(np.diff(indices) > 0)


def test_interactive_downsampling_zero_keeps_all_points() -> None:
    module = _load("check_lerobot_v21_numeric_and_plot")
    assert module.downsample_indices(7, 0).tolist() == list(range(7))


def test_missing_audit_checks_no_torso_component() -> None:
    module = _load("check_lerobot_v21_missing_audit")
    assert set(module.DETAILED_SKIP_KEYS) == {
        "cam_left_wrist", "cam_right_wrist",
        "state_left_arm", "state_left_gripper", "state_right_arm", "state_right_gripper",
        "action_left_arm", "action_left_gripper", "action_right_arm", "action_right_gripper",
    }
