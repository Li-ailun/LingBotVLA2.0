from pathlib import Path

import pytest

from simulation_rtc.config import RTCConfig, RobotActionLayout


ROOT = Path(__file__).resolve().parents[3]
TOOL = Path(__file__).resolve().parents[1]


def test_shipped_config_has_fixed_fast_simulation_values():
    config = RTCConfig.from_yaml(TOOL / "configs/lingbot_action_step2000_rtc_fast.yaml")
    assert (config.chunk_horizon, config.request_index, config.fixed_delay_steps) == (50, 10, 10)
    assert (config.execution_horizon, config.crossfade_steps, config.ema_alpha) == (10, 4, 0.10)
    assert config.request_timeout_s == 120.0


def test_environment_overrides_runtime_tuning_without_changing_yaml(monkeypatch):
    monkeypatch.setenv("LINGBOT_RTC_REQUEST_INDEX", "10")
    monkeypatch.setenv("LINGBOT_RTC_FIXED_DELAY_STEPS", "5")
    monkeypatch.setenv("LINGBOT_RTC_EXECUTION_HORIZON", "10")
    monkeypatch.setenv("LINGBOT_RTC_CROSSFADE_STEPS", "3")
    monkeypatch.setenv("LINGBOT_RTC_EMA_ALPHA", "0.5")

    config = RTCConfig.from_yaml(TOOL / "configs/lingbot_action_step2000_rtc_fast.yaml")

    assert config.request_index == 10
    assert config.fixed_delay_steps == 5
    assert config.execution_horizon == 10
    assert config.crossfade_steps == 3
    assert config.ema_alpha == 0.5


@pytest.mark.parametrize("override", [
    {"chunk_horizon": 49},
    {"request_index": 50},
    {"fixed_delay_steps": 11},
    {"execution_horizon": 11},
    {"crossfade_steps": -1},
    {"ema_alpha": 0.0},
])
def test_invalid_config_is_rejected(override):
    values = dict(chunk_horizon=50, request_index=40, fixed_delay_steps=5,
                  execution_horizon=10, crossfade_steps=4, ema_alpha=0.36,
                  max_guidance_weight=10.0, prefix_attention_schedule="exp",
                  request_timeout_s=30.0)
    values.update(override)
    with pytest.raises(ValueError):
        RTCConfig(**values)


def test_exported_robot_layout_is_derived_from_config():
    layout = RobotActionLayout.from_yaml(
        ROOT / "models/lingbot_action_step2000/configs/robot_configs/robotwin.yaml"
    )
    assert layout.action_dim == 14
    assert layout.gripper_indices == (6, 13)


def test_overlapping_layout_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("""
actions:
- action.arm.position:
    origin_keys:
    - action: {start: 0, end: 7}
- action.effector.position:
    origin_keys:
    - action: {start: 6, end: 8}
""", encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        RobotActionLayout.from_yaml(path)
