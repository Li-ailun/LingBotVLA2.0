from pathlib import Path

import numpy as np
import pytest

from myscripts.inference_tools.lingbot_rtc.lingbot_vla_v2_rtc_policy import LingBotVlaV2RTCPolicy


class FakeBackend:
    def infer(self, observation, rtc_request=None):
        return np.zeros((50, 20), dtype=np.float32)


def observation():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    return {"images": {"cam_high": image, "cam_left_wrist": image, "cam_right_wrist": image},
            "state": np.zeros(20, dtype=np.float32), "prompt": "pick object"}


def test_policy_validates_contract_and_returns_full_chunk(tmp_path: Path):
    config = tmp_path / "robot.yaml"; config.write_text("ready: true")
    norm = tmp_path / "norm.json"; norm.write_text('{"norm_stats": {}}')
    policy = LingBotVlaV2RTCPolicy(backend=FakeBackend(), robot_config_path=config, norm_stats_path=norm)
    result = policy.infer(observation())
    assert result["actions"].shape == (50, 20)
    assert result["policy_timing"]["rtc_enabled"] is False


def test_missing_assets_fail_closed_for_physical_inference():
    policy = LingBotVlaV2RTCPolicy(backend=FakeBackend())
    assert not policy.production_assets_ready
    with pytest.raises(RuntimeError, match="production assets"):
        policy.infer(observation())


def test_bad_camera_or_state_is_rejected(tmp_path: Path):
    config = tmp_path / "r"; config.write_text("x")
    norm = tmp_path / "n"; norm.write_text("x")
    policy = LingBotVlaV2RTCPolicy(backend=FakeBackend(), robot_config_path=config, norm_stats_path=norm)
    bad = observation(); bad["state"] = np.zeros(19)
    with pytest.raises(ValueError, match="20"):
        policy.infer(bad)
