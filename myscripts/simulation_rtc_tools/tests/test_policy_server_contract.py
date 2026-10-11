import numpy as np
import pytest

from simulation_rtc.protocol import PROTOCOL_VERSION
from server.policy_server_rtc import RTCServerCore, normalize_physical_actions


def payload(enabled):
    value = {"protocol_version": PROTOCOL_VERSION, "episode_id": 4, "request_id": 2,
             "observation.state": np.zeros(14, np.float32), "task": "lift pot",
             "rtc": {"enabled": enabled}}
    if enabled:
        tail = np.repeat(np.arange(15, 50, dtype=np.float32)[:, None], 14, axis=1)
        value["rtc"].update(previous_actions=tail, old_index=15,
                            prefix_length=35, inference_delay=5, execution_horizon=10,
                            max_guidance_weight=10.0, prefix_attention_schedule="exp")
    return value


def test_bootstrap_uses_ordinary_sampler_and_returns_versioned_physical_chunk():
    calls = []
    core = RTCServerCore(lambda obs: calls.append("ordinary") or np.zeros((50, 14), np.float32),
                         lambda obs, request: calls.append("rtc"), lambda actions: actions, 14)
    result = core.infer(payload(False))
    assert calls == ["ordinary"]
    assert result["action"].shape == (50, 14)
    assert result["protocol_version"] == PROTOCOL_VERSION
    assert result["episode_id"] == 4 and result["request_id"] == 2
    assert result["rtc_ack"]["enabled"] is False


def test_rtc_request_normalizes_previous_chunk_and_calls_rtc_sampler():
    seen = {}
    def normalize(actions):
        seen["physical"] = actions.copy()
        return np.pad(actions * 2, ((0, 0), (0, 18)))
    def rtc_infer(obs, request):
        seen["request"] = request
        return np.full((50, 14), 3, np.float32)
    core = RTCServerCore(lambda obs: None, rtc_infer, normalize, 14)
    result = core.infer(payload(True))
    assert seen["physical"].shape == (50, 14)
    assert np.all(seen["physical"][0] == 15)
    assert np.all(seen["physical"][34] == 49)
    assert np.all(seen["physical"][35:] == 49)
    assert seen["request"]["target"].shape == (1, 50, 32)
    assert seen["request"]["valid_action_mask"].shape == (1, 1, 32)
    assert seen["request"]["inference_delay"] == 5
    assert result["action"].shape == (50, 14)
    assert result["rtc_ack"]["enabled"] is True


def test_physical_target_conversion_uses_action_only_mapping_without_full_apply():
    import torch
    class IdentityNormalizer:
        def normalize(self, item): return item
    class FakeTransform:
        class model_config:
            max_action_dim = 55
        class feature_config:
            joints = ["arm.position", "effector.position"]
            joints_max_dim = {"arm.position": 24, "effector.position": 8}
        actions = ["action.arm.position", "action.effector.position"]
        key_mapping = {
            "action.arm.position": {"origin_keys": {"action": {"start": 0, "end": 6}, "action*": {"start": 7, "end": 13}}},
            "action.effector.position": {"origin_keys": {"action": {"start": 6, "end": 7}, "action*": {"start": 13, "end": 14}}},
        }
        normalizer = IdentityNormalizer()
        def pad_and_concat(self, item, w_action=True):
            raise AssertionError("action-only conversion must not call pad_and_concat")
        def unused_pad_and_concat(self, item, w_action=True):
            action = torch.cat([item["action.arm.position"], item["action.effector.position"]], dim=-1)
            return {"actions": torch.nn.functional.pad(action, (0, 18)), "action_joint_mask": torch.tensor([True] * 14 + [False] * 18)}
    target, mask = normalize_physical_actions(FakeTransform(), np.ones((50, 14), np.float32))
    assert target.shape == (50, 55) and mask.shape == (1, 1, 55)
    assert int(mask.sum()) == 14


def test_server_rejects_bad_protocol_and_nonfinite_model_output():
    core = RTCServerCore(lambda obs: np.full((50, 14), np.nan), lambda obs, req: None,
                         lambda value: value, 14)
    with pytest.raises(ValueError, match="protocol"):
        core.infer(dict(payload(False), protocol_version="bad"))
    with pytest.raises(ValueError, match="finite"):
        core.infer(payload(False))
