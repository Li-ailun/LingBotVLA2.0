import numpy as np
import pytest

from simulation_rtc.config import RTCConfig, RobotActionLayout
from simulation_rtc.protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    build_bootstrap_request,
    build_rtc_request,
    validate_response,
)


CONFIG = RTCConfig(50, 15, 5, 10, 4, 0.36, 10.0, "exp", 30.0)
LAYOUT = RobotActionLayout(14, (6, 13))
OBS = {"observation.state": np.zeros(14, np.float32), "task": "lift pot"}


def test_bootstrap_request_is_versioned_and_has_no_rtc_target():
    request = build_bootstrap_request(OBS, episode_id=3, request_id=0)
    payload = request.to_payload()
    assert payload["protocol_version"] == PROTOCOL_VERSION
    assert payload["episode_id"] == 3 and payload["request_id"] == 0
    assert payload["rtc"]["enabled"] is False


def test_rtc_request_carries_only_time_aligned_old_tail_and_exact_parameters():
    old = np.repeat(np.arange(50, dtype=np.float32)[:, None], 14, axis=1)
    request = build_rtc_request(OBS, old, episode_id=3, request_id=1,
                                old_index=15, config=CONFIG)
    payload = request.to_payload()
    assert payload["rtc"]["enabled"] is True
    assert payload["rtc"]["old_index"] == 15
    assert payload["rtc"]["inference_delay"] == 5
    assert payload["rtc"]["execution_horizon"] == 10
    assert payload["rtc"]["prefix_length"] == 35
    previous = np.asarray(payload["rtc"]["previous_actions"])
    assert previous.shape == (35, 14)
    assert np.all(previous[0] == 15) and np.all(previous[-1] == 49)


def test_valid_response_round_trips():
    response = validate_response({
        "protocol_version": PROTOCOL_VERSION,
        "episode_id": 3,
        "request_id": 1,
        "action": np.zeros((50, 14), np.float32),
        "server_timing": {"infer_ms": 280.0},
        "rtc_ack": {"enabled": True},
    }, 3, 1, LAYOUT)
    assert response.actions.shape == (50, 14)
    assert response.server_timing["infer_ms"] == 280.0


@pytest.mark.parametrize("mutation", [
    {"protocol_version": "wrong"},
    {"episode_id": 2},
    {"request_id": 0},
    {"action": np.zeros((49, 14), np.float32)},
    {"action": np.full((50, 14), np.nan, np.float32)},
])
def test_invalid_or_stale_response_is_rejected(mutation):
    payload = {"protocol_version": PROTOCOL_VERSION, "episode_id": 3, "request_id": 1,
               "action": np.zeros((50, 14), np.float32), "server_timing": {}, "rtc_ack": {}}
    payload.update(mutation)
    with pytest.raises(ProtocolError):
        validate_response(payload, 3, 1, LAYOUT)
