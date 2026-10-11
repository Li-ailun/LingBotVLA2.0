import pytest

from myscripts.inference_tools.serve_lingbot_rtc_policy import build_server_metadata, validate_server_metadata


def test_metadata_identifies_strict_lingbot_contract():
    metadata = build_server_metadata("checkpoint", "norm")
    validate_server_metadata(metadata)
    assert metadata["action_horizon"] == 50
    assert metadata["physical_action_dim"] == 20
    assert metadata["rtc_capable"] is True


def test_wrong_backend_is_rejected():
    metadata = build_server_metadata("checkpoint", "norm")
    metadata["model_backend"] = "openpi"
    with pytest.raises(ValueError, match="backend"):
        validate_server_metadata(metadata)
