import pytest
import torch

from myscripts.inference_tools.lingbot_rtc.rtc_request import LingBotRTCRequest
from myscripts.inference_tools.lingbot_rtc.rtc_sampler import (
    RTC_SCHEDULE_EXP,
    RTC_SCHEDULE_LINEAR,
    RTC_SCHEDULE_ONES,
    RTC_SCHEDULE_ZEROS,
    apply_rtc_velocity_guidance,
    build_rtc_prefix_weights,
)


def test_rtc_request_validates_model_space_payload():
    target = torch.zeros(1, 50, 20)
    request = LingBotRTCRequest(target=target, prefix_length=20, inference_delay=4)
    request.validate_like(target)
    with pytest.raises(ValueError, match="finite"):
        LingBotRTCRequest(target=target.fill_(float("nan")), prefix_length=20).validate_like(torch.zeros_like(target))


@pytest.mark.parametrize("schedule", [RTC_SCHEDULE_LINEAR, RTC_SCHEDULE_EXP, RTC_SCHEDULE_ONES, RTC_SCHEDULE_ZEROS])
def test_prefix_schedules_are_bounded_and_zero_after_end(schedule):
    weights = build_rtc_prefix_weights(-4, 10, 50, schedule, device=torch.device("cpu"), dtype=torch.float32)
    assert weights.shape == (50,)
    assert torch.all((weights >= 0) & (weights <= 1))
    assert torch.count_nonzero(weights[10:]) == 0


def test_guidance_changes_only_valid_dimensions():
    x = torch.ones(1, 50, 4)
    target = torch.zeros_like(x)
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[:, :10, :2] = True
    request = LingBotRTCRequest(target, 20, 4, 10, 10.0, RTC_SCHEDULE_EXP, mask)
    guided = apply_rtc_velocity_guidance(lambda value: value * 0.5, x, torch.tensor(0.5), request)
    base = x * 0.5
    assert torch.any(guided[:, :10, :2] != base[:, :10, :2])
    assert torch.equal(guided[:, :, 2:], base[:, :, 2:])
    assert not guided.requires_grad


def test_invalid_schedule_and_bounds_are_rejected():
    with pytest.raises(ValueError, match="schedule"):
        build_rtc_prefix_weights(0, 1, 50, 99, device=torch.device("cpu"), dtype=torch.float32)
    with pytest.raises(ValueError, match="prefix"):
        LingBotRTCRequest(torch.zeros(1, 50, 2), 51).validate_like(torch.zeros(1, 50, 2))
