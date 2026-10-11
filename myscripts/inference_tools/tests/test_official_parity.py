import torch

from myscripts.inference_tools.lingbot_rtc.modeling_lingbot_vla_v2_rtc import denoise_actions_rtc
from myscripts.inference_tools.lingbot_rtc.rtc_request import LingBotRTCRequest


class FakeFlow:
    class Config:
        num_steps = 10
    config = Config()

    @staticmethod
    def predict_velocity(state, prefix_masks, cache, x_t, time, prefix_position_ids=None):
        return x_t * 0.25 + time[:, None, None]


def official_loop(flow, noise):
    dt = torch.tensor(-1.0 / flow.config.num_steps)
    x_t, time = noise.clone(), torch.tensor(1.0)
    while time >= -dt / 2:
        x_t += dt * flow.predict_velocity(None, None, None, x_t, time.expand(x_t.shape[0]), prefix_position_ids=None)
        time += dt
    return x_t


def test_disabled_rtc_is_stepwise_equivalent():
    noise = torch.randn(1, 50, 4)
    actual = denoise_actions_rtc(FakeFlow(), None, None, None, noise.clone(), rtc_request=None)
    assert torch.equal(actual, official_loop(FakeFlow(), noise))


def test_enabled_rtc_is_finite_and_changes_overlap():
    noise = torch.randn(1, 50, 4)
    request = LingBotRTCRequest(torch.zeros_like(noise), 20, 4, 10)
    actual = denoise_actions_rtc(FakeFlow(), None, None, None, noise, rtc_request=request)
    assert torch.isfinite(actual).all()
