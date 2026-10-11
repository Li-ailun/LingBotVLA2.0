"""RLinf-compatible RTC velocity guidance for LingBot flow matching."""

from collections.abc import Callable

import torch

from .rtc_request import LingBotRTCRequest

RTC_SCHEDULE_LINEAR = 0
RTC_SCHEDULE_EXP = 1
RTC_SCHEDULE_ONES = 2
RTC_SCHEDULE_ZEROS = 3


def build_rtc_prefix_weights(start, end, total, schedule, *, device, dtype):
    if total <= 0:
        raise ValueError("total must be positive")
    if schedule not in (0, 1, 2, 3):
        raise ValueError(f"Unsupported RTC prefix schedule: {schedule}")
    end = max(0, min(int(end), total))
    start = max(0, min(int(start), end))
    indices = torch.arange(total, device=device, dtype=dtype)
    linear = torch.clamp((start - 1 - indices) / max(end - start + 1, 1) + 1, 0, 1)
    choices = {
        RTC_SCHEDULE_LINEAR: linear,
        RTC_SCHEDULE_EXP: linear * torch.expm1(linear) / (torch.e - 1),
        RTC_SCHEDULE_ONES: torch.ones_like(indices),
        RTC_SCHEDULE_ZEROS: (indices < start).to(dtype),
    }
    return torch.where(indices >= end, torch.zeros_like(indices), choices[schedule])


def rtc_guidance_weight(time, max_guidance_weight):
    tau = 1.0 - time
    one_minus_tau_sq = torch.square(1.0 - tau)
    tiny = torch.finfo(time.dtype).eps
    value = ((1.0 - tau) / torch.clamp(tau, min=tiny)) * (
        (one_minus_tau_sq + torch.square(tau)) / torch.clamp(one_minus_tau_sq, min=tiny)
    )
    return torch.nan_to_num(value, nan=0.0, posinf=max_guidance_weight, neginf=0.0).clamp(0, max_guidance_weight)


def apply_rtc_velocity_guidance(
    base_velocity_fn: Callable[[torch.Tensor], torch.Tensor],
    x_t: torch.Tensor,
    time: torch.Tensor,
    request: LingBotRTCRequest | None,
) -> torch.Tensor:
    if request is None or not request.enabled:
        return base_velocity_fn(x_t)
    request.validate_like(x_t)
    weights = build_rtc_prefix_weights(
        request.inference_delay, request.execution_horizon, x_t.shape[1],
        request.prefix_attention_schedule, device=x_t.device, dtype=x_t.dtype,
    )[None, :, None]
    target = request.target.to(device=x_t.device, dtype=x_t.dtype)
    mask = torch.ones_like(x_t) if request.valid_action_mask is None else request.valid_action_mask.to(x_t).expand_as(x_t)
    with torch.enable_grad():
        differentiable_x = x_t.detach().requires_grad_(True)
        base_velocity = base_velocity_fn(differentiable_x)
        x1_estimate = differentiable_x - time * base_velocity
        correction = torch.autograd.grad(
            x1_estimate, differentiable_x,
            grad_outputs=(target - x1_estimate) * weights * mask,
            create_graph=False, retain_graph=False,
        )[0]
        guided = base_velocity - rtc_guidance_weight(time, request.max_guidance_weight) * correction
    return guided.detach()
