"""Validated model-space request passed into the RTC denoising loop."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LingBotRTCRequest:
    target: torch.Tensor | None = None
    prefix_length: int = 0
    inference_delay: int = 0
    execution_horizon: int = 10
    max_guidance_weight: float = 10.0
    prefix_attention_schedule: int = 1
    valid_action_mask: torch.Tensor | None = None

    @property
    def enabled(self) -> bool:
        return self.target is not None and self.prefix_length > 0 and self.execution_horizon > 0

    def validate_like(self, actions: torch.Tensor) -> None:
        if not self.enabled:
            return
        if actions.ndim != 3 or actions.shape[1] != 50:
            raise ValueError(f"LingBot actions must have shape [B, 50, D], got {tuple(actions.shape)}")
        if self.target is None or self.target.shape != actions.shape:
            raise ValueError("RTC target shape must exactly match [B, 50, D] sampled actions")
        if not torch.isfinite(self.target).all():
            raise ValueError("RTC target must contain only finite values")
        if not 0 <= self.inference_delay <= self.execution_horizon <= self.prefix_length <= 50:
            raise ValueError("RTC bounds require 0 <= delay <= horizon <= prefix <= 50")
        if not torch.isfinite(torch.as_tensor(self.max_guidance_weight)) or self.max_guidance_weight < 0:
            raise ValueError("RTC guidance weight must be finite and non-negative")
        if self.valid_action_mask is not None:
            try:
                torch.broadcast_shapes(self.valid_action_mask.shape, actions.shape)
            except RuntimeError as error:
                raise ValueError("RTC valid action mask is not broadcastable to target") from error
