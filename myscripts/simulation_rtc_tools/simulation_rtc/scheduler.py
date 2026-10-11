from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from .config import RTCConfig
from .protocol import RTCResponse


class SchedulerState(Enum):
    BOOTSTRAP = auto()
    EXECUTING = auto()
    REQUEST_INFLIGHT = auto()
    READY_GATED = auto()
    WAITING_RESULT = auto()
    EPISODE_DONE = auto()
    FAILED = auto()


@dataclass(frozen=True)
class SchedulerDecision:
    action: np.ndarray | None
    chunk_index: int
    request_due: bool = False
    should_wait: bool = False
    promoted: bool = False
    prior_chunk: np.ndarray | None = None
    prior_index: int | None = None
    crop_steps: int = 0


class RTCScheduler:
    def __init__(self, config: RTCConfig):
        self.config = config
        self.state = SchedulerState.BOOTSTRAP
        self.episode_id = -1
        self.current_chunk: np.ndarray | None = None
        self.chunk_index = 0
        self.inflight_request_id: int | None = None
        self.ready_response: RTCResponse | None = None

    def start_episode(self, episode_id: int) -> None:
        self.episode_id = int(episode_id)
        self.current_chunk = None
        self.chunk_index = 0
        self.inflight_request_id = None
        self.ready_response = None
        self.state = SchedulerState.BOOTSTRAP

    def install_bootstrap(self, chunk: np.ndarray) -> None:
        actions = self._validate_chunk(chunk)
        self.current_chunk = actions.copy()
        self.chunk_index = 0
        self.state = SchedulerState.EXECUTING

    def mark_request_submitted(self, request_id: int) -> None:
        if self.current_chunk is None or self.chunk_index != self.config.request_index:
            raise RuntimeError("request can only be submitted at request_index")
        if self.inflight_request_id is not None:
            raise RuntimeError("an RTC request is already in flight")
        self.inflight_request_id = int(request_id)
        self.state = SchedulerState.REQUEST_INFLIGHT

    def accept_response(self, response: RTCResponse) -> bool:
        if self.state in {SchedulerState.EPISODE_DONE, SchedulerState.FAILED}:
            return False
        if response.episode_id != self.episode_id or response.request_id != self.inflight_request_id:
            return False
        self._validate_chunk(response.actions)
        self.ready_response = response
        if self.chunk_index - self.config.request_index < self.config.fixed_delay_steps:
            self.state = SchedulerState.READY_GATED
        return True

    def before_action(self) -> SchedulerDecision:
        if self.current_chunk is None:
            raise RuntimeError("bootstrap chunk is not installed")
        if self.state in {SchedulerState.EPISODE_DONE, SchedulerState.FAILED}:
            raise RuntimeError("episode is no longer active")
        elapsed = self.chunk_index - self.config.request_index
        if self.inflight_request_id is not None and elapsed >= self.config.fixed_delay_steps:
            if self.ready_response is None:
                self.state = SchedulerState.WAITING_RESULT
                return SchedulerDecision(None, self.chunk_index, should_wait=True)
            old_chunk, old_index = self.current_chunk, self.chunk_index
            crop_steps = max(0, old_index - self.config.request_index)
            raw_chunk = self.ready_response.actions.copy()
            if crop_steps >= len(raw_chunk):
                raise RuntimeError("RTC delay exhausted the returned action chunk")
            aligned = raw_chunk[crop_steps:].copy()
            if crop_steps:
                padding = np.repeat(raw_chunk[-1:], crop_steps, axis=0)
                aligned = np.concatenate((aligned, padding), axis=0)
            self.current_chunk = aligned
            self.chunk_index = 0
            self.ready_response = None
            self.inflight_request_id = None
            self.state = SchedulerState.EXECUTING
            return SchedulerDecision(self.current_chunk[0].copy(), 0, promoted=True,
                                     prior_chunk=old_chunk.copy(), prior_index=old_index,
                                     crop_steps=crop_steps)
        if self.chunk_index >= len(self.current_chunk):
            self.state = SchedulerState.WAITING_RESULT
            return SchedulerDecision(None, self.chunk_index, should_wait=True)
        request_due = (self.chunk_index == self.config.request_index
                       and self.inflight_request_id is None)
        return SchedulerDecision(self.current_chunk[self.chunk_index].copy(), self.chunk_index,
                                 request_due=request_due)

    def after_action(self) -> None:
        if self.current_chunk is None or self.state in {SchedulerState.WAITING_RESULT,
                                                        SchedulerState.EPISODE_DONE,
                                                        SchedulerState.FAILED}:
            raise RuntimeError("no executable action is pending")
        self.chunk_index += 1

    def finish_episode(self) -> None:
        self.ready_response = None
        self.inflight_request_id = None
        self.state = SchedulerState.EPISODE_DONE

    def fail_episode(self) -> None:
        self.ready_response = None
        self.inflight_request_id = None
        self.state = SchedulerState.FAILED

    def _validate_chunk(self, chunk: np.ndarray) -> np.ndarray:
        actions = np.asarray(chunk, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[0] != self.config.chunk_horizon:
            raise ValueError("action chunk must be finite [50, D]")
        if not np.isfinite(actions).all():
            raise ValueError("action chunk must be finite [50, D]")
        return actions
