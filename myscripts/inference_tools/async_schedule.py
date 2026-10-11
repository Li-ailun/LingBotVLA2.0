"""Deterministic q16/K20 scheduling decisions used by the ROS runtime."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AsyncChunkSchedule:
    prefetch_step: int = 16
    requery_step: int = 20
    horizon: int = 50

    def should_prefetch(self, action_index, *, inflight, ready):
        return action_index >= self.prefetch_step and not inflight and not ready

    def should_promote(self, action_index, *, ready):
        return action_index >= self.requery_step and ready

    def should_hold(self, action_index, *, ready):
        return action_index >= self.horizon and not ready
