"""Optional wall-clock pacing for live visualization; disabled by default."""

import time


class RealTimePacer:
    def __init__(self, timestep, *, enabled=False, monotonic=time.monotonic, sleep=time.sleep):
        self.timestep = float(timestep)
        self.enabled = bool(enabled)
        self._monotonic = monotonic
        self._sleep = sleep
        self._deadline = None

    def reset(self):
        self._deadline = None

    def after_step(self):
        if not self.enabled:
            return
        now = self._monotonic()
        if self._deadline is None:
            self._deadline = now
        self._deadline += self.timestep
        remaining = self._deadline - now
        if remaining > 0:
            self._sleep(remaining)
        elif remaining < -max(0.25, self.timestep * 10):
            # Rendering/inference fell far behind: resync instead of accumulating debt.
            self._deadline = now
