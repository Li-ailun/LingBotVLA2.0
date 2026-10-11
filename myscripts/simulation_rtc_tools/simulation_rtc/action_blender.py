from __future__ import annotations

import numpy as np


def _finite(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def crossfade_chunks(old_chunk, old_index: int, new_chunk, *, steps: int = 4,
                     excluded_indices: tuple[int, ...] = ()) -> np.ndarray:
    old = _finite(old_chunk, "old_chunk")
    new = _finite(new_chunk, "new_chunk").copy()
    if old.ndim != 2 or new.ndim != 2 or old.shape[1] != new.shape[1]:
        raise ValueError("chunks must be 2D with equal action dimensions")
    if not 0 <= int(old_index) <= len(old):
        raise ValueError("old_index is out of range")
    count = min(int(steps), len(old) - int(old_index), len(new))
    if count <= 0:
        return new
    t = np.arange(1, count + 1, dtype=np.float32) / float(count + 1)
    weight = (3.0 * t**2 - 2.0 * t**3)[:, None]
    old_tail = old[int(old_index):int(old_index) + count]
    new[:count] = old_tail * (1.0 - weight) + new[:count] * weight
    for index in excluded_indices:
        if not 0 <= index < new.shape[1]:
            raise ValueError("excluded action index is out of range")
        new[:count, index] = _finite(new_chunk, "new_chunk")[:count, index]
    return new


def ema_action(previous, current, *, alpha: float = 0.36,
               excluded_indices: tuple[int, ...] = ()) -> np.ndarray:
    old = _finite(previous, "previous")
    new = _finite(current, "current")
    if old.shape != new.shape or old.ndim != 1:
        raise ValueError("EMA actions must be equal 1D vectors")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    result = (1.0 - alpha) * old + alpha * new
    for index in excluded_indices:
        if not 0 <= index < len(result):
            raise ValueError("excluded action index is out of range")
        result[index] = new[index]
    return result
