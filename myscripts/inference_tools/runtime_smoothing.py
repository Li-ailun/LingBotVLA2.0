"""Pure NumPy form of the RLinf XFade4 and EMA0.36 runtime filters."""

import numpy as np


def crossfade_chunks(old_chunk, old_index, new_chunk, *, steps=4):
    old = np.asarray(old_chunk, dtype=np.float32)
    new = np.asarray(new_chunk, dtype=np.float32).copy()
    count = min(int(steps), len(old) - int(old_index), len(new))
    if count <= 0:
        return new
    t = np.linspace(0.0, 1.0, count, dtype=np.float32)
    weight = (3 * t**2 - 2 * t**3)[:, None]
    new[:count] = old[int(old_index):int(old_index) + count] * (1 - weight) + new[:count] * weight
    return new


def ema_action(previous, current, *, alpha=0.36, excluded=(7, 15)):
    previous = np.asarray(previous, dtype=np.float32)
    current = np.asarray(current, dtype=np.float32)
    result = (1.0 - alpha) * previous + alpha * current
    result[list(excluded)] = current[list(excluded)]
    return result
