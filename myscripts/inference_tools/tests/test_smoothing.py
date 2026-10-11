import numpy as np

from myscripts.inference_tools.runtime_smoothing import crossfade_chunks, ema_action


def test_four_step_smoothstep_crossfade():
    old = np.zeros((50, 20), np.float32)
    new = np.ones((50, 20), np.float32)
    blended = crossfade_chunks(old, 16, new, steps=4)
    expected = np.array([0.0, 7/27, 20/27, 1.0], np.float32)
    assert np.allclose(blended[:4, 0], expected)


def test_ema_excludes_grippers():
    previous = np.zeros(20, np.float32)
    current = np.ones(20, np.float32)
    result = ema_action(previous, current, alpha=0.36, excluded=(7, 15))
    assert np.allclose(result[[0, 16]], 0.36)
    assert np.allclose(result[[7, 15]], 1.0)
