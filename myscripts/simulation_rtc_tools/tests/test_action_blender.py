import numpy as np

from simulation_rtc.action_blender import crossfade_chunks, ema_action


def test_crossfade_matches_smoothstep_xfade4():
    old = np.ones((50, 14), np.float32)
    new = np.zeros((50, 14), np.float32)
    blended = crossfade_chunks(old, 20, new, steps=4, excluded_indices=(6, 13))
    t = np.arange(1, 5, dtype=np.float32) / 5.0
    expected = 1.0 - (3 * t**2 - 2 * t**3)
    np.testing.assert_allclose(blended[:4, 0], expected)
    assert np.all(blended[:4, 6] == 0) and np.all(blended[:4, 13] == 0)
    assert np.all(blended[4:] == 0)


def test_ema_excludes_derived_gripper_indices():
    previous = np.zeros(14, np.float32)
    current = np.ones(14, np.float32)
    result = ema_action(previous, current, alpha=0.36, excluded_indices=(6, 13))
    assert result[0] == np.float32(0.36)
    assert result[6] == 1.0 and result[13] == 1.0


def test_blenders_reject_non_finite_or_shape_mismatch():
    old = np.zeros((50, 14), np.float32)
    new = np.zeros((50, 13), np.float32)
    try:
        crossfade_chunks(old, 45, new, steps=4)
    except ValueError:
        pass
    else:
        raise AssertionError("shape mismatch must fail")
    current = np.ones(14, np.float32)
    current[2] = np.nan
    try:
        ema_action(np.zeros(14), current, alpha=0.36, excluded_indices=(6, 13))
    except ValueError:
        pass
    else:
        raise AssertionError("NaN must fail")
