"""Opt-in 1x wall-clock pacing injected only by the realtime viewer launcher."""

import os
import threading
import time


if os.environ.get("ROBOTWIN_REALTIME_VISUALIZATION") == "1":
    try:
        import sapien.core as sapien
    except ImportError:
        # The inference conda environment does not contain SAPIEN. Only the
        # simulator process is expected to install the wrapper.
        pass
    else:
        if not getattr(sapien.Scene.step, "_robotwin_realtime_wrapped", False):
            _original_step = sapien.Scene.step
            _state = threading.local()

            def _realtime_step(scene, *args, **kwargs):
                result = _original_step(scene, *args, **kwargs)
                now = time.monotonic()
                timestep = float(scene.get_timestep())
                deadline = getattr(_state, "deadline", None)
                if deadline is None:
                    deadline = now
                deadline += timestep
                remaining = deadline - now
                if remaining > 0:
                    time.sleep(remaining)
                elif remaining < -max(0.25, timestep * 10):
                    # Inference/rendering caused a long pause. Do not run the
                    # simulation faster later to repay that wall-clock debt.
                    deadline = now
                _state.deadline = deadline
                return result

            _realtime_step._robotwin_realtime_wrapped = True
            sapien.Scene.step = _realtime_step
