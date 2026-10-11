"""SAPIEN 1x physics pacing plus a 30 FPS live viewer refresh."""

import os
import threading
import time


if os.environ.get("ROBOTWIN_REALTIME_VISUALIZATION") == "1":
    try:
        import sapien.core as sapien
        from sapien.utils.viewer import Viewer
    except ImportError:
        pass
    else:
        _viewers = {}
        _original_set_scene = Viewer.set_scene
        _original_step = sapien.Scene.step
        _state = threading.local()

        def _set_scene(viewer, scene, *args, **kwargs):
            result = _original_set_scene(viewer, scene, *args, **kwargs)
            _viewers[id(scene)] = viewer
            return result

        def _step(scene, *args, **kwargs):
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
                now = time.monotonic()
            elif remaining < -max(0.25, timestep * 10):
                deadline = now
            _state.deadline = deadline

            viewer = _viewers.get(id(scene))
            next_render = getattr(_state, "next_render", 0.0)
            if viewer is not None and now >= next_render:
                try:
                    scene.update_render()
                    viewer.render()
                except Exception:
                    _viewers.pop(id(scene), None)
                _state.next_render = now + 1.0 / 30.0
            return result

        _set_scene._robotwin_realtime_wrapped = True
        _step._robotwin_realtime_wrapped = True
        Viewer.set_scene = _set_scene
        sapien.Scene.step = _step
