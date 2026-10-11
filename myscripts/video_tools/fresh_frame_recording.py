"""Recording-only hook that refreshes RoboTwin's cached camera frame per action."""

from __future__ import annotations

from typing import Any


def install_fresh_frame_recording(task_env: Any) -> None:
    """Refresh ``now_obs`` before each recorded policy action, exactly once."""
    if getattr(task_env, "_lingbot_fresh_frame_recording", False):
        return

    original_take_action = task_env.take_action

    def take_action_with_fresh_frame(action, *args, **kwargs):
        if getattr(task_env, "eval_video_path", None) is not None:
            task_env.get_obs()
        return original_take_action(action, *args, **kwargs)

    task_env.take_action = take_action_with_fresh_frame
    task_env._lingbot_fresh_frame_recording = True
