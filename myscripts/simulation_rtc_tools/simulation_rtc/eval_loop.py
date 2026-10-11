from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

import numpy as np

from .action_blender import crossfade_chunks, ema_action
from .config import RTCConfig, RobotActionLayout
from .protocol import build_bootstrap_request, build_rtc_request
from .scheduler import RTCScheduler


@dataclass(frozen=True)
class EpisodeRTCResult:
    success: bool
    action_count: int
    request_count: int
    pause_seconds: float


def format_robotwin_observation(observation: dict[str, Any], instruction: str | None = None) -> dict[str, Any]:
    if "observation.state" in observation:
        result = dict(observation)
        if instruction is not None:
            result["task"] = instruction
        return result
    return {
        "observation.images.cam_high": observation["observation"]["head_camera"]["rgb"],
        "observation.images.cam_left_wrist": observation["observation"]["left_camera"]["rgb"],
        "observation.images.cam_right_wrist": observation["observation"]["right_camera"]["rgb"],
        "observation.state": observation["joint_action"]["vector"],
        "task": instruction,
    }


def _raise_worker_error(result) -> None:
    if result is None:
        raise RuntimeError("policy worker returned no result")
    if result.error is not None:
        raise RuntimeError("policy worker inference failed") from result.error


def run_rtc_episode(task_env, worker, config: RTCConfig, layout: RobotActionLayout,
                    trace_writer, *, episode_id: int, robo_name: str,
                    observation_formatter: Callable[[dict, str | None], dict] = format_robotwin_observation
                    ) -> EpisodeRTCResult:
    scheduler = RTCScheduler(config)
    scheduler.start_episode(episode_id)
    worker.reset_episode(episode_id)
    instruction = task_env.get_instruction() if hasattr(task_env, "get_instruction") else None
    initial = observation_formatter(task_env.get_obs(), instruction)
    initial.update(reset_policy=True, robo_name=robo_name)
    bootstrap = build_bootstrap_request(initial, episode_id=episode_id, request_id=0)
    worker.submit(bootstrap)
    boot_result = worker.poll()
    if boot_result is None:
        boot_result = worker.wait(config.request_timeout_s)
    _raise_worker_error(boot_result)
    scheduler.install_bootstrap(boot_result.response.actions)
    trace_writer.write("bootstrap", episode_id=episode_id,
                       policy_step=int(task_env.take_action_cnt),
                       server_timing=boot_result.response.server_timing)

    next_request_id = 1
    request_count = 0
    pause_seconds = 0.0
    previous_action = None
    try:
        while task_env.take_action_cnt < task_env.step_lim and not task_env.eval_success:
            arrived = worker.poll()
            if arrived is not None:
                _raise_worker_error(arrived)
                scheduler.accept_response(arrived.response)

            decision = scheduler.before_action()
            if decision.request_due:
                fresh = observation_formatter(task_env.get_obs(), instruction)
                request = build_rtc_request(
                    fresh, scheduler.current_chunk, episode_id=episode_id,
                    request_id=next_request_id, old_index=decision.chunk_index, config=config,
                )
                worker.submit(request)
                scheduler.mark_request_submitted(next_request_id)
                trace_writer.write("request", episode_id=episode_id,
                                   request_id=next_request_id,
                                   policy_step=int(task_env.take_action_cnt),
                                   chunk_index=decision.chunk_index,
                                   eligible_policy_step=int(task_env.take_action_cnt) + config.fixed_delay_steps)
                next_request_id += 1
                request_count += 1

            if decision.should_wait:
                wait_start = time.monotonic()
                trace_writer.write("wait", episode_id=episode_id,
                                   request_id=scheduler.inflight_request_id,
                                   policy_step=int(task_env.take_action_cnt))
                result = worker.wait(config.request_timeout_s)
                pause = time.monotonic() - wait_start
                pause_seconds += pause
                _raise_worker_error(result)
                if not scheduler.accept_response(result.response):
                    raise RuntimeError("worker returned stale RTC response")
                continue

            action = decision.action
            if decision.promoted:
                blended = crossfade_chunks(decision.prior_chunk, decision.prior_index,
                                           scheduler.current_chunk,
                                           steps=config.crossfade_steps,
                                           excluded_indices=layout.gripper_indices)
                scheduler.current_chunk = blended
                action = blended[0].copy()
                trace_writer.write("promotion", episode_id=episode_id,
                                   policy_step=int(task_env.take_action_cnt),
                                   chunk_index=decision.prior_index,
                                   crop_steps=decision.crop_steps,
                                   remaining_old_actions=len(decision.prior_chunk) - decision.prior_index)
            if action is None or np.asarray(action).shape != (layout.action_dim,):
                raise ValueError("scheduler produced an invalid physical action")
            if not np.isfinite(action).all():
                raise ValueError("scheduler produced a non-finite physical action")
            if previous_action is not None:
                action = ema_action(previous_action, action, alpha=config.ema_alpha,
                                    excluded_indices=layout.gripper_indices)
            task_env.take_action(action)
            previous_action = np.asarray(action, dtype=np.float32).copy()
            scheduler.after_action()
    except BaseException:
        scheduler.fail_episode()
        trace_writer.write("episode_error", episode_id=episode_id,
                           policy_step=int(task_env.take_action_cnt))
        raise
    finally:
        if scheduler.inflight_request_id is not None:
            leftover = worker.poll()
            if leftover is None:
                leftover = worker.wait(config.request_timeout_s)
            _raise_worker_error(leftover)
            trace_writer.write("discard_after_episode", episode_id=episode_id, request_id=leftover.request.request_id, policy_step=int(task_env.take_action_cnt))
        if scheduler.state.name not in {"FAILED", "EPISODE_DONE"}:
            scheduler.finish_episode()


    result = EpisodeRTCResult(bool(task_env.eval_success), int(task_env.take_action_cnt),
                              request_count, pause_seconds)
    trace_writer.write("episode_summary", episode_id=episode_id, success=result.success,
                       action_count=result.action_count, request_count=result.request_count,
                       pause_seconds=result.pause_seconds)
    return result
