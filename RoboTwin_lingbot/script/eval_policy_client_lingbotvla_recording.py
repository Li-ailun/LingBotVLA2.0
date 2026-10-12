"""Isolated synchronous RoboTwin client with provenance sidecar logging.

The action loop matches the official LingBot-VLA client.  This copy only
counts inference calls and records metadata after RoboTwin has decided the
episode result; it never replaces the official client source file.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROBOTWIN_ROOT = Path(__file__).resolve().parents[1]
if str(ROBOTWIN_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOTWIN_ROOT))

PROJECT_ROOT = Path(os.environ["LINGBOT_PROJECT_ROOT"]).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import script.eval_policy_client_lingbotvla as upstream
import script.deploy_rtc.websocket_client_policy as isolated_websocket_client
from envs.utils.create_actor import UnStableError
from evaluation.eval_provenance import EvaluationRecorder

# upstream.main imports this name lazily. Route only this copied process to the
# isolated helper package deployed by our launcher; no official file is copied
# over or changed.
sys.modules["script.deploy.websocket_client_policy"] = isolated_websocket_client


def _video_path(task_env, episode_index: int, result_tag: str) -> str | None:
    if task_env.eval_video_path is None:
        return None
    return str(Path(task_env.eval_video_path) / f"episode{episode_index}_{result_tag}.mp4")


def eval_policy(task_name, TASK_ENV, args, model, st_seed, test_num=100,
                video_size=None, video_fps="10", instruction_type=None,
                usr_args=None):
    usr_args = usr_args or {}
    recorder = EvaluationRecorder(
        Path(usr_args["output_dir"]).resolve().parent, task=task_name
    )
    expert_check = True
    initial_episode_count = int(usr_args.get("initial_episode_count", 0))
    initial_success_count = int(usr_args.get("initial_success_count", 0))
    TASK_ENV.suc = initial_success_count
    TASK_ENV.test_num = initial_episode_count
    now_id = initial_episode_count
    succ_seed = initial_episode_count
    now_seed = st_seed
    clear_cache_freq = args["clear_cache_freq"]
    args["eval_mode"] = True

    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0
        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError:
                TASK_ENV.close_env(); now_seed += 1; args["render_freq"] = render_freq
                continue
            except Exception as error:
                TASK_ENV.close_env(); now_seed += 1; args["render_freq"] = render_freq
                print(f"error occurs ! {error}")
                continue
        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
        else:
            now_seed += 1; args["render_freq"] = render_freq
            continue
        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        descriptions = upstream.generate_episode_descriptions(
            args["task_name"], [episode_info["info"]], test_num,
        )
        instruction = np.random.choice(descriptions[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)

        episode_index = int(TASK_ENV.test_num)
        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen([
                "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                "-pixel_format", "rgb24", "-video_size", video_size,
                "-framerate", video_fps, "-i", "-", "-pix_fmt", "yuv420p",
                "-vcodec", "libx264", "-crf", "23",
                f"{TASK_ENV.eval_video_path}/episode{episode_index}.mp4",
            ], stdin=subprocess.PIPE)
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        path_to_pi_model = usr_args.get("new_ckpt_path")
        model.infer(dict(reset=True, robo_name=usr_args["robo_name"],
                         path_to_pi_model=path_to_pi_model))
        succ = False
        inference_calls = 0
        server_timings = []
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim and not succ:
            observation = TASK_ENV.get_obs()
            formatted = {
                "observation.images.cam_high": observation["observation"]["head_camera"]["rgb"],
                "observation.images.cam_left_wrist": observation["observation"]["left_camera"]["rgb"],
                "observation.images.cam_right_wrist": observation["observation"]["right_camera"]["rgb"],
                "observation.state": observation["joint_action"]["vector"],
                "task": TASK_ENV.get_instruction(),
            }
            response = model.infer(formatted)
            inference_calls += 1
            timing = response.get("server_timing")
            if timing is not None:
                try:
                    server_timings.append(float(timing))
                except (TypeError, ValueError):
                    pass
            action = response["action"]
            if len(action.shape) == 2:
                for act in action:
                    TASK_ENV.take_action(act)
                    if TASK_ENV.eval_success:
                        succ = True
                        break
            else:
                TASK_ENV.take_action(action)
                if TASK_ENV.eval_success:
                    succ = True
            print(f"infer time {timing}")

        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()
        result_tag = "success" if succ else "failure"
        old_name = f"{TASK_ENV.eval_video_path}/episode{episode_index}.mp4"
        new_name = _video_path(TASK_ENV, episode_index, result_tag)
        if new_name is not None and os.path.exists(old_name):
            os.rename(old_name, new_name)
            print(f"Video saved: {new_name}")

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        recorder.record_episode({
            "episode_index": episode_index,
            "environment_seed": int(now_seed),
            "success": bool(succ),
            "instruction": str(instruction),
            "policy_steps": int(TASK_ENV.take_action_cnt),
            "inference_calls": inference_calls,
            "server_timing_count": len(server_timings),
            "server_timing_mean": (
                sum(server_timings) / len(server_timings) if server_timings else None
            ),
            "mode": "non_rtc",
            "inference_port": int(usr_args["port"]),
            "video_path": new_name if new_name and os.path.exists(new_name) else None,
        })

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))
        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()
        TASK_ENV.test_num += 1
        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | "
            f"\033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => "
            f"\033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, "
            f"current seed: \033[90m{now_seed}\033[0m\n"
        )
        now_seed += 1
    return now_seed, TASK_ENV.suc


if __name__ == "__main__":
    upstream.eval_policy = eval_policy
    upstream.main(upstream.parse_args_and_config())
