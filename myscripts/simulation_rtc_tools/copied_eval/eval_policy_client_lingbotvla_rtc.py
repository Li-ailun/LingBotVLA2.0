"""Isolated RTC evaluation client.

Episode construction and accounting are retained from RoboTwin's
``eval_policy_client_lingbotvla.py``. Only the policy action loop is replaced.
The source file is copied to ``RoboTwin_lingbot/script/`` under this unique
name by the isolated launcher; it never overwrites the synchronous client.
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

import script.eval_policy_client_lingbotvla as upstream
from envs.utils.create_actor import UnStableError


TOOL_ROOT = Path(os.environ["LINGBOT_RTC_TOOL_ROOT"]).resolve()
PROJECT_ROOT = Path(os.environ["LINGBOT_RTC_PROJECT_ROOT"]).resolve()
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation_rtc.async_worker import AsyncPolicyWorker
from simulation_rtc.config import RTCConfig, RobotActionLayout
from simulation_rtc.eval_loop import run_rtc_episode
from simulation_rtc.metrics import JSONLTraceWriter
from myscripts.video_tools.fresh_frame_recording import install_fresh_frame_recording


class RoboTwinWebsocketTransport:
    """Worker-owned adapter around the official msgpack websocket client."""

    def __init__(self, port: int):
        from script.deploy_rtc.websocket_client_policy import WebsocketClientPolicy
        self.client = WebsocketClientPolicy(port=port)

    def infer(self, payload: dict, timeout_s: float) -> dict:
        from script.deploy_rtc.msgpack_numpy import unpackb
        self.client._ws.send(self.client._packer.pack(payload))
        response = self.client._ws.recv(timeout=timeout_s)
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return unpackb(response)

    def close(self) -> None:
        self.client._ws.close()


def eval_policy(task_name, TASK_ENV, args, unused_model, st_seed, test_num=100,
                video_size=None, video_fps="10", instruction_type=None,
                usr_args=None):
    usr_args = usr_args or {}
    install_fresh_frame_recording(TASK_ENV)
    try:
        unused_model._ws.close()
    except Exception:
        pass
    config = RTCConfig.from_yaml(Path(os.environ["LINGBOT_RTC_CONFIG"]))
    robot_config = PROJECT_ROOT / "models/lingbot_action_step2000/configs/robot_configs/robotwin.yaml"
    layout = RobotActionLayout.from_yaml(robot_config)
    trace_path = Path(usr_args["output_dir"]) / task_name / "rtc_trace.jsonl"
    trace = JSONLTraceWriter(trace_path)
    worker = AsyncPolicyWorker(
        lambda: RoboTwinWebsocketTransport(int(usr_args["port"])),
        layout, config.request_timeout_s,
    )
    worker.start()

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

    try:
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
            results = upstream.generate_episode_descriptions(
                args["task_name"], [episode_info["info"]], test_num,
            )
            instruction = np.random.choice(results[0][instruction_type])
            TASK_ENV.set_instruction(instruction=instruction)

            if TASK_ENV.eval_video_path is not None:
                ffmpeg = subprocess.Popen([
                    "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                    "-pixel_format", "rgb24", "-video_size", video_size,
                    "-framerate", video_fps, "-i", "-", "-pix_fmt", "yuv420p",
                    "-vcodec", "libx264", "-crf", "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ], stdin=subprocess.PIPE)
                TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

            episode_id = (int(now_seed) << 20) + int(now_id)
            rtc_result = run_rtc_episode(
                TASK_ENV, worker, config, layout, trace,
                episode_id=episode_id, robo_name=usr_args["robo_name"],
            )
            succ = rtc_result.success

            if TASK_ENV.eval_video_path is not None:
                TASK_ENV._del_eval_video_ffmpeg()
            result_tag = "success" if succ else "failure"
            old_name = f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4"
            new_name = f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}_{result_tag}.mp4"
            if os.path.exists(old_name):
                os.rename(old_name, new_name)
                print(f"Video saved: {new_name}")
            if succ:
                TASK_ENV.suc += 1
                print("\033[92mSuccess!\033[0m")
            else:
                print("\033[91mFail!\033[0m")
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
    finally:
        worker.close(config.request_timeout_s + 1.0)
    return now_seed, TASK_ENV.suc


if __name__ == "__main__":
    upstream.eval_policy = eval_policy
    upstream.main(upstream.parse_args_and_config())
