from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
TOOL = Path(__file__).resolve().parents[1]


def run(*args):
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=True).stdout


def test_one_task_dry_run_uses_unique_rtc_server_and_client():
    output = run("bash", str(TOOL / "scripts/run_rtc_fast_one_task.sh"), "lift_pot", "--dry-run")
    assert "policy_server_rtc.py" in output
    assert "eval_policy_client_lingbotvla_rtc.py" in output
    assert "--task_name lift_pot" in output
    assert "--num_episodes 1" in output
    assert "--no_video" in output


def test_batch_dry_run_defaults_to_clean_50_by_50():
    output = run("bash", str(TOOL / "scripts/run_rtc_fast_50x50.sh"), "--dry-run")
    assert "--num_tasks 50" in output
    assert "--num_episodes 50" in output
    assert "demo_clean" in output
    assert "--instruction_type unseen" in output
    assert "LINGBOT_RTC_REQUEST_INDEX=10" in output
    assert "LINGBOT_RTC_FIXED_DELAY_STEPS=10" in output
    assert "LINGBOT_RTC_EXECUTION_HORIZON=10" in output
    assert "LINGBOT_RTC_CROSSFADE_STEPS=4" in output
    assert "LINGBOT_RTC_EMA_ALPHA=0.1" in output


def test_rtc_batch_supports_randomized_setting_and_separate_base_port():
    output = run(
        "bash", str(TOOL / "scripts/run_rtc_fast_50x50.sh"),
        "--setting", "demo_randomized", "--base-port", "9340", "--dry-run",
    )
    assert "--task_config demo_randomized" in output
    assert "results/step2000/bf16/randomized" in output
    assert "--start_port 9340" in output
    assert "--instruction_type unseen" in output


def test_rtc_batch_rejects_unknown_setting():
    result = subprocess.run(
        [
            "bash", str(TOOL / "scripts/run_rtc_fast_50x50.sh"),
            "--setting", "random", "--dry-run",
        ],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert result.returncode == 2
    assert "setting must be demo_clean or demo_randomized" in result.stderr


def test_rtc_batch_selects_full5k_and_separates_results():
    output = run(
        "bash", str(TOOL / "scripts/run_rtc_fast_50x50.sh"),
        "--model", "full5k", "--precision", "bf16", "--dry-run",
    )
    assert "checkpoints/global_step_5000/hf_ckpt" in output
    assert "LINGBOT_MODEL_ASSET_ROOT=" in output
    assert "models/lingbot_full_5k_6_60" in output
    assert "results/full5k/bf16/clean" in output
    assert "--use_bf16 True" in output
    assert "--use_fp32 False" in output
    assert "LINGBOT_EVAL_MODE=rtc" in output
    assert "LINGBOT_PROVENANCE_ENABLED=1" in output


def test_rtc_batch_selects_full15k_and_separates_results():
    output = run(
        "bash", str(TOOL / "scripts/run_rtc_fast_50x50.sh"),
        "--model", "full15k", "--precision", "bf16", "--dry-run",
    )
    assert "--model_path" in output
    assert "models/lingbot_15K" in output
    assert "models/lingbot_full_5k_6_60" in output
    assert "results/full15k/bf16/clean" in output


def test_full15k_rtc_randomized_fp32_uses_new_result_root_and_port():
    output = run(
        "bash", str(TOOL / "scripts/run_rtc_fast_50x50.sh"),
        "--model", "full15k", "--precision", "fp32",
        "--setting", "demo_randomized", "--episodes", "50",
        "--gpu-ids", "0,6,8", "--base-port", "9340", "--dry-run",
    )
    assert "models/lingbot_15K" in output
    assert "models/lingbot_full_5k_6_60" in output
    assert "results/full15k/fp32/randomized" in output
    assert "--task_config demo_randomized" in output
    assert "--num_episodes 50" in output
    assert "--start_port 9340" in output
    assert "--use_bf16 False" in output
    assert "--use_fp32 True" in output
    assert "--gpu_ids 0\\,6\\,8" in output


def test_full15k_rtc_video_uses_new_result_root_and_replay_seed():
    output = run(
        "bash", str(TOOL / "scripts/run_rtc_video_eval.sh"),
        "--model", "full15k", "--precision", "bf16",
        "--setting", "demo_clean", "--task", "place_empty_cup",
        "--episodes", "1", "--start-seed", "100012", "--dry-run",
    )
    assert "models/lingbot_15K" in output
    assert "models/lingbot_full_5k_6_60" in output
    assert "evaluation/video_comparison/full15k/bf16/rtc/clean/" in output
    assert "place_empty_cup" in output
    assert "--start_seed 100012" in output
    assert "--instruction_type unseen" in output


def test_rtc_batch_can_explicitly_override_instruction_type_to_seen():
    output = run(
        "bash", str(TOOL / "scripts/run_rtc_fast_50x50.sh"),
        "--setting", "demo_clean", "--instruction-type", "seen", "--dry-run",
    )
    assert "--instruction_type seen" in output


def test_rtc_batch_rejects_invalid_instruction_type():
    result = subprocess.run(
        [
            "bash", str(TOOL / "scripts/run_rtc_fast_50x50.sh"),
            "--instruction-type", "invalid", "--dry-run",
        ],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert result.returncode == 2
    assert "instruction type must be seen or unseen" in result.stderr


def test_rtc_batch_supports_fp32_and_resume_directory():
    resume = "/tmp/full5k-rtc-resume"
    output = run(
        "bash", str(TOOL / "scripts/run_rtc_fast_50x50.sh"),
        "--model", "full5k", "--precision", "fp32",
        "--resume-run-dir", resume, "--resume-task", "place_object_stand",
        "--initial-episode-count", "10", "--initial-success-count", "2",
        "--start-seed", "100011", "--dry-run",
    )
    assert "--use_bf16 False" in output
    assert "--use_fp32 True" in output
    assert f"--resume_run_dir {resume}" in output
    assert "--task_name place_object_stand" in output
    assert "--initial_episode_count 10" in output
    assert "--initial_success_count 2" in output
    assert "--start_seed 100011" in output


def test_rtc_resume_can_select_physical_gpu_explicitly():
    output = run(
        "bash", str(TOOL / "scripts/run_rtc_fast_50x50.sh"),
        "--model", "full5k", "--gpu", "8",
        "--resume-run-dir", "/tmp/run", "--resume-task", "place_object_basket",
        "--initial-episode-count", "10", "--initial-success-count", "0",
        "--start-seed", "100017", "--dry-run",
    )
    assert "--gpu_ids 8" in output
    assert "LINGBOT_GPU_IDS=8" in output


def test_rtc_parallel_resume_passes_resume_all_with_arbitrary_gpu_list():
    output = run(
        "bash", str(TOOL / "scripts/run_rtc_fast_50x50.sh"),
        "--model", "full15k", "--gpu-ids", "1,3,4",
        "--resume-run-dir", "/tmp/run", "--resume-all", "--dry-run",
    )
    assert "--resume_run_dir /tmp/run" in output
    assert "--resume_all" in output
    assert "--gpu_ids 1\\,3\\,4" in output


def test_rtc_parallel_resume_requires_a_run_directory():
    result = subprocess.run(
        [
            "bash", str(TOOL / "scripts/run_rtc_fast_50x50.sh"),
            "--resume-all", "--dry-run",
        ],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert result.returncode == 2
    assert "--resume-all requires --resume-run-dir" in result.stderr


def test_rtc_parallel_resume_rejects_manual_per_task_resume_arguments():
    result = subprocess.run(
        [
            "bash", str(TOOL / "scripts/run_rtc_fast_50x50.sh"),
            "--resume-run-dir", "/tmp/run", "--resume-all",
            "--resume-task", "lift_pot", "--dry-run",
        ],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert result.returncode == 2
    assert "cannot be combined" in result.stderr


def test_low_level_launcher_has_per_task_parallel_resume_state():
    text = (TOOL / "scripts/start_robotwin_rtc_infer_and_eval.sh").read_text()
    assert "--resume_all" in text
    assert "task_resume_episode_count" in text
    assert "task_resume_success_count" in text
    assert "task_resume_start_seed" in text


def test_rtc_batch_accepts_arbitrary_non_contiguous_gpu_list():
    output = run(
        "bash", str(TOOL / "scripts/run_rtc_fast_50x50.sh"),
        "--model", "full5k", "--gpu-ids", "2,5,8", "--dry-run",
    )
    assert "--gpu_ids 2\\,5\\,8" in output
    assert "LINGBOT_GPU_IDS=2\\,5\\,8" in output or "LINGBOT_GPU_IDS=2,5,8" in output


def test_launcher_contains_explicit_gpu_list_mapping_and_validation():
    text = (TOOL / "scripts/start_robotwin_rtc_infer_and_eval.sh").read_text()
    assert "--gpu_ids" in text
    assert 'gpu_id="${selected_gpu_ids[$((slot % num_gpus))]}"' in text
    assert "nvidia-smi -L" in text

    output = run(
        "bash", str(TOOL / "scripts/start_robotwin_rtc_infer_and_eval.sh"),
        "--validate_only", "--gpu_ids", "2,5,8", "--num_tasks", "3",
        "--num_episodes", "1",
    )
    assert "physical_gpu_ids=2,5,8" in output
    assert "num_gpus=3" in output
    assert "instruction_type=unseen" in output


def test_rtc_client_uses_selected_model_asset_root():
    text = (TOOL / "copied_eval/eval_policy_client_lingbotvla_rtc.py").read_text(encoding="utf-8")
    assert 'Path(os.environ["LINGBOT_MODEL_ASSET_ROOT"])' in text
    assert 'models/lingbot_action_step2000/configs/robot_configs/robotwin.yaml' not in text


def test_one_task_dry_run_accepts_runtime_rtc_overrides():
    output = run(
        "bash", str(TOOL / "scripts/run_rtc_fast_one_task.sh"),
        "--task", "lift_pot", "--dry-run",
        "--request-index", "10",
        "--execution-horizon", "10",
        "--fixed-delay-steps", "5",
        "--crossfade-steps", "3",
        "--ema-alpha", "0.5",
    )
    assert "LINGBOT_RTC_REQUEST_INDEX=10" in output
    assert "LINGBOT_RTC_EXECUTION_HORIZON=10" in output
    assert "LINGBOT_EVAL_MODE=rtc" in output
    assert "LINGBOT_RTC_FIXED_DELAY_STEPS=5" in output
    assert "LINGBOT_RTC_CROSSFADE_STEPS=3" in output
    assert "LINGBOT_RTC_EMA_ALPHA=0.5" in output
    assert "--task_name lift_pot" in output
    assert "RTC effective config: request_index=10 fixed_delay_steps=5 execution_horizon=10 crossfade_steps=3 ema_alpha=0.5" in output


def test_batch_dry_run_accepts_runtime_rtc_overrides():
    output = run(
        "bash", str(TOOL / "scripts/run_rtc_fast_50x50.sh"),
        "--dry-run", "--request-index", "10", "--execution-horizon", "10",
    )
    assert "LINGBOT_RTC_REQUEST_INDEX=10" in output
    assert "LINGBOT_RTC_EXECUTION_HORIZON=10" in output


def test_one_task_rejects_invalid_runtime_rtc_combination_before_launch():
    result = subprocess.run(
        [
            "bash", str(TOOL / "scripts/run_rtc_fast_one_task.sh"),
            "--task", "lift_pot", "--dry-run",
            "--request-index", "45", "--execution-horizon", "10",
        ],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert result.returncode != 0
    assert "remaining chunk" in result.stderr


def test_local_launcher_never_targets_synchronous_client_name():
    text = (TOOL / "scripts/start_robotwin_rtc_infer_and_eval.sh").read_text(encoding="utf-8")
    assert 'eval_client_name="${LINGBOT_EVAL_CLIENT_NAME:-eval_policy_client_lingbotvla_rtc.py}"' in text
    assert 'eval_client_dst="${eval_workdir}/script/${eval_client_name}"' in text
    assert 'eval_client_dst="${eval_workdir}/script/eval_policy_client_lingbotvla.py"' not in text
    assert 'deploy_pkg_dst="${eval_workdir}/script/deploy_rtc"' in text
    assert 'deploy_pkg_dst="${eval_workdir}/script/deploy"' not in text


def test_copied_client_adds_robotwin_root_before_importing_script_package():
    text = (TOOL / "copied_eval/eval_policy_client_lingbotvla_rtc.py").read_text(encoding="utf-8")
    assert text.index("ROBOTWIN_ROOT = Path(__file__).resolve().parents[1]") < text.index("import script.eval_policy_client_lingbotvla as upstream")


def test_preflight_does_not_start_episode():
    output = run("bash", str(TOOL / "scripts/preflight.sh"))
    assert "RTC PREFLIGHT OK: no simulator episodes were started" in output


def test_headless_rtc_video_eval_records_without_gui_or_realtime():
    output = run(
        "bash", str(TOOL / "scripts/run_rtc_video_eval.sh"),
        "--task", "click_alarmclock", "--episodes", "1", "--dry-run",
    )
    assert "--video" in output
    assert "--render_freq 0" in output
    assert "--task_name click_alarmclock" in output
    assert "--num_episodes 1" in output
    assert (
        "evaluation/video_comparison/step2000/bf16/rtc/clean/"
        "request_10-delay_10-switch_20-horizon_10-xfade_4-ema_0p1/click_alarmclock"
    ) in output
    assert "ROBOTWIN_REALTIME_VISUALIZATION=1" not in output


def test_rtc_video_path_records_effective_runtime_parameters():
    output = run(
        "bash", str(TOOL / "scripts/run_rtc_video_eval.sh"),
        "--task", "click_alarmclock", "--episodes", "1", "--dry-run",
        "--request-index", "20", "--fixed-delay-steps", "20",
        "--execution-horizon", "20", "--crossfade-steps", "10",
        "--ema-alpha", "0.21",
    )
    tag = "request_20-delay_20-switch_40-horizon_20-xfade_10-ema_0p21"
    assert f"evaluation/video_comparison/step2000/bf16/rtc/clean/{tag}/click_alarmclock" in output
    assert f"RTC result config directory: {tag}" in output


def test_rtc_video_eval_can_replay_a_recorded_environment_seed():
    output = run(
        "bash", str(TOOL / "scripts/run_rtc_video_eval.sh"),
        "--task", "adjust_bottle", "--episodes", "1",
        "--model", "full5k", "--precision", "bf16",
        "--start-seed", "100002", "--dry-run",
    )
    assert "--start_seed 100002" in output
    assert (
        "evaluation/video_comparison/full5k/bf16/rtc/clean/"
        "request_10-delay_10-switch_20-horizon_10-xfade_4-ema_0p1/adjust_bottle"
    ) in output


def test_rtc_client_installs_fresh_frame_recording_hook():
    text = (TOOL / "copied_eval/eval_policy_client_lingbotvla_rtc.py").read_text(encoding="utf-8")
    assert "install_fresh_frame_recording(TASK_ENV)" in text


def test_full5k_rtc_video_path_identifies_model_precision_parameters_and_task():
    output = run(
        "bash", str(TOOL / "scripts/run_rtc_video_eval.sh"),
        "--task", "open_microwave", "--episodes", "1",
        "--model", "full5k", "--precision", "fp32", "--dry-run",
    )
    assert (
        "evaluation/video_comparison/full5k/fp32/rtc/clean/"
        "request_10-delay_10-switch_20-horizon_10-xfade_4-ema_0p1/open_microwave"
    ) in output


def test_rtc_entrypoints_do_not_depend_on_point_flow_sim():
    scripts = [
        "preflight.sh",
        "run_rtc_fast_one_task.sh",
        "run_rtc_fast_50x50.sh",
        "run_rtc_video_eval.sh",
    ]
    for name in scripts:
        text = (TOOL / "scripts" / name).read_text(encoding="utf-8")
        assert "point-flow-sim" not in text


def test_rtc_entrypoints_share_overridable_runtime_paths():
    scripts = [
        "preflight.sh",
        "run_rtc_fast_one_task.sh",
        "run_rtc_fast_50x50.sh",
        "run_rtc_video_eval.sh",
    ]
    for name in scripts:
        text = (TOOL / "scripts" / name).read_text(encoding="utf-8")
        assert "runtime_paths.sh" in text


def test_preflight_validates_selected_model_and_runtime_assets():
    output = run(
        "bash", str(TOOL / "scripts/preflight.sh"),
        "--model", "full5k",
    )
    assert "MODEL ADAPTER VALID: full5k" in output
    assert "ASSET LAYOUT VALID" in output
    assert "RUNTIME IMPORTS VALID" in output


def test_preflight_forwards_selected_model_to_both_rtc_dry_runs():
    text = (TOOL / "scripts/preflight.sh").read_text(encoding="utf-8")
    assert text.count('--model "$MODEL"') == 2
