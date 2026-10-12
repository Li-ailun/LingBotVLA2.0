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
        "request_10-delay_10-switch_20-horizon_10-xfade_4-ema_0p36/click_alarmclock"
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
        "request_10-delay_10-switch_20-horizon_10-xfade_4-ema_0p36/open_microwave"
    ) in output
