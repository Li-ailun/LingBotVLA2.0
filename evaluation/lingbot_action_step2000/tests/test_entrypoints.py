import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def dry(script, *args):
    result = subprocess.run(["bash", str(ROOT / script), *args, "--dry-run"], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_formal_clean_is_50_tasks_by_100_and_new_checkpoint():
    output = dry("run_formal_clean.sh")
    assert "models/lingbot_action_step2000" in output
    assert "--num_tasks 50" in output
    assert "--num_episodes 100" in output
    assert "--task_config demo_clean" in output
    assert "baseline_base_model_adapter" not in output


def test_non_rtc_selects_full5k_without_copying_checkpoint():
    output = dry(
        "run_eval.sh", "demo_clean", "--model", "full5k", "--precision", "bf16"
    )
    assert "checkpoints/global_step_5000/hf_ckpt" in output
    assert "LINGBOT_MODEL_ASSET_ROOT=" in output
    assert "models/lingbot_full_5k_6_60" in output
    assert "results/full5k/bf16/clean" in output
    assert "--use_bf16 True" in output
    assert "--use_fp32 False" in output
    assert "eval_policy_client_lingbotvla_recording.py" in output
    assert "LINGBOT_EVAL_MODE=non_rtc" in output
    assert "LINGBOT_PROVENANCE_ENABLED=1" in output


def test_non_rtc_supports_fp32_and_resume_directory():
    resume = "/tmp/full5k-non-rtc-resume"
    output = dry(
        "run_eval.sh", "demo_clean", "--model", "full5k", "--precision", "fp32",
        "--resume-run-dir", resume, "--resume-task", "place_object_stand",
        "--initial-episode-count", "10", "--initial-success-count", "2",
        "--start-seed", "100011",
    )
    assert "--use_bf16 False" in output
    assert "--use_fp32 True" in output
    assert f"--resume_run_dir {resume}" in output
    assert "--task_name place_object_stand" in output
    assert "--initial_episode_count 10" in output
    assert "--initial_success_count 2" in output
    assert "--start_seed 100011" in output
    assert "--num_tasks 50" in output
    assert "LINGBOT_PRECISION=fp32" in output


def test_non_rtc_rejects_unknown_model():
    result = subprocess.run(
        ["bash", str(ROOT / "run_eval.sh"), "demo_clean", "--model", "missing", "--dry-run"],
        text=True, capture_output=True,
    )
    assert result.returncode != 0
    assert "Unknown model" in result.stderr


def test_non_rtc_can_select_physical_gpu_without_changing_default_command():
    selected = dry(
        "run_eval.sh", "demo_clean", "--model", "full5k", "--gpu", "8"
    )
    default = dry("run_eval.sh", "demo_clean", "--model", "full5k")

    assert "--gpu_ids 8" in selected
    assert "LINGBOT_GPU_IDS=8" in selected
    assert "--gpu_ids" not in default


def test_non_rtc_rejects_non_numeric_gpu_id():
    result = subprocess.run(
        [
            "bash", str(ROOT / "run_eval.sh"), "demo_clean",
            "--gpu", "eight", "--dry-run",
        ],
        text=True, capture_output=True,
    )
    assert result.returncode != 0
    assert "GPU" in result.stderr


def test_non_rtc_accepts_arbitrary_non_contiguous_gpu_list():
    output = dry(
        "run_eval.sh", "demo_clean", "--model", "full5k",
        "--gpu-ids", "1,4,7",
    )
    assert "--gpu_ids 1\\,4\\,7" in output


def test_policy_server_resolves_auxiliary_assets_independently_of_weight_layout():
    text = (ROOT / "policy_server.py").read_text(encoding="utf-8")
    assert 'os.environ.get("LINGBOT_MODELS_ROOT"' in text
    assert 'models_root / "lingbot-vla-v2-6b"' in text


def test_realtime_is_single_task_and_one_episode():
    output = dry("run_realtime_visualization.sh", "lift_pot", "1")
    assert "--task_name lift_pot" in output
    assert "--render_freq 1" in output
    assert "--num_tasks 1" in output
    assert "--num_episodes 1" in output
    assert "ROBOTWIN_REALTIME_VISUALIZATION=1" in output


def test_headless_video_eval_records_without_gui_or_realtime():
    output = dry(
        "run_video_eval.sh", "demo_clean", "--task", "click_alarmclock", "--episodes", "1"
    )
    assert "--video" in output
    assert "--render_freq 0" in output
    assert "--task_name click_alarmclock" in output
    assert "--num_episodes 1" in output
    assert "evaluation/video_comparison/step2000/bf16/non_rtc/clean/click_alarmclock" in output
    assert "ROBOTWIN_REALTIME_VISUALIZATION=1" not in output
    assert f"PYTHONPATH={ROOT.parent.parent}:{ROOT.parent.parent / 'lingbot-vla-v2'}" in output


def test_full5k_video_path_identifies_model_precision_mode_and_task():
    output = dry(
        "run_video_eval.sh", "demo_clean", "--task", "open_microwave",
        "--episodes", "1", "--model", "full5k", "--precision", "fp32",
    )
    assert "evaluation/video_comparison/full5k/fp32/non_rtc/clean/open_microwave" in output


def test_video_launcher_deploys_an_isolated_fresh_frame_client():
    launcher = (ROOT / "video_eval_launcher/start_robotwin_video_infer_and_eval.sh").read_text()
    assert "eval_policy_client_lingbotvla_video.py" in launcher
    assert 'eval_client_dst="${eval_workdir}/script/eval_policy_client_lingbotvla_video.py"' in launcher
    assert 'eval_client_dst="${eval_workdir}/script/eval_policy_client_lingbotvla.py"' not in launcher
