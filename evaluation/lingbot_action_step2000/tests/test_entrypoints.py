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
    assert "evaluation/video_comparison/non_rtc/clean" in output
    assert "ROBOTWIN_REALTIME_VISUALIZATION=1" not in output
    assert f"PYTHONPATH={ROOT.parent.parent}:{ROOT.parent.parent / 'lingbot-vla-v2'}" in output


def test_video_launcher_deploys_an_isolated_fresh_frame_client():
    launcher = (ROOT / "video_eval_launcher/start_robotwin_video_infer_and_eval.sh").read_text()
    assert "eval_policy_client_lingbotvla_video.py" in launcher
    assert 'eval_client_dst="${eval_workdir}/script/eval_policy_client_lingbotvla_video.py"' in launcher
    assert 'eval_client_dst="${eval_workdir}/script/eval_policy_client_lingbotvla.py"' not in launcher
