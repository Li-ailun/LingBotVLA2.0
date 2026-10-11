from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "experiment/robotwin/start_robotwin_infer_and_eval.sh"


def run_launcher(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(LAUNCHER), "--validate_only", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_validate_only_reports_official_default_of_100_episodes() -> None:
    result = run_launcher()

    assert result.returncode == 0, result.stderr
    assert "num_episodes=100" in result.stdout
    assert "client_override=--num_episodes 100" in result.stdout


def test_validate_only_propagates_explicit_50_episodes() -> None:
    result = run_launcher("--num_episodes", "50")

    assert result.returncode == 0, result.stderr
    assert "num_episodes=50" in result.stdout
    assert "client_override=--num_episodes 50" in result.stdout


def test_validate_only_rejects_non_positive_episode_count() -> None:
    result = run_launcher("--num_episodes", "0")

    assert result.returncode != 0
    assert "num_episodes must be a positive integer" in result.stderr


def test_validate_only_rejects_non_numeric_episode_count() -> None:
    result = run_launcher("--num_episodes", "fifty")

    assert result.returncode != 0
    assert "num_episodes must be a positive integer" in result.stderr


def test_validate_only_propagates_visualization_and_resume_state() -> None:
    result = run_launcher(
        "--task_name", "lift_pot",
        "--render_freq", "1",
        "--initial_episode_count", "19",
        "--initial_success_count", "0",
        "--start_seed", "100046",
        "--resume_run_dir", "/tmp/existing-run",
    )

    assert result.returncode == 0, result.stderr
    assert "task_name=lift_pot" in result.stdout
    assert "render_freq=1" in result.stdout
    assert "initial_episode_count=19" in result.stdout
    assert "initial_success_count=0" in result.stdout
    assert "start_seed=100046" in result.stdout
    assert "resume_run_dir=/tmp/existing-run" in result.stdout
