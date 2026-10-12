from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "evaluation/prepare_runtime.sh"


def run(*args):
    return subprocess.run(
        ["bash", str(SCRIPT), *map(str, args)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


def test_installs_tracked_task_configs_into_fresh_robotwin(tmp_path):
    robotwin = tmp_path / "RoboTwin_lingbot"
    robotwin.mkdir()

    result = run("--robotwin-dir", robotwin, "--task-config-only")

    assert result.returncode == 0, result.stderr
    assert (robotwin / "task_config/demo_clean.yml").is_file()
    assert (robotwin / "task_config/demo_randomized.yml").is_file()
    assert "TASK CONFIG READY" in result.stdout


def test_rejects_accidentally_nested_assets_directory(tmp_path):
    robotwin = tmp_path / "RoboTwin_lingbot"
    (robotwin / "assets/assets/objects").mkdir(parents=True)

    result = run("--robotwin-dir", robotwin, "--check-only")

    assert result.returncode != 0
    assert "assets/assets" in result.stderr


def test_missing_curobo_reports_pinned_download_source(tmp_path):
    robotwin = tmp_path / "RoboTwin_lingbot"
    (robotwin / "assets/objects/objaverse").mkdir(parents=True)
    (robotwin / "assets/objects/objaverse/list.json").write_text("[]")
    (robotwin / "assets/embodiments/aloha-agilex").mkdir(parents=True)
    (robotwin / "assets/background_texture").mkdir(parents=True)

    result = run("--robotwin-dir", robotwin, "--check-only")

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "https://github.com/NVlabs/curobo.git" in combined
    assert "d64c4b005459db10c5dd867d8b30a87d5bda9bdb" in combined
