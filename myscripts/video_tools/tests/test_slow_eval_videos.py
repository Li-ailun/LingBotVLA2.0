from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "myscripts/video_tools/slow_eval_videos.sh"


def test_dry_run_accepts_multiple_videos_and_preserves_sources(tmp_path):
    first = tmp_path / "rtc.mp4"
    second = tmp_path / "sync.mp4"
    first.touch()
    second.touch()
    result = subprocess.run(
        ["bash", str(SCRIPT), "--speed", "0.25", "--dry-run", str(first), str(second)],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "rtc_slow_0.25x.mp4" in result.stdout
    assert "sync_slow_0.25x.mp4" in result.stdout
    assert "setpts=PTS/0.25" in result.stdout


def test_invalid_speed_is_rejected(tmp_path):
    source = tmp_path / "source.mp4"
    source.touch()
    result = subprocess.run(
        ["bash", str(SCRIPT), "--speed", "0", "--dry-run", str(source)],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert result.returncode != 0
    assert "speed must be" in result.stderr
