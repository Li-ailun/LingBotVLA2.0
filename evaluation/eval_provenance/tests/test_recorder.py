import json
from pathlib import Path

from evaluation.eval_provenance.recorder import EvaluationRecorder


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_records_episode_once_and_builds_success_index(tmp_path: Path) -> None:
    recorder = EvaluationRecorder(tmp_path, task="lift_pot")
    episode = {
        "episode_index": 3,
        "environment_seed": 100141,
        "success": True,
        "instruction": "lift the pot",
        "policy_steps": 120,
        "inference_calls": 4,
    }

    assert recorder.record_episode(episode) is True
    assert recorder.record_episode(episode) is False

    task_rows = _read_jsonl(tmp_path / "eval_results/lift_pot/episodes.jsonl")
    success_rows = _read_jsonl(tmp_path / "successful_episodes.jsonl")
    assert task_rows == [{"task": "lift_pot", **episode}]
    assert success_rows == [{"task": "lift_pot", **episode}]


def test_failure_is_not_added_to_success_index(tmp_path: Path) -> None:
    recorder = EvaluationRecorder(tmp_path, task="lift_pot")

    recorder.record_episode({
        "episode_index": 0,
        "environment_seed": 100001,
        "success": False,
    })

    assert len(_read_jsonl(tmp_path / "eval_results/lift_pot/episodes.jsonl")) == 1
    assert not (tmp_path / "successful_episodes.jsonl").exists()


def test_manifest_preserves_first_run_and_appends_resume_session(tmp_path: Path) -> None:
    recorder = EvaluationRecorder(tmp_path, task="lift_pot")
    base = {
        "mode": "rtc",
        "model": {"id": "full5k", "path": "/models/full5k"},
        "precision": "bf16",
        "rtc": {"request_index": 20, "fixed_delay_steps": 10},
    }

    recorder.record_session({**base, "resume": False, "started_at": "first"})
    recorder.record_session({**base, "resume": True, "started_at": "second"})

    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["mode"] == "rtc"
    assert manifest["model"]["id"] == "full5k"
    assert manifest["precision"] == "bf16"
    assert manifest["rtc"]["request_index"] == 20
    assert [session["started_at"] for session in manifest["sessions"]] == ["first", "second"]
    assert manifest["sessions"][1]["resume"] is True
