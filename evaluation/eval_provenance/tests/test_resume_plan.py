import json
from pathlib import Path

import pytest

from evaluation.eval_provenance.resume_plan import ResumePlanError, build_resume_plan


def write_manifest(run_dir: Path, **overrides):
    manifest = {
        "mode": "rtc",
        "model": {"id": "full30k"},
        "precision": "bf16",
        "evaluation": {
            "task_config": "demo_clean",
            "instruction_type": "unseen",
            "episodes_per_task": 50,
        },
        "rtc": {
            "request_index": 20,
            "fixed_delay_steps": 10,
            "execution_horizon": 20,
            "crossfade_steps": 10,
            "ema_alpha": 0.21,
        },
    }
    manifest.update(overrides)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def write_episodes(run_dir: Path, task: str, rows):
    path = run_dir / "eval_results" / task / "episodes.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def expected_config(**overrides):
    config = {
        "mode": "rtc",
        "model_id": "full30k",
        "precision": "bf16",
        "task_config": "demo_clean",
        "instruction_type": "unseen",
        "episodes": 50,
        "request_index": 20,
        "fixed_delay_steps": 10,
        "execution_horizon": 20,
        "crossfade_steps": 10,
        "ema_alpha": 0.21,
    }
    config.update(overrides)
    return config


def test_build_resume_plan_tracks_each_task_independently_and_skips_complete(tmp_path):
    write_manifest(tmp_path)
    write_episodes(
        tmp_path,
        "lift_pot",
        [
            {"task": "lift_pot", "episode_index": i, "environment_seed": 100000 + i, "success": i == 1}
            for i in range(3)
        ],
    )
    write_episodes(
        tmp_path,
        "click_bell",
        [
            {"task": "click_bell", "episode_index": i, "environment_seed": 100000 + i, "success": True}
            for i in range(50)
        ],
    )

    plan = build_resume_plan(tmp_path, ["lift_pot", "hanging_mug", "click_bell"], expected_config())

    assert [(item.task, item.episode_count, item.success_count, item.next_seed) for item in plan] == [
        ("lift_pot", 3, 1, 100003),
        ("hanging_mug", 0, 0, 100000),
    ]


def test_build_resume_plan_rejects_non_contiguous_episode_indices(tmp_path):
    write_manifest(tmp_path)
    write_episodes(tmp_path, "lift_pot", [
        {"task": "lift_pot", "episode_index": 0, "environment_seed": 100000, "success": False},
        {"task": "lift_pot", "episode_index": 2, "environment_seed": 100002, "success": False},
    ])

    with pytest.raises(ResumePlanError, match="contiguous"):
        build_resume_plan(tmp_path, ["lift_pot"], expected_config())


def test_build_resume_plan_rejects_non_increasing_environment_seeds(tmp_path):
    write_manifest(tmp_path)
    write_episodes(tmp_path, "lift_pot", [
        {"task": "lift_pot", "episode_index": 0, "environment_seed": 100003, "success": False},
        {"task": "lift_pot", "episode_index": 1, "environment_seed": 100002, "success": False},
    ])

    with pytest.raises(ResumePlanError, match="strictly increasing"):
        build_resume_plan(tmp_path, ["lift_pot"], expected_config())


def test_build_resume_plan_rejects_legacy_log_without_episode_records(tmp_path):
    write_manifest(tmp_path)
    log = tmp_path / "eval_logs/lift_pot.log"
    log.parent.mkdir(parents=True)
    log.write_text("Success rate: 3/10 => 30.0%, current seed: 100012\n", encoding="utf-8")

    with pytest.raises(ResumePlanError, match="manual single-task resume"):
        build_resume_plan(tmp_path, ["lift_pot"], expected_config())


def test_build_resume_plan_rejects_manifest_mismatch(tmp_path):
    write_manifest(tmp_path)

    with pytest.raises(ResumePlanError, match="instruction_type"):
        build_resume_plan(tmp_path, ["lift_pot"], expected_config(instruction_type="seen"))


def test_build_resume_plan_rejects_missing_manifest(tmp_path):
    with pytest.raises(ResumePlanError, match="run_manifest.json"):
        build_resume_plan(tmp_path, ["lift_pot"], expected_config())
