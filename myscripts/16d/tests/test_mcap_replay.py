from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "record_tools" / "replay_r1pro_mcap_keyboard.py"


def _load():
    spec = importlib.util.spec_from_file_location("replay_r1pro_mcap_keyboard", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _episode(task_dir: Path, index: int, stamp: str = "stamp") -> Path:
    episode = task_dir / f"episode_{index:06d}_{stamp}"
    bag = episode / "bag"
    bag.mkdir(parents=True)
    (bag / "bag_0.mcap").write_bytes(b"test")
    (bag / "metadata.yaml").write_text("test\n")
    return episode


def test_selection_follows_operator_then_instruction_layout(tmp_path: Path) -> None:
    module = _load()
    selected = module.selected_task_directory(tmp_path, " LIU ", "2")
    assert selected == (
        tmp_path
        / "stack_plates_and_cups"
        / "liu"
        / "stack_plates_cups_02"
    )


def test_discover_episodes_sorts_by_numeric_mcap_index(tmp_path: Path) -> None:
    module = _load()
    task_dir = tmp_path / "task"
    _episode(task_dir, 109)
    _episode(task_dir, 7)
    _episode(task_dir, 42)

    episodes = module.discover_episodes(task_dir)

    assert [episode.episode_index for episode in episodes] == [7, 42, 109]
    assert episodes[-1].episode_dir.name.startswith("episode_000109_")


def test_discover_episodes_rejects_missing_mcap(tmp_path: Path) -> None:
    module = _load()
    task_dir = tmp_path / "task"
    (task_dir / "episode_000003_stamp" / "bag").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="missing MCAP"):
        module.discover_episodes(task_dir)


def test_cli_exposes_recording_order_selection_options() -> None:
    result = subprocess.run(
        ["/usr/bin/python3", str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for option in ("--root", "--operator", "--instruction-id", "--episode-index", "--rate"):
        assert option in result.stdout


@pytest.mark.parametrize(
    ("key", "expected_rate"),
    [
        (ord("1"), 1.0),
        (ord("2"), 2.0),
        (ord("5"), 5.0),
        (ord("0"), 10.0),
        (ord("w"), 20.0),
        (ord("f"), 50.0),
    ],
)
def test_playback_speed_shortcuts_match_displayed_rates(
    key: int, expected_rate: float
) -> None:
    module = _load()

    assert module.playback_rate_for_key(key) == expected_rate


def test_unrelated_key_does_not_change_playback_speed() -> None:
    module = _load()

    assert module.playback_rate_for_key(ord("n")) is None


def test_normal_playback_advances_one_reference_frame() -> None:
    module = _load()

    assert module.next_reference_index((0.0, 0.1, 0.2, 0.3), 1, 10.0) == 2


def test_fast_review_skips_to_timestamp_for_requested_rate() -> None:
    module = _load()
    timestamps = tuple(index / 15.0 for index in range(1000))

    assert module.next_reference_index(timestamps, 0, 20.0) == 20
    assert module.next_reference_index(timestamps, 0, 50.0) == 50


def test_fast_review_always_moves_forward_at_end_of_stream() -> None:
    module = _load()

    assert module.next_reference_index((0.0, 0.1, 0.2), 1, 50.0) == 2


def test_last_reference_frame_advances_past_end() -> None:
    module = _load()

    assert module.next_reference_index((0.0, 0.1, 0.2), 2, 50.0) == 3


def test_quarantine_episode_moves_directory_and_records_manifest(tmp_path: Path) -> None:
    module = _load()
    task_dir = (
        tmp_path / "stack_plates_and_cups" / "li" / "stack_plates_cups_01"
    )
    source = _episode(task_dir, 109, "20261015103209687")
    episode = module.discover_episodes(task_dir)[0]

    destination = module.quarantine_episode(
        tmp_path,
        episode,
        "li",
        "stack_plates_cups_01",
        deleted_at="20261015_120000_123456",
    )

    assert not source.exists()
    assert destination == (
        tmp_path
        / "_deleted_episodes"
        / "20261015_120000_123456"
        / "li"
        / "stack_plates_cups_01"
        / source.name
    )
    assert (destination / "bag" / "bag_0.mcap").read_bytes() == b"test"
    manifest = tmp_path / "_deleted_episodes" / "deletion_manifest.jsonl"
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert records == [
        {
            "action": "quarantine",
            "deleted_at": "20261015_120000_123456",
            "episode_index": 109,
            "instruction_id": "stack_plates_cups_01",
            "operator_id": "li",
            "original_path": str(source.resolve()),
            "quarantine_path": str(destination.resolve()),
        }
    ]


def test_quarantine_episode_refuses_to_overwrite_existing_destination(
    tmp_path: Path,
) -> None:
    module = _load()
    task_dir = (
        tmp_path / "stack_plates_and_cups" / "liu" / "stack_plates_cups_02"
    )
    source = _episode(task_dir, 7, "stamp")
    episode = module.discover_episodes(task_dir)[0]
    destination = (
        tmp_path
        / "_deleted_episodes"
        / "same_time"
        / "liu"
        / "stack_plates_cups_02"
        / source.name
    )
    destination.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="quarantine destination already exists"):
        module.quarantine_episode(
            tmp_path,
            episode,
            "liu",
            "stack_plates_cups_02",
            deleted_at="same_time",
        )

    assert source.exists()
    assert not (tmp_path / "_deleted_episodes" / "deletion_manifest.jsonl").exists()
