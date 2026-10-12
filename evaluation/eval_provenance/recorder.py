"""Append-only evaluation provenance files.

This module deliberately has no simulator or model dependencies.  Evaluation
clients call it only after an episode has completed, so recording cannot alter
the policy action path or RoboTwin's success decision.
"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import tempfile
from typing import Any


class EvaluationRecorder:
    """Write run and per-episode metadata with resume-safe de-duplication."""

    def __init__(self, run_dir: str | Path, task: str):
        self.run_dir = Path(run_dir).resolve()
        self.task = task
        self.task_dir = self.run_dir / "eval_results" / task
        self.task_dir.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.run_dir / ".provenance.lock"

    def record_session(self, metadata: dict[str, Any]) -> None:
        """Create the run manifest or append a launch/resume session."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self._locked():
            path = self.run_dir / "run_manifest.json"
            if path.exists():
                manifest = json.loads(path.read_text(encoding="utf-8"))
            else:
                manifest = {
                    key: value for key, value in metadata.items()
                    if key not in {"resume", "started_at"}
                }
                manifest["schema_version"] = 1
                manifest["sessions"] = []
            session = dict(metadata)
            session["resume"] = bool(metadata.get("resume", False))
            manifest["sessions"].append(session)
            self._atomic_json(path, manifest)

    def record_episode(self, episode: dict[str, Any]) -> bool:
        """Append an episode once; successful episodes also enter the index."""
        row = {"task": self.task, **episode}
        key = self._episode_key(row)
        task_path = self.task_dir / "episodes.jsonl"
        with self._locked():
            if key in self._existing_keys(task_path):
                return False
            self._append_jsonl(task_path, row)
            if bool(row.get("success")):
                self._append_jsonl(self.run_dir / "successful_episodes.jsonl", row)
        return True

    @staticmethod
    def _episode_key(row: dict[str, Any]) -> tuple[Any, Any, Any]:
        return row.get("task"), row.get("episode_index"), row.get("environment_seed")

    def _existing_keys(self, path: Path) -> set[tuple[Any, Any, Any]]:
        if not path.exists():
            return set()
        keys = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                keys.add(self._episode_key(json.loads(line)))
        return keys

    @staticmethod
    def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _atomic_json(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def _locked(self):
        return _FileLock(self._lock_path)


class _FileLock:
    def __init__(self, path: Path):
        self.path = path
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+", encoding="utf-8")
        fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        assert self.stream is not None
        fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()
