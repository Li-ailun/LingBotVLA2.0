from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "record_tools" / "record_r1pro_mcap_episode.py"


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/usr/bin/python3", "-c", code], capture_output=True, text=True, check=False
    )


def test_recorder_exposes_only_torso_free_16d_core_topics() -> None:
    result = _run_python(
        "import importlib.util,json;"
        f"s=importlib.util.spec_from_file_location('recorder',{str(SCRIPT)!r});"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "print(json.dumps(m.r1pro_16d_core_topics()))"
    )
    assert result.returncode == 0, result.stderr
    topics = json.loads(result.stdout)
    assert len(topics) == 11
    assert not any("torso" in topic.lower() for topic in topics)


def test_recorder_cli_has_no_torso_option() -> None:
    result = subprocess.run(
        ["/usr/bin/python3", str(SCRIPT), "--help"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--record-all-topics" not in result.stdout
    assert "--record-minimal-with-ee" not in result.stdout
    assert "torso" not in result.stdout.lower()
