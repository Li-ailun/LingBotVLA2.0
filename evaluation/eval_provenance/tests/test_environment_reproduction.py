from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / "evaluation/environment_reproduction.py"
EXPORTER = ROOT / "scripts/export_verified_environments.sh"


def _load():
    spec = importlib.util.spec_from_file_location("environment_reproduction", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sanitize_conda_export_replaces_name_and_removes_prefix() -> None:
    module = _load()
    raw = """name: old
channels:
  - defaults
dependencies:
  - python=3.10.21
  - pip:
      - websockets==15.0.1
prefix: /home/hddData/User/lixiang/conda-envs/old
"""

    result = module.sanitize_conda_export(raw, "RoboTwin")

    assert result.startswith("name: RoboTwin\n")
    assert "prefix:" not in result
    assert "websockets==15.0.1" in result
    assert result.endswith("\n")


@pytest.mark.parametrize(
    "bad_line",
    [
        "  - package @ file:///tmp/package.whl",
        "      - -e /home/lixiang/src/package",
        "      - package @ /home/hddData/User/lixiang/package",
    ],
)
def test_sanitize_conda_export_rejects_nonportable_dependencies(
    bad_line: str,
) -> None:
    module = _load()
    raw = f"name: old\ndependencies:\n{bad_line}\n"

    with pytest.raises(ValueError, match="non-portable"):
        module.sanitize_conda_export(raw, "lingbotvla")


def test_find_nonportable_references_reports_each_unique_reference() -> None:
    module = _load()
    text = """a=file:///tmp/a
b=/home/lixiang/code
c=-e /home/lixiang/code
"""

    findings = module.find_nonportable_references(text)

    assert any("file://" in item for item in findings)
    assert any("/home/lixiang/" in item for item in findings)
    assert len(findings) == len(set(findings))


def test_sanitize_cli_keeps_existing_output_when_input_is_invalid(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bad.yml"
    output = tmp_path / "portable.yml"
    source.write_text(
        "name: old\ndependencies:\n  - pkg @ file:///tmp/pkg.whl\n",
        encoding="utf-8",
    )
    output.write_text("keep-me\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(MODULE),
            "sanitize-conda",
            "--input",
            str(source),
            "--output",
            str(output),
            "--name",
            "lingbotvla",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "non-portable" in result.stderr
    assert output.read_text(encoding="utf-8") == "keep-me\n"


def test_exporter_uses_verified_server_prefix_defaults() -> None:
    text = EXPORTER.read_text(encoding="utf-8")
    assert "/home/hddData/User/lixiang/conda-envs/lingbotvla" in text
    assert "/home/hddData/User/lixiang/conda-envs/RoboTwin" in text
    assert "mktemp" in text
