from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / "evaluation/environment_reproduction.py"
EXPORTER = ROOT / "scripts/export_verified_environments.sh"
ASSET_MANIFEST = ROOT / "environment/assets-full30k.json"
CANONICAL_RUNNER = ROOT / "scripts/run_full30k_rtc_clean_unseen.sh"


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


def test_sanitize_conda_export_can_drop_editable_checkout_dependencies() -> None:
    module = _load()
    raw = """name: old
dependencies:
  - python=3.10
  - pip:
      - -e /home/hddData/User/lixiang/LingBotVLA2.0_new/lingbot-vla-v2
      - websockets==15.0.1
prefix: /home/hddData/User/lixiang/conda-envs/lingbotvla
"""

    result = module.sanitize_conda_export(
        raw, "lingbotvla", drop_editable=True
    )

    assert "-e " not in result
    assert "lingbot-vla-v2" not in result
    assert "websockets==15.0.1" in result
    assert "prefix:" not in result


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
    assert "--drop-editable" in text
    assert "pip list --format=freeze --exclude-editable" in text
    assert "pip freeze" not in text
    assert "check-portable" in text
    assert "--output /dev/null" not in text


def test_exporter_initializes_local_name_before_using_it_with_nounset() -> None:
    text = EXPORTER.read_text(encoding="utf-8")

    assert 'local name="$1" prefix="$2"\n' in text
    assert 'local raw="$stage/$name.raw.yml"\n' in text


def test_check_portable_cli_rejects_absolute_checkout_reference(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "freeze.txt"
    snapshot.write_text(
        "pkg @ file:///home/lixiang/PROJECT/pkg\n", encoding="utf-8"
    )

    result = subprocess.run(
        [sys.executable, str(MODULE), "check-portable", "--input", str(snapshot)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "non-portable" in result.stderr


def test_create_plan_requires_both_exported_environment_files(tmp_path: Path) -> None:
    module = _load()

    with pytest.raises(FileNotFoundError, match="lingbotvla.environment.yml"):
        module.build_create_plan(
            root=ROOT,
            conda_exe=Path("/opt/conda/bin/conda"),
            environment_dir=tmp_path,
            lingbot_prefix=tmp_path / "envs/lingbotvla",
            robotwin_prefix=tmp_path / "envs/RoboTwin",
            update=False,
        )


def test_create_plan_refuses_existing_environment_without_update(
    tmp_path: Path,
) -> None:
    module = _load()
    (tmp_path / "lingbotvla.environment.yml").write_text("name: lingbotvla\n")
    (tmp_path / "RoboTwin.environment.yml").write_text("name: RoboTwin\n")
    prefix = tmp_path / "envs/lingbotvla"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin/python").touch()

    with pytest.raises(FileExistsError, match="--update"):
        module.build_create_plan(
            root=ROOT,
            conda_exe=Path("/opt/conda/bin/conda"),
            environment_dir=tmp_path,
            lingbot_prefix=prefix,
            robotwin_prefix=tmp_path / "envs/RoboTwin",
            update=False,
        )


def test_create_plan_uses_current_checkout_and_deterministic_names(
    tmp_path: Path,
) -> None:
    module = _load()
    (tmp_path / "lingbotvla.environment.yml").write_text("name: lingbotvla\n")
    (tmp_path / "RoboTwin.environment.yml").write_text("name: RoboTwin\n")
    lingbot_prefix = tmp_path / "envs/lingbotvla"
    robotwin_prefix = tmp_path / "envs/RoboTwin"

    commands = module.build_create_plan(
        root=ROOT,
        conda_exe=Path("/opt/conda/bin/conda"),
        environment_dir=tmp_path,
        lingbot_prefix=lingbot_prefix,
        robotwin_prefix=robotwin_prefix,
        update=False,
    )

    assert commands[0][:4] == [
        "/opt/conda/bin/conda", "env", "create", "--prefix"
    ]
    assert commands[1][:4] == [
        "/opt/conda/bin/conda", "env", "create", "--prefix"
    ]
    assert [str(ROOT / "lingbot-vla-v2")] in [command[-1:] for command in commands]
    curobo = str(ROOT / "RoboTwin_lingbot/envs/curobo")
    assert any(curobo in command for command in commands)


def test_create_script_dry_run_does_not_create_environments(tmp_path: Path) -> None:
    env_dir = tmp_path / "definitions"
    env_dir.mkdir()
    (env_dir / "lingbotvla.environment.yml").write_text("name: lingbotvla\n")
    (env_dir / "RoboTwin.environment.yml").write_text("name: RoboTwin\n")
    envs = tmp_path / "envs"
    script = ROOT / "scripts/create_eval_environments.sh"

    result = subprocess.run(
        [
            "bash", str(script), "--dry-run", "--conda", "/bin/true",
            "--environment-dir", str(env_dir),
            "--lingbot-prefix", str(envs / "lingbotvla"),
            "--robotwin-prefix", str(envs / "RoboTwin"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "env create" in result.stdout
    assert not envs.exists()


def test_full30k_asset_manifest_lists_every_runtime_input() -> None:
    import json

    manifest = json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))
    ids = {asset["id"] for asset in manifest["assets"]}
    assert {
        "qwen3_vl_4b", "full30k_checkpoint", "full5k_runtime_config",
        "moge_depth", "lingbot_depth", "dino_video",
        "robotwin_objects", "robotwin_embodiments",
        "robotwin_background_texture", "curobo_source",
    } <= ids
    assert manifest["canonical_evaluation"]["precision"] == "fp32"
    assert manifest["canonical_evaluation"]["instruction_type"] == "unseen"
    assert manifest["canonical_evaluation"]["ema_alpha"] == 0.10


def test_validate_assets_checks_indexed_shards_and_incomplete_files(
    tmp_path: Path,
) -> None:
    import json

    module = _load()
    checkpoint = tmp_path / "models/lingbot_30K"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "one.safetensors", "b": "two.safetensors"}})
    )
    (checkpoint / "one.safetensors").write_bytes(b"one")
    (checkpoint / "two.safetensors").write_bytes(b"two")
    manifest = {
        "assets": [{
            "id": "full30k_checkpoint",
            "type": "hf_indexed_checkpoint",
            "path": "models/lingbot_30K",
            "required": ["config.json", "model.safetensors.index.json"],
        }]
    }

    assert module.validate_assets(tmp_path, manifest) == []
    (checkpoint / "two.safetensors").unlink()
    assert any("two.safetensors" in item for item in module.validate_assets(tmp_path, manifest))
    (checkpoint / "two.safetensors").write_bytes(b"two")
    (checkpoint / "model-00003.incomplete").touch()
    assert any("incomplete" in item.lower() for item in module.validate_assets(tmp_path, manifest))


@pytest.mark.parametrize("value", ["", "1,1", "1,-2", "1,a", "1,,3"])
def test_runtime_plan_rejects_invalid_or_duplicate_gpu_ids(value: str) -> None:
    module = _load()
    with pytest.raises(ValueError):
        module.build_runtime_plan(value, 9380, 55000)


def test_runtime_plan_preserves_noncontiguous_gpu_ids_and_port_range() -> None:
    module = _load()

    plan = module.build_runtime_plan("1,2,5,8", 9380, 55000)

    assert plan == {
        "gpu_ids": [1, 2, 5, 8],
        "ports": [9380, 9381, 9382, 9383],
        "min_free_mib": 55000,
    }


def test_canonical_runner_dry_run_has_full_fp32_rtc_contract() -> None:
    result = subprocess.run(
        [
            "bash", str(CANONICAL_RUNNER),
            "--gpu-ids", "1,2,3,5",
            "--base-port", "9380",
            "--episodes", "50",
            "--min-free-mib", "55000",
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout
    for expected in (
        "--setting demo_clean", "--model full30k", "--precision fp32",
        "--episodes 50", "--gpu-ids 1,2,3,5", "--base-port 9380",
        "--instruction-type unseen", "--request-index 20",
        "--fixed-delay-steps 10", "--execution-horizon 20",
        "--crossfade-steps 10", "--ema-alpha 0.10",
    ):
        assert expected in output
    assert "/home/hddData" not in output
