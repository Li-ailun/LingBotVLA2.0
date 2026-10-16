import subprocess
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REGISTRY = ROOT / "evaluation/model_registry.sh"


def resolve(name: str):
    command = (
        f'set -e; source "{REGISTRY}"; '
        f'resolve_lingbot_model "{name}" "{ROOT}"; '
        "printf '%s\\n%s\\n%s\\n' \"$LINGBOT_MODEL_ID\" \"$LINGBOT_MODEL_PATH\" \"$LINGBOT_MODEL_ASSET_ROOT\""
    )
    return subprocess.run(["bash", "-c", command], text=True, capture_output=True)


def make_required_file(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def make_full5k_auxiliary_assets(root: Path):
    asset_root = root / "models/lingbot_full_5k_6_60"
    for relative in (
        "assets/norm_stats/robotwin_clean.json",
        "configs/robot_configs_clean/robotwin.yaml",
        "lingbotvla_cli.yaml",
    ):
        make_required_file(asset_root / relative)


def make_model_export(model: Path, shard_names=("model-00001-of-00001.safetensors",)):
    make_required_file(model / "config.json")
    weight_map = {f"parameter_{index}": name for index, name in enumerate(shard_names)}
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )
    for name in shard_names:
        make_required_file(model / name)


def resolve_at(name: str, root: Path):
    command = (
        f'source "{REGISTRY}"; resolve_lingbot_model "{name}" "{root}"; status=$?; '
        "if [[ $status -eq 0 ]]; then "
        "printf '%s\\n%s\\n%s\\n' \"$LINGBOT_MODEL_ID\" \"$LINGBOT_MODEL_PATH\" \"$LINGBOT_MODEL_ASSET_ROOT\"; "
        "fi; exit $status"
    )
    return subprocess.run(["bash", "-c", command], text=True, capture_output=True)


def test_registry_resolves_flat_step2000_model():
    result = resolve("step2000")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "step2000",
        str(ROOT / "models/lingbot_action_step2000"),
        str(ROOT / "models/lingbot_action_step2000"),
    ]


def test_registry_resolves_nested_full5k_model_without_weight_copy():
    result = resolve("full5k")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "full5k",
        str(ROOT / "models/lingbot_full_5k_6_60/checkpoints/global_step_5000/hf_ckpt"),
        str(ROOT / "models/lingbot_full_5k_6_60"),
    ]


def test_full5k_uses_its_exported_training_config_and_shared_dependency_root():
    command = (
        f'set -e; source "{REGISTRY}"; resolve_lingbot_model full5k "{ROOT}"; '
        "printf '%s\\n%s\\n' \"$LINGBOT_TRAINING_CONFIG\" \"$LINGBOT_MODELS_ROOT\""
    )
    result = subprocess.run(["bash", "-c", command], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        str(ROOT / "models/lingbot_full_5k_6_60/lingbotvla_cli.yaml"),
        str(ROOT / "models"),
    ]


def test_registry_resolves_flat_full15k_weights_with_full5k_auxiliary_assets():
    result = resolve("full15k")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "full15k",
        str(ROOT / "models/lingbot_15K"),
        str(ROOT / "models/lingbot_full_5k_6_60"),
    ]


def test_registry_accepts_all_full15k_aliases():
    for alias in ("full15k", "15k", "lingbot_15K"):
        result = resolve(alias)
        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines()[0] == "full15k"


def test_full15k_reuses_exact_full5k_configuration_paths():
    command = (
        f'set -e; source "{REGISTRY}"; resolve_lingbot_model full15k "{ROOT}"; '
        "printf '%s\\n%s\\n%s\\n%s\\n' "
        '"$LINGBOT_NORM_STATS" "$LINGBOT_ROBOT_CONFIG" '
        '"$LINGBOT_TRAINING_CONFIG" "$LINGBOT_MODELS_ROOT"'
    )
    result = subprocess.run(["bash", "-c", command], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        str(ROOT / "models/lingbot_full_5k_6_60/assets/norm_stats/robotwin_clean.json"),
        str(ROOT / "models/lingbot_full_5k_6_60/configs/robot_configs_clean/robotwin.yaml"),
        str(ROOT / "models/lingbot_full_5k_6_60/lingbotvla_cli.yaml"),
        str(ROOT / "models"),
    ]


def test_registry_resolves_flat_full30k_weights_with_full5k_auxiliary_assets(tmp_path):
    make_full5k_auxiliary_assets(tmp_path)
    model = tmp_path / "models/lingbot_30K"
    make_model_export(model)

    result = resolve_at("full30k", tmp_path)

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "full30k",
        str(model),
        str(tmp_path / "models/lingbot_full_5k_6_60"),
    ]


def test_registry_resolves_nested_full30k_checkpoint_export(tmp_path):
    make_full5k_auxiliary_assets(tmp_path)
    model = tmp_path / "models/lingbot_30K/checkpoints/global_step_30000/hf_ckpt"
    make_model_export(model)

    result = resolve_at("30k", tmp_path)

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0:2] == ["full30k", str(model)]


def test_registry_accepts_all_full30k_aliases(tmp_path):
    make_full5k_auxiliary_assets(tmp_path)
    model = tmp_path / "models/lingbot_30K"
    make_model_export(model)

    for alias in ("full30k", "30k", "lingbot_30K"):
        result = resolve_at(alias, tmp_path)
        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines()[0] == "full30k"


def test_registry_rejects_incomplete_full30k_download_and_lists_checked_layouts(tmp_path):
    make_full5k_auxiliary_assets(tmp_path)
    (tmp_path / "models/lingbot_30K").mkdir(parents=True)

    result = resolve_at("full30k", tmp_path)

    assert result.returncode == 2
    assert "30K model download is incomplete" in result.stderr
    assert "models/lingbot_30K/config.json" in result.stderr
    assert "checkpoints/global_step_30000/hf_ckpt/config.json" in result.stderr


def test_registry_rejects_full30k_layout_with_incomplete_weight_shards(tmp_path):
    make_full5k_auxiliary_assets(tmp_path)
    model = tmp_path / "models/lingbot_30K"
    make_model_export(model)
    make_required_file(model / "model-00001-of-00006.safetensors.incomplete")

    result = resolve_at("full30k", tmp_path)

    assert result.returncode == 2
    assert "30K model download is incomplete" in result.stderr
    assert "model-00001-of-00006.safetensors.incomplete" in result.stderr


def test_registry_rejects_full30k_when_an_indexed_shard_is_missing(tmp_path):
    make_full5k_auxiliary_assets(tmp_path)
    model = tmp_path / "models/lingbot_30K"
    make_model_export(model, ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"))
    (model / "model-00002-of-00002.safetensors").unlink()

    result = resolve_at("full30k", tmp_path)

    assert result.returncode == 2
    assert "indexed weight shard is missing" in result.stderr
    assert "model-00002-of-00002.safetensors" in result.stderr


def test_registry_rejects_unknown_model():
    result = resolve("unknown")
    assert result.returncode != 0
    assert "Unknown model" in result.stderr


def test_precision_mapping_is_exclusive():
    command = (
        f'source "{REGISTRY}"; resolve_lingbot_precision fp32; '
        "printf '%s %s %s\\n' \"$LINGBOT_PRECISION\" \"$LINGBOT_USE_BF16\" \"$LINGBOT_USE_FP32\""
    )
    result = subprocess.run(["bash", "-c", command], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "fp32 False True"
