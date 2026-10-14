import subprocess
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
