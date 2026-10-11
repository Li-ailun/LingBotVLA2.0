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
