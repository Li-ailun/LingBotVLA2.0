from pathlib import Path


def test_inference_module_is_isolated_from_official_simulation() -> None:
    root = Path(__file__).resolve().parents[3]
    assert (root / "lingbot-vla-v2/experiment/robotwin").is_dir()
    assert (root / "baseline_base_model_adapter").is_dir()
    assert (root / "myscripts/inference_tools").is_dir()
