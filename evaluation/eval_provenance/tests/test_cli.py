import json
import os
from pathlib import Path
import subprocess
import sys


def test_init_run_writes_model_precision_mode_and_rtc_config(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    (model_dir / "config.json").write_text('{"model_type": "lingbot"}')
    env = os.environ.copy()
    env.update({
        "LINGBOT_EVAL_MODE": "rtc",
        "LINGBOT_MODEL_ID": "full5k",
        "LINGBOT_MODEL_PATH": str(model_dir),
        "LINGBOT_MODEL_ASSET_ROOT": "/models/full5k",
        "LINGBOT_PRECISION": "bf16",
        "LINGBOT_RTC_REQUEST_INDEX": "20",
        "LINGBOT_RTC_FIXED_DELAY_STEPS": "10",
        "LINGBOT_RTC_EXECUTION_HORIZON": "20",
        "LINGBOT_RTC_CROSSFADE_STEPS": "10",
        "LINGBOT_RTC_EMA_ALPHA": "0.21",
    })
    result = subprocess.run([
        sys.executable, "-m", "evaluation.eval_provenance.cli", "init-run",
        "--run-dir", str(tmp_path), "--task-config", "demo_clean",
        "--episodes", "50", "--action-chunk", "50",
        "--use-compile", "True", "--num-gpus", "1", "--num-servers", "1",
    ], text=True, capture_output=True, env=env)

    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["mode"] == "rtc"
    assert manifest["model"]["asset_root"] == "/models/full5k"
    assert manifest["model"]["id"] == "full5k"
    assert manifest["model"]["path"] == str(model_dir)
    assert manifest["model"]["files"]["config.json"]["sha256"] == \
        "20270c659223ac1fbbdf7f34793e27c9699e33b4a46901b2c5381acac89a4147"
    assert manifest["model"]["files"]["model.safetensors.index.json"]["size"] == 18
    assert manifest["precision"] == "bf16"
    assert manifest["rtc"] == {
        "crossfade_steps": 10,
        "ema_alpha": 0.21,
        "execution_horizon": 20,
        "fixed_delay_steps": 10,
        "request_index": 20,
        "switch_index": 30,
    }
    assert manifest["evaluation"]["episodes_per_task"] == 50
    assert manifest["inference"]["action_chunk"] == 50
