from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

import yaml


MODULE = Path(__file__).resolve().parents[1]
DATASET = Path("/home/lixiang/PROJECT/LingBotVLA2.0/RoboTwin2_data/lerobot_dataset/RoboTwin_lerobot_v30")


class NormAndSmokeTest(unittest.TestCase):
    def test_norm_config_selects_only_official_2500_episode_clean_dataset(self) -> None:
        config = yaml.safe_load((MODULE / "configs/compute_norm_stats.yaml").read_text())
        self.assertEqual(config["data"]["data_name"], "robotwin")
        self.assertEqual(Path(config["data"]["train_path"]).resolve(), DATASET)
        info = json.loads((DATASET / "meta/info.json").read_text())
        self.assertEqual(info["total_episodes"], 2500)
        self.assertEqual(info["splits"], {"train": "0:2500"})

    def test_smoke_check_rejects_missing_norm_stats(self) -> None:
        result = subprocess.run(
            [sys.executable, str(MODULE / "tools/smoke_load.py"),
             "--adapter", str(MODULE), "--check-only",
             "--norm-stats", str(MODULE / "missing.json")],
            text=True, capture_output=True, check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("norm stats", result.stderr.lower())


if __name__ == "__main__":
    unittest.main()
