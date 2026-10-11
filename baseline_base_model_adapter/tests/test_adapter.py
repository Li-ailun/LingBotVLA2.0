from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


MODULE = Path(__file__).resolve().parents[1]
BUILDER = MODULE / "tools/build_adapter.py"
VALIDATOR = MODULE / "tools/validate_adapter.py"


class AdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.source = root / "lingbot-vla-v2-6b"
        self.qwen = root / "Qwen3-VL-4B-Instruct"
        self.adapter = root / "adapter"
        self.norm = root / "robotwin_clean.json"
        self.source.mkdir()
        self.qwen.mkdir()
        self.norm.write_text('{"count": 2500}\n')
        (self.source / "config.json").write_text('{"vlm_family": "qwen3vl"}\n')
        (self.source / "model.safetensors.index.json").write_text('{}\n')
        for index in range(1, 7):
            (self.source / f"model-{index:05d}-of-00006.safetensors").write_bytes(
                f"shard-{index}".encode()
            )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def build(self, source: Path | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(BUILDER),
                "--source-model", str(source or self.source),
                "--qwen-path", str(self.qwen),
                "--norm-stats", str(self.norm),
                "--output", str(self.adapter),
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_builds_relative_links_and_valid_manifest(self) -> None:
        result = self.build()
        self.assertEqual(result.returncode, 0, result.stderr)

        checkpoint = self.adapter / "checkpoints/global_step_0/hf_ckpt"
        shards = sorted(checkpoint.glob("model-*-of-00006.safetensors"))
        self.assertEqual(len(shards), 6)
        self.assertTrue(all(path.is_symlink() for path in shards))
        self.assertTrue(all(not path.readlink().is_absolute() for path in shards))
        manifest = json.loads((self.adapter / "manifest.json").read_text())
        self.assertEqual(manifest["source_model"], str(self.source.resolve()))
        self.assertEqual(len(manifest["weights"]), 6)
        config = yaml.safe_load((self.adapter / "lingbotvla_cli.yaml").read_text())
        self.assertTrue(all(isinstance(item, str) for item in config["data"]["joints"]))
        self.assertTrue(all(isinstance(item, str) for item in config["data"]["norm_type"]))

        validation = subprocess.run(
            [sys.executable, str(VALIDATOR), "--adapter", str(self.adapter)],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(validation.returncode, 0, validation.stderr)
        self.assertIn("ADAPTER VALID", validation.stdout)

    def test_rejects_forbidden_robotwin_posttrained_source(self) -> None:
        forbidden = Path(self.tmp.name) / "lingbot-vla-v2-6b-robotwin"
        forbidden.symlink_to(self.source, target_is_directory=True)
        result = self.build(forbidden)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("forbidden", result.stderr.lower())

    def test_rejects_incomplete_checkpoint(self) -> None:
        (self.source / "model-00006-of-00006.safetensors").unlink()
        result = self.build()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exactly 6", result.stderr)


if __name__ == "__main__":
    unittest.main()
