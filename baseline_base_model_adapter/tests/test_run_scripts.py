from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


MODULE = Path(__file__).resolve().parents[1]


class RunScriptsTest(unittest.TestCase):
    def command(self, script: str, *args: str) -> str:
        result = subprocess.run(
            ["bash", str(MODULE / script), *args, "--dry-run"],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def assert_common_baseline_contract(self, command: str) -> None:
        self.assertIn("--num_tasks 50", command)
        self.assertIn("--num_episodes 50", command)
        self.assertIn("--num_gpus 1", command)
        self.assertIn("--num_per_gpu 1", command)
        self.assertIn("--use_bf16 True", command)
        self.assertIn("--use_fp32 False", command)
        self.assertIn("--no_video", command)
        self.assertIn("baseline_base_model_adapter/checkpoints/global_step_0/hf_ckpt", command)
        self.assertNotIn("lingbot-vla-v2-6b-robotwin", command)

    def test_clean_script_prints_zero_shot_50_by_50_command(self) -> None:
        command = self.command("run_clean_50x50.sh")
        self.assert_common_baseline_contract(command)
        self.assertIn("--task_config demo_clean", command)
        self.assertIn("results/zero_shot_clean", command)

    def test_clean_visualize_is_explicit_single_task_live_view(self) -> None:
        command = self.command(
            "run_clean_50x50.sh",
            "--visualize", "--task", "lift_pot", "--num-episodes", "1",
        )
        self.assertIn("--task_name lift_pot", command)
        self.assertIn("--num_tasks 1", command)
        self.assertIn("--num_episodes 1", command)
        self.assertIn("--render_freq 1", command)
        self.assertIn("--no_video", command)
        self.assertIn("results/visualization_clean", command)
        self.assertNotIn("results/zero_shot_clean", command)

    def test_launcher_signal_handlers_exit_after_cleanup(self) -> None:
        launcher = (
            MODULE.parent
            / "lingbot-vla-v2/experiment/robotwin/start_robotwin_infer_and_eval.sh"
        ).read_text()
        self.assertIn("trap 'exit 130' INT", launcher)
        self.assertIn("trap 'exit 143' TERM", launcher)
        self.assertIn("trap cleanup EXIT", launcher)

    def test_clean_resume_preserves_recorded_checkpoint(self) -> None:
        command = self.command(
            "run_clean_50x50.sh",
            "--resume-run",
            str(MODULE / "results/zero_shot_clean/baseline_base_model_adapter_0k_demo_clean_20260930_143902"),
        )
        self.assertIn("--resume_run_dir", command)
        self.assertIn("--task_name lift_pot", command)
        self.assertIn("--initial_episode_count 19", command)
        self.assertIn("--initial_success_count 0", command)
        self.assertIn("--start_seed 100046", command)
        self.assertIn("--no_video", command)

    def test_randomized_script_prints_zero_shot_50_by_50_command(self) -> None:
        command = self.command("run_randomized_50x50.sh")
        self.assert_common_baseline_contract(command)
        self.assertIn("--task_config demo_randomized", command)
        self.assertIn("results/zero_shot_randomized", command)

    def test_preflight_executes_launcher_validation_for_50_episodes(self) -> None:
        result = subprocess.run(
            ["bash", str(MODULE / "preflight.sh")],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("num_episodes=50", result.stdout)
        self.assertIn("RoboTwin websockets 15.0.1", result.stdout)
        self.assertIn("PREFLIGHT OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
