# Base-model zero-shot RoboTwin baseline specification

## Objective

Evaluate the untouched `robbyant/lingbot-vla-v2-6b` foundation checkpoint on
the 50 competition RoboTwin tasks before any 50-task post-training. Each task
must run 50 episodes in both `demo_clean` and `demo_randomized` settings.

## Compliance boundary

- Use `/home/lixiang/PROJECT/LingBotVLA2.0/models/lingbot-vla-v2-6b` only.
- Never download or reference `lingbot-vla-v2-6b-robotwin`.
- Never edit, copy, or resave foundation-model tensors.
- Use only the competition clean dataset to derive robot normalization stats.
- Keep all compatibility artifacts separate from future trained baselines.

## Layout

Create `baseline_base_model_adapter/` at the workspace root. It contains:

- `lingbotvla_cli.yaml`: inference metadata matching the official RoboTwin
  training configuration while pointing at the untouched foundation model,
  local Qwen3-VL dependency, and clean-only normalization statistics.
- `checkpoints/global_step_0/hf_ckpt/`: relative symbolic links to the six
  foundation-model safetensor shards and their index/config metadata.
- `assets/norm_stats/robotwin_clean_50x50.json`: statistics computed from the
  official 2,500 clean demonstrations.
- `README.md`: provenance, limitations, validation, smoke-test, and full
  50-task commands.
- `manifest.json`: resolved sources and SHA-256 hashes for auditable provenance.

## Evaluation interface

Add `--num_episodes` to the LingBot RoboTwin launcher and evaluation client.
Its default remains 100 for official reproduction. The local zero-shot
baseline commands pass `--num_episodes 50` explicitly. Summary completeness
checks must use the selected episode count rather than a hard-coded 100.

The target machine has one RTX 3090 24 GB, so the supplied command uses one
server, BF16, and no video. BF16 is a local baseline and is not claimed to
reproduce the published FP32 checkpoint results.

## Acceptance criteria

1. Adapter validation rejects missing/non-symlink tensor shards, forbidden
   RobotWin post-trained paths, wrong shard counts, missing norm stats, and
   evaluation commands that do not request 50 episodes.
2. Model state dictionaries in the adapter resolve to the exact original
   files and hashes.
3. The inference configuration parses and all referenced local paths exist.
4. The launcher dry validation accepts 50 tasks and 50 episodes.
5. A model-load smoke test reaches successful strict weight loading on the
   available GPU without starting the 2,500-episode benchmark.

