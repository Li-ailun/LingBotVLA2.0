# Base-model Zero-shot RoboTwin Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare an isolated, auditable inference adapter for the untouched LingBot-VLA 2.0 6B foundation model and a 50-task × 50-episode RoboTwin evaluation command.

**Architecture:** Preserve the downloaded model byte-for-byte and expose it through the checkpoint layout expected by the official inference loader using relative symbolic links. Generate clean-only robot normalization statistics, parameterize the official launcher episode count without changing its 100-episode default, and validate all provenance and paths before handing off the benchmark command.

**Tech Stack:** Bash, Python 3.12, PyYAML, safetensors, pytest, LingBot-VLA 2.0, RoboTwin 2.0.

**Spec:** `docs/superpowers/specs/2026-09-30-base-model-zero-shot-baseline.md`

## Global Constraints

- The sole VLA weight source is `models/lingbot-vla-v2-6b`.
- `lingbot-vla-v2-6b-robotwin` is forbidden as a source, teacher, or reference.
- Original model tensors and official downloaded assets remain unmodified.
- Robot normalization statistics use only the official 2,500 clean demonstrations.
- Adapter artifacts live only under `baseline_base_model_adapter/`.
- Official episode default remains 100; the local baseline explicitly selects 50.

## Review Focus

- A similarly named forbidden post-trained path must be rejected by validation.
- Broken or absolute tensor symlinks must fail provenance validation.
- A partial six-shard checkpoint must not be accepted.
- The launcher must propagate 50 episodes into every queued task and its summary.
- A 24 GB GPU command must use BF16 and one resident server to avoid guaranteed FP32 OOM.

---

### Task 1: Parameterize episodes without changing official defaults

**Files:**
- Modify: `lingbot-vla-v2/experiment/robotwin/start_robotwin_infer_and_eval.sh`
- Modify: `lingbot-vla-v2/experiment/robotwin/eval_policy_client_lingbotvla.py`
- Create: `lingbot-vla-v2/tests/test_robotwin_episode_count.py`

**Interfaces:**
- Consumes: launcher CLI arguments.
- Produces: `--num_episodes INT`, default `100`, forwarded to every client and used by summary completeness checks.

- [ ] Write tests asserting the default is 100, explicit 50 propagates to the client command, invalid non-positive values fail, and summary messages use the selected value.
- [ ] Run the focused test and verify it fails because `--num_episodes` does not exist.
- [ ] Add the minimum launcher/client argument plumbing and dynamic summary checks.
- [ ] Run the focused test and verify it passes.

### Task 2: Build and audit the isolated base-model adapter

**Files:**
- Create: `baseline_base_model_adapter/tools/build_adapter.py`
- Create: `baseline_base_model_adapter/tools/validate_adapter.py`
- Create: `baseline_base_model_adapter/tests/test_adapter.py`
- Create: generated adapter metadata, symlinks, and documentation under `baseline_base_model_adapter/`

**Interfaces:**
- Consumes: foundation checkpoint path, Qwen3-VL path, clean dataset path, norm-stat path.
- Produces: `checkpoints/global_step_0/hf_ckpt`, `lingbotvla_cli.yaml`, and `manifest.json` without copying model tensors.

- [ ] Write tests for exact shard count, relative symlinks, forbidden-name rejection, resolved hashes, YAML path existence, and missing norm-stat failure.
- [ ] Run the focused test and verify it fails because builder/validator are absent.
- [ ] Implement the minimal builder and validator, then generate the adapter around the original shards.
- [ ] Run the focused test and verify it passes.

### Task 3: Compute clean-only normalization and smoke-test loading

**Files:**
- Create: `baseline_base_model_adapter/configs/compute_norm_stats.yaml`
- Create: `baseline_base_model_adapter/tools/smoke_load.py`
- Modify: `baseline_base_model_adapter/README.md`

**Interfaces:**
- Consumes: official clean LeRobot v3 dataset and generated adapter.
- Produces: `assets/norm_stats/robotwin_clean_50x50.json` and a strict-load smoke-test result.

- [ ] Write tests asserting the norm configuration resolves exactly one clean dataset with 2,500 episodes and the smoke loader refuses an absent stats file.
- [ ] Run the focused test and verify the expected failures.
- [ ] Compute full clean-only stats, finish the YAML, and implement strict-load smoke validation without running simulator episodes.
- [ ] Run the focused tests, adapter validation, YAML parsing, and strict GPU load smoke test.

### Task 4: Final preflight and handoff commands

**Files:**
- Create: `baseline_base_model_adapter/run_clean_50x50.sh`
- Create: `baseline_base_model_adapter/run_randomized_50x50.sh`
- Create: `baseline_base_model_adapter/preflight.sh`
- Modify: `baseline_base_model_adapter/README.md`

**Interfaces:**
- Consumes: verified adapter and the two conda environments.
- Produces: non-destructive preflight plus user-run clean/randomized benchmark commands.

- [ ] Write tests asserting both scripts select 50 tasks, 50 episodes, one GPU/server, BF16, no video, separate output roots, and no forbidden checkpoint string.
- [ ] Run the tests and verify they fail before scripts exist.
- [ ] Add the minimum preflight and two run scripts.
- [ ] Run all focused tests, the repository test command, shell syntax checks, adapter validation, and preflight; report any unrelated existing failures explicitly.

