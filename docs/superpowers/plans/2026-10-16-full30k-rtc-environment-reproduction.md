# Full30K RTC Environment Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Git-tracked, server-derived environment capture/create/verify workflow and make full30k FP32 RTC Clean-unseen with `20/10/20/10/0.10` the documented canonical evaluation.

**Architecture:** A small Python helper owns deterministic sanitization and manifest validation; thin shell entrypoints collect DRLServer facts, create the two Conda environments, and invoke existing runtime checks. Portable environment files are generated only from the verified DRLServer and are never fabricated from the local workstation. Large assets remain external and are described by a machine-readable manifest.

**Tech Stack:** Bash, Python 3 standard library, Conda CLI, pytest, JSON, YAML text emitted by Conda, existing LingBotVLA evaluation scripts.

**Spec:** `docs/superpowers/specs/2026-10-16-full30k-rtc-environment-reproduction-design.md`

## Global Constraints

- DRLServer's successfully exercised `lingbotvla` and `RoboTwin` prefixes are the authoritative environment source.
- Do not commit models, simulator assets, Conda packages, credentials, caches, or machine-specific absolute paths.
- Canonical evaluation is `full30k`, FP32, RTC, `demo_clean`, unseen, 50 tasks × 50 episodes, `20/10/20/10/0.10`.
- Target servers select physical GPU IDs and base port without editing repository code.
- Existing user-authored reproduction-guide content must be preserved while stale statements are corrected narrowly.
- Every behavior change follows RED-GREEN TDD.

## Review Focus

- A Conda export containing an absolute `prefix`, `file://` dependency, or editable local path must be rejected before tracked output is replaced (Task 1 tests).
- Export interruption must not leave partially rewritten tracked environment artifacts (Task 1 tests atomic staging).
- Environment creation must refuse accidental overwrite and must not silently change CUDA/PyTorch builds (Task 2 dry-run tests).
- GPU validation must inspect exactly the comma-separated IDs requested, including non-contiguous IDs, enforce a configurable FP32 free-memory threshold, and derive the matching port range (Task 3 tests).
- Asset verification must distinguish missing, incomplete, and wrong-index full30k checkpoints without hashing multi-gigabyte shards during every run (Task 3 tests).

---

### Task 1: Verified Environment Export and Sanitization

**Files:**
- Create: `evaluation/environment_reproduction.py`
- Create: `scripts/export_verified_environments.sh`
- Create: `environment/README.md`
- Create: `environment/system-background.md`
- Create: `environment/snapshots/.gitkeep`
- Test: `evaluation/eval_provenance/tests/test_environment_reproduction.py`

**Interfaces:**
- Produces: `sanitize_conda_export(text: str, name: str) -> str`, `find_nonportable_references(text: str) -> list[str]`, and CLI subcommand `sanitize-conda`.
- Produces: atomic exporter outputs named `lingbotvla.environment.yml`, `robotwin.environment.yml`, four snapshot files, and `server-runtime.txt`.
- Consumes: Conda prefixes supplied by `LINGBOTVLA_ENV_PREFIX` and `ROBOTWIN_ENV_PREFIX`, defaulting to the verified DRLServer paths.

- [ ] **Step 1: Write failing sanitization tests**

Test portable name replacement, prefix removal, ordinary pip preservation, rejection of `/home/...`, `file://`, and editable paths, and unchanged destination files when sanitization fails.

- [ ] **Step 2: Run tests and verify RED**

Run: `python3 -m pytest -q evaluation/eval_provenance/tests/test_environment_reproduction.py`
Expected: FAIL because the helper and exporter do not exist.

- [ ] **Step 3: Implement the helper and atomic export shell entrypoint**

Use temporary files under the destination filesystem, validate every staged artifact, then rename all outputs. Record repository commit and runtime facts without environment secrets.

- [ ] **Step 4: Run Task 1 tests and shell syntax checks**

Run: `python3 -m pytest -q evaluation/eval_provenance/tests/test_environment_reproduction.py && bash -n scripts/export_verified_environments.sh`
Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Commit message: `增加已验证环境导出工具`

### Task 2: Portable Environment Creation

**Files:**
- Create: `scripts/create_eval_environments.sh`
- Modify: `evaluation/environment_reproduction.py`
- Modify: `environment/README.md`
- Test: `evaluation/eval_provenance/tests/test_environment_reproduction.py`

**Interfaces:**
- Consumes: `environment/lingbotvla.environment.yml`, `environment/robotwin.environment.yml`, current checkout CuRobo source, and optional `CONDA_EXE`.
- Produces: CLI subcommand `plan-create` emitting deterministic creation commands and shell options `--dry-run`, `--update`, and `--conda`.

- [ ] **Step 1: Write failing creation-plan tests**

Test missing YAML refusal, existing environment refusal without `--update`, deterministic environment names, current-repository CuRobo install path, and dry-run behavior.

- [ ] **Step 2: Run tests and verify RED**

Run the Task 2 test selection and expect missing creation-planning behavior.

- [ ] **Step 3: Implement creation planning and shell execution**

The shell script must use Conda's environment update/create commands exactly as planned, install repository packages in their intended environments, and invoke verification only after successful installation.

- [ ] **Step 4: Run Task 2 tests and syntax checks**

Run helper tests plus `bash -n scripts/create_eval_environments.sh`.
Expected: PASS.

- [ ] **Step 5: Commit Task 2**

Commit message: `增加评测环境创建入口`

### Task 3: Full30K Asset, GPU, Port, and Runtime Verification

**Files:**
- Create: `environment/assets-full30k.json`
- Create: `scripts/verify_full30k_rtc_environment.sh`
- Modify: `evaluation/environment_reproduction.py`
- Modify: `environment/README.md`
- Test: `evaluation/eval_provenance/tests/test_environment_reproduction.py`

**Interfaces:**
- Produces: CLI subcommands `validate-assets --root PATH --manifest PATH` and `runtime-plan --gpu-ids CSV --base-port INT --min-free-mib INT`.
- Consumes: existing `evaluation/model_registry.sh`, `evaluation/prepare_runtime.sh`, full30k index, full5k configuration assets, Qwen and auxiliary weights, and RoboTwin asset paths.

- [ ] **Step 1: Write failing manifest and runtime-plan tests**

Test all required asset IDs, missing and `.incomplete` shards, index-referenced shard validation, exact non-contiguous GPU selection, invalid/duplicate GPU IDs, and port range `base..base+n-1`.

- [ ] **Step 2: Run tests and verify RED**

Expected: FAIL because validation subcommands do not exist.

- [ ] **Step 3: Implement manifest and read-only verifier**

The verifier checks assets, selected GPU free memory, port availability, imports, repository-local CuRobo, `prepare_runtime.sh --check-only`, and `preflight.sh --model full30k --deep`; it never installs or rewrites dependencies.

- [ ] **Step 4: Run Task 3 tests and syntax/JSON checks**

Run helper tests, `bash -n`, and `python3 -m json.tool environment/assets-full30k.json`.
Expected: PASS.

- [ ] **Step 5: Commit Task 3**

Commit message: `增加full30k运行环境验证`

### Task 4: Canonical RTC Defaults and Entrypoint Regression

**Files:**
- Modify: `myscripts/simulation_rtc_tools/configs/lingbot_action_step2000_rtc_fast.yaml`
- Modify: `myscripts/simulation_rtc_tools/scripts/preflight.sh`
- Modify: `myscripts/simulation_rtc_tools/tests/test_config.py`
- Modify: `myscripts/simulation_rtc_tools/tests/test_entrypoints.py`
- Modify: other RTC tests only where they assert the prior default.

**Interfaces:**
- Produces: default `ema_alpha=0.10` while preserving explicit `--ema-alpha` override behavior.
- Consumes: existing RTC config loader and all current entrypoints.

- [ ] **Step 1: Change tests to require default `0.10` and explicit override preservation**

- [ ] **Step 2: Run focused tests and verify RED against the old default**

Run: `python3 -m pytest -q myscripts/simulation_rtc_tools/tests/test_config.py myscripts/simulation_rtc_tools/tests/test_entrypoints.py`
Expected: FAIL only on old-default expectations.

- [ ] **Step 3: Change the YAML default and preflight assertion**

- [ ] **Step 4: Run the complete RTC test suite**

Run: `python3 -m pytest -q myscripts/simulation_rtc_tools/tests`
Expected: PASS.

- [ ] **Step 5: Commit Task 4**

Commit message: `更新RTC默认EMA参数为0.10`

### Task 5: Reproduction Guide and Canonical Commands

**Files:**
- Modify: `docs/SERVER_REPRODUCTION_GUIDE.md`
- Modify: `environment/README.md`
- Create: `scripts/run_full30k_rtc_clean_unseen.sh`
- Modify: `evaluation/environment_reproduction.py`
- Test: `evaluation/eval_provenance/tests/test_environment_reproduction.py`

**Interfaces:**
- Produces: a canonical wrapper accepting `--gpu-ids`, `--base-port`, `--episodes`, environment prefixes, and `--dry-run` while forwarding fixed benchmark semantics and explicit RTC parameters to the existing launcher.
- Consumes: Task 2 environment paths and Task 3 verifier/runtime plan.

- [ ] **Step 1: Write failing canonical-command tests**

Assert full30k, FP32, RTC, `demo_clean`, unseen, 50 episodes, selected GPU IDs, selected base port, and explicit `20/10/20/10/0.10`; assert no server absolute project path.

- [ ] **Step 2: Run tests and verify RED**

- [ ] **Step 3: Implement the wrapper and update documentation**

Correct stale model and multi-GPU statements, add environment export/create/verify flow, list full30k assets and sources, retain historical full5k/full15k content, and present canonical smoke/full/resume/video commands.

- [ ] **Step 4: Run documentation, shell, helper, model-registry, and entrypoint checks**

Run all touched suites, shell syntax checks, JSON validation, and `git diff --check`.
Expected: PASS.

- [ ] **Step 5: Commit Task 5**

Commit message: `完善full30k RTC一键复现指南`

### Task 6: Whole-Repository Verification and DRLServer Handoff

**Files:**
- Modify only files required by Critical/Important review findings, with a failing regression test first.

**Interfaces:**
- Consumes: all prior tasks.
- Produces: locally verified repository plus the exact DRLServer export and acceptance commands needed to populate authoritative environment artifacts.

- [ ] **Step 1: Run focused complete suites**

Run environment reproduction, model registry, provenance, and RTC suites; compile Python; check all touched shell scripts; parse YAML/JSON; run `git diff --check`.

- [ ] **Step 2: Perform a whole-branch review against the spec and Review Focus**

- [ ] **Step 3: Fix Critical/Important findings in one RED-GREEN pass**

- [ ] **Step 4: Provide the DRLServer export command and acceptance checklist**

The handoff must state clearly that authoritative YAML/snapshot content is not complete until the exporter runs on DRLServer and those generated text artifacts are reviewed and committed.

- [ ] **Step 5: Commit verified fixes, if any**

Commit message: `验证full30k RTC环境复现流程`
