# LingBot-VLA RLinf-Style RTC Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated LingBot-VLA 2.0 real-robot inference module that embeds RLinf-compatible RTC into the official flow-matching sampler and preserves the validated asynchronous R1 Pro smoothing and safety pipeline.

**Architecture:** Copy the required official deployment and RLinf control files into `myscripts/inference_tools`, record their provenance, and modify only the copies. A LingBot RTC sampler wraps the official `FlowMatchingV2.predict_velocity` denoising loop, while the copied ROS client retains q16/K20 scheduling, delay crop, XFade4, EMA0.36, gripper stabilization, and safety interlocks.

**Tech Stack:** Python 3.10+, PyTorch, LingBot-VLA 2.0, Qwen3-VL, NumPy, WebSockets/msgpack, ROS2 Humble, pytest, Bash.

**Spec:** `docs/superpowers/specs/2026-10-09-lingbot-rlinf-rtc-inference-design.md`

## Global Constraints

- Do not modify `/home/lixiang/PROJECT/RLinf`.
- Do not modify `/home/lixiang/PROJECT/LingBotVLA2.0/lingbot-vla-v2`.
- Copy every upstream file before modification and record source commit/path/hash in `provenance.json`.
- Load only LingBot-VLA checkpoints; the new server must not load OpenPI models or checkpoints.
- Preserve action horizon 50, control rate 15 Hz, K20, q16, RTC horizon 10, delay cap 4, XFade4, and EMA alpha 0.36.
- Exclude RECAP, dataset writing, keyboard episode management, and VR intervention.
- Default to dry-run; prohibit `--execute` until trained R1 Pro LingBot normalization/config assets pass validation.
- The workspace root is not currently a Git repository. Run all verification steps, but do not invent commit operations; commit only if the user later initializes a repository.

## Review Focus

- A stale or malformed RTC prefix must be rejected before model sampling; Task 2 tests shape, finite values, and prefix bounds.
- RTC-disabled inference must remain numerically equivalent to official LingBot inference; Task 3 adds fixed-noise parity tests.
- Late, failed, or duplicated asynchronous responses must never block the ROS control loop or promote stale chunks; Task 6 tests these schedules.
- Missing production normalization/config assets must make `--execute` fail closed while preserving dry-run tests; Task 7 pins this interlock.
- Server/client metadata or action-shape mismatch must prevent chunk promotion; Tasks 5 and 7 test both handshake and runtime rejection.

---

## File Structure

Create the following independent module:

```text
myscripts/inference_tools/
├── README.md
├── provenance.json
├── sync_upstream_sources.py
├── run_lingbot_rtc_policy_server.sh
├── serve_lingbot_rtc_policy.py
├── websocket_client_policy.py
├── websocket_policy_server.py
├── msgpack_numpy.py
├── r1pro_ros_client_node.py
├── r1pro_constants.py
├── r1pro_action_utils.py
├── r1pro_safety.py
├── configs/
│   ├── formal_r1pro_rtc.yaml
│   └── r1pro_lingbot.yaml.example
├── lingbot_rtc/
│   ├── __init__.py
│   ├── rtc_request.py
│   ├── rtc_sampler.py
│   ├── modeling_lingbot_vla_v2_rtc.py
│   └── lingbot_vla_v2_rtc_policy.py
└── tests/
    ├── conftest.py
    ├── test_provenance.py
    ├── test_rtc_math.py
    ├── test_official_parity.py
    ├── test_policy_contract.py
    ├── test_websocket_contract.py
    ├── test_async_schedule.py
    ├── test_smoothing.py
    └── test_execution_interlock.py
```

`modeling_lingbot_vla_v2_rtc.py` owns only the RTC-enabled inference subclass/override; it must not duplicate the entire 1,300-line official model unless a test proves subclassing cannot preserve the weight/state interface.

---

### Task 1: Provenance and isolated source copies

**Files:**
- Create: `myscripts/inference_tools/sync_upstream_sources.py`
- Create: `myscripts/inference_tools/provenance.json`
- Create: `myscripts/inference_tools/tests/test_provenance.py`
- Copy: RLinf WebSocket, R1 Pro action, constants, safety, and ROS client files into `myscripts/inference_tools/`

**Interfaces:**
- Produces: `sha256_file(path: Path) -> str`
- Produces: `verify_provenance(manifest_path: Path) -> list[str]`, returning validation errors.
- Produces: immutable copied baselines used by later tasks.

- [ ] **Step 1: Write failing provenance tests**

Test that the manifest names every copied file, records RLinf commit `e14297c0`, records each source path and source hash, and reports a mismatch after a temporary copied file is altered.

- [ ] **Step 2: Run the provenance test and confirm failure**

Run: `pytest -q myscripts/inference_tools/tests/test_provenance.py`

Expected: FAIL because the manifest and verifier do not exist.

- [ ] **Step 3: Implement the copy/provenance utility**

Implement exact-source copying with refusal to overwrite an already modified target. Record source and destination pre-modification SHA-256 values. Include the official LingBot deploy wrapper as a provenance source even though the RTC policy copy is introduced in Task 4.

- [ ] **Step 4: Copy the RLinf baseline files**

Copy only the formal, non-`exp` inference files. Do not copy `record_tools`, generated ROS build directories, checkpoints, or historical archives.

- [ ] **Step 5: Run provenance tests**

Run: `pytest -q myscripts/inference_tools/tests/test_provenance.py`

Expected: PASS.

---

### Task 2: RTC request model and mathematical kernel

**Files:**
- Create: `myscripts/inference_tools/lingbot_rtc/rtc_request.py`
- Create: `myscripts/inference_tools/lingbot_rtc/rtc_sampler.py`
- Create: `myscripts/inference_tools/lingbot_rtc/__init__.py`
- Create: `myscripts/inference_tools/tests/test_rtc_math.py`

**Interfaces:**
- Produces: `LingBotRTCRequest` frozen dataclass with `target`, `prefix_length`, `inference_delay`, `execution_horizon`, `max_guidance_weight`, `prefix_attention_schedule`, and `valid_action_mask`.
- Produces: `build_rtc_prefix_weights(start, end, total, schedule, *, device, dtype) -> Tensor`.
- Produces: `rtc_guidance_weight(time, max_guidance_weight) -> Tensor`.
- Produces: `apply_rtc_velocity_guidance(base_velocity_fn, x_t, time, request) -> Tensor`.

- [ ] **Step 1: Write failing unit tests for schedules and validation**

Cover linear/exp/ones/zeros schedules, invalid schedule, negative/bounds-clamped indices, disabled request, malformed target shape, non-finite target, and mask/target shape mismatch.

- [ ] **Step 2: Run tests and confirm failure**

Run: `pytest -q myscripts/inference_tools/tests/test_rtc_math.py`

Expected: FAIL on missing modules.

- [ ] **Step 3: Implement the immutable request and validation**

Require target shape `[B, 50, max_action_dim]`, mask broadcastability to the same shape, `0 <= inference_delay <= execution_horizon <= prefix_length <= 50`, and finite target/weight values.

- [ ] **Step 4: Port the RLinf RTC math**

Preserve the official linear/exponential/hard schedule equations and VJP correction. Apply `valid_action_mask` to the overlap error so padded dimensions receive no guidance.

- [ ] **Step 5: Run RTC math tests**

Run: `pytest -q myscripts/inference_tools/tests/test_rtc_math.py`

Expected: PASS with no parameter gradients retained after a call.

---

### Task 3: RTC-enabled LingBot FlowMatchingV2 inference

**Files:**
- Create: `myscripts/inference_tools/lingbot_rtc/modeling_lingbot_vla_v2_rtc.py`
- Create: `myscripts/inference_tools/tests/test_official_parity.py`

**Interfaces:**
- Consumes: `LingBotRTCRequest`, `apply_rtc_velocity_guidance` from Task 2.
- Produces: `sample_actions_rtc(flow_model, images, img_masks, lang_tokens, lang_masks, state, *, rtc_request=None, noise=None, image_grid_thw=None) -> Tensor`.
- Produces: `install_rtc_sampler(policy) -> None`, attaching the copied RTC entry without changing parameter names or state dict keys.

- [ ] **Step 1: Write the fixed-noise parity test**

Using a lightweight fake FlowMatchingV2 with the same callable surface, assert RTC-disabled output matches its official sampler step-for-step and RTC-enabled output is finite and differs only where the valid prefix mask permits.

- [ ] **Step 2: Write an integration parity test guarded by local model availability**

With a deterministic observation and noise, compare official LingBot `sample_actions` against the copied RTC entry with RTC disabled. Mark GPU/model-dependent execution explicitly; do not silently pass when invoked in the full integration profile.

- [ ] **Step 3: Run tests and confirm failure**

Run: `pytest -q myscripts/inference_tools/tests/test_official_parity.py -m 'not gpu'`

Expected: FAIL because the RTC sampler entry does not exist.

- [ ] **Step 4: Implement the copied RTC sampling path**

Reuse official prefix embedding, attention mask, KV-cache, `predict_velocity`, time step, and Euler update behavior. Insert VJP guidance after base velocity prediction and before `x_t += dt * velocity`. When RTC is disabled, take the official no-grad path without enabling gradients.

- [ ] **Step 5: Run CPU parity tests**

Run: `pytest -q myscripts/inference_tools/tests/test_official_parity.py -m 'not gpu'`

Expected: PASS.

---

### Task 4: LingBot checkpoint policy and deferred R1 Pro transform

**Files:**
- Create: `myscripts/inference_tools/lingbot_rtc/lingbot_vla_v2_rtc_policy.py`
- Create: `myscripts/inference_tools/configs/r1pro_lingbot.yaml.example`
- Create: `myscripts/inference_tools/tests/test_policy_contract.py`

**Interfaces:**
- Consumes: RTC sampler installer from Task 3.
- Produces: `LingBotVlaV2RTCPolicy.__init__(model_path, qwen_path, robot_config_path=None, norm_stats_path=None, *, use_bf16=True, use_compile=False)`.
- Produces: `infer(observation: dict) -> dict`, returning `actions` and `policy_timing`.
- Produces: `production_assets_ready: bool` and `production_asset_errors: tuple[str, ...]`.

- [ ] **Step 1: Write failing policy contract tests**

Test exact three-camera keys, 20D state, non-empty prompt, optional RTC payload, full 50-step output, finite actions, strict backend metadata, and failure on malformed inputs.

- [ ] **Step 2: Add tests for deferred assets**

Assert missing robot config/norm stats sets `production_assets_ready=False` without blocking mock/model-space inference. Assert physical 20D output is unavailable until the transform passes validation.

- [ ] **Step 3: Run tests and confirm failure**

Run: `pytest -q myscripts/inference_tools/tests/test_policy_contract.py`

Expected: FAIL because the policy copy does not exist.

- [ ] **Step 4: Copy and adapt the official deployment policy**

Preserve strict LingBot weight loading and Qwen3-VL setup. Replace the call to official sampling with Task 3's RTC-aware call. Do not load OpenPI or import RLinf OpenPI model code.

- [ ] **Step 5: Implement the transform boundary**

Define explicit methods `physical_observation_to_model`, `physical_rtc_target_to_model`, and `model_actions_to_physical`. In the initial version, fail closed for physical inference when production assets are absent; allow injected fake transforms for tests.

- [ ] **Step 6: Run policy tests**

Run: `pytest -q myscripts/inference_tools/tests/test_policy_contract.py`

Expected: PASS.

---

### Task 5: WebSocket server and launch scripts

**Files:**
- Create: `myscripts/inference_tools/serve_lingbot_rtc_policy.py`
- Create: `myscripts/inference_tools/run_lingbot_rtc_policy_server.sh`
- Create: `myscripts/inference_tools/tests/test_websocket_contract.py`
- Modify copied: `myscripts/inference_tools/websocket_policy_server.py`
- Modify copied: `myscripts/inference_tools/websocket_client_policy.py`

**Interfaces:**
- Consumes: `LingBotVlaV2RTCPolicy` from Task 4.
- Produces: `/healthz` endpoint and msgpack inference endpoint.
- Produces server metadata containing `model_backend`, `action_horizon`, `physical_action_dim`, `rtc_capable`, `checkpoint_id`, and `normalization_id`.

- [ ] **Step 1: Write failing WebSocket contract tests**

Test metadata handshake, health response, bootstrap inference, RTC replan inference, string traceback rejection, wrong backend rejection, and wrong output-shape rejection.

- [ ] **Step 2: Run tests and confirm failure**

Run: `pytest -q myscripts/inference_tools/tests/test_websocket_contract.py`

Expected: FAIL on missing server entry.

- [ ] **Step 3: Implement the server entry**

Require explicit model and Qwen paths. Support BF16/FP32 exclusivity, optional compile, configurable port, and dry-run/model-space smoke mode. Print resolved paths and checkpoint identity without printing secrets.

- [ ] **Step 4: Implement the Bash launcher**

Set only target-specific `PYTHONPATH` entries. Do not modify Conda/CUDA installations. Refuse missing model/Qwen paths before Python startup.

- [ ] **Step 5: Run WebSocket tests**

Run: `pytest -q myscripts/inference_tools/tests/test_websocket_contract.py`

Expected: PASS.

---

### Task 6: Preserve RLinf async scheduling and smoothing behavior

**Files:**
- Modify copied: `myscripts/inference_tools/r1pro_ros_client_node.py`
- Verify copied: `myscripts/inference_tools/r1pro_constants.py`
- Verify copied: `myscripts/inference_tools/r1pro_action_utils.py`
- Verify copied: `myscripts/inference_tools/r1pro_safety.py`
- Create: `myscripts/inference_tools/configs/formal_r1pro_rtc.yaml`
- Create: `myscripts/inference_tools/tests/test_async_schedule.py`
- Create: `myscripts/inference_tools/tests/test_smoothing.py`

**Interfaces:**
- Consumes: Task 5 WebSocket protocol.
- Produces unchanged RLinf control behavior with LingBot backend identity checks.

- [ ] **Step 1: Write deterministic schedule tests**

With a fake clock/server, assert first 16 actions execute before prefetch, ready chunks promote at K20, late chunks continue the old horizon, stale responses do not promote, only one request is in flight, and horizon exhaustion holds the last safe action.

- [ ] **Step 2: Write smoothing tests**

Assert four SmoothStep coefficients and blended arm/torso outputs, EMA `0.64*previous + 0.36*current`, exclusion of gripper dimensions 7 and 15, seven-frame debounce, and seven-frame reversal lock.

- [ ] **Step 3: Run tests and confirm any adaptation failures**

Run: `pytest -q myscripts/inference_tools/tests/test_async_schedule.py myscripts/inference_tools/tests/test_smoothing.py`

Expected: tests expose only import/path/backend naming differences; mathematical behavior from the copied baseline already passes.

- [ ] **Step 4: Apply minimal LingBot-specific client changes**

Change names/help text and require server metadata `model_backend=lingbot-vla-v2`. Do not restructure the timer, async thread, RTC payload, crop, smoothing, gripper, or safety code.

- [ ] **Step 5: Add and validate formal YAML**

Store the exact formal values and reject q>=K, K>50, RTC horizon>50, invalid EMA alpha, and non-positive rates.

- [ ] **Step 6: Run schedule and smoothing tests**

Run: `pytest -q myscripts/inference_tools/tests/test_async_schedule.py myscripts/inference_tools/tests/test_smoothing.py`

Expected: PASS.

---

### Task 7: Fail-closed execution interlock

**Files:**
- Create: `myscripts/inference_tools/tests/test_execution_interlock.py`
- Modify: `myscripts/inference_tools/r1pro_ros_client_node.py`
- Modify: `myscripts/inference_tools/serve_lingbot_rtc_policy.py`

**Interfaces:**
- Consumes: policy asset readiness and server metadata from Tasks 4 and 5.
- Produces: `validate_execute_readiness(client_config, server_metadata) -> tuple[bool, tuple[str, ...]]`.

- [ ] **Step 1: Write failing execution-interlock tests**

Reject execution for missing norm stats, missing robot config, unvalidated 20D schema, non-LingBot backend, RTC-incapable server, horizon/dimension mismatch, and non-finite warmup output. Permit dry-run under the same missing-asset conditions.

- [ ] **Step 2: Run tests and confirm failure**

Run: `pytest -q myscripts/inference_tools/tests/test_execution_interlock.py`

Expected: FAIL because the readiness validator does not exist.

- [ ] **Step 3: Implement the validator and startup gate**

Run the gate before creating ROS target publishers or accepting `--execute`. Print every failed condition and exit nonzero; never silently downgrade an explicitly requested execute run.

- [ ] **Step 4: Run interlock tests**

Run: `pytest -q myscripts/inference_tools/tests/test_execution_interlock.py`

Expected: PASS.

---

### Task 8: Documentation, smoke checks, and final verification

**Files:**
- Create: `myscripts/inference_tools/README.md`
- Create: `myscripts/inference_tools/tests/conftest.py`
- Modify: `myscripts/inference_tools/provenance.json`

**Interfaces:**
- Produces documented model-only smoke, server/client smoke, ROS dry-run, and deferred production-asset workflows.

- [ ] **Step 1: Document the supported state**

Explain that RTC/model/protocol/control tests are available now, while real execution remains locked until trained R1 Pro assets are supplied. Document all formal parameters and expected logs.

- [ ] **Step 2: Add smoke commands**

Provide commands for provenance verification, CPU unit tests, GPU official-parity test, model server health check, fake-server async test, and ROS dry-run. Do not provide an executable real-motion command before the asset gate exists.

- [ ] **Step 3: Verify no source repository was modified**

Run read-only Git status checks in RLinf and `lingbot-vla-v2`; compare recorded source hashes. Any unexpected modification fails completion.

- [ ] **Step 4: Run the full non-hardware test suite**

Run: `pytest -q myscripts/inference_tools/tests`

Expected: all CPU tests pass; GPU/hardware tests are explicitly marked and reported, never silently swallowed.

- [ ] **Step 5: Run syntax and shell checks**

Run: `python3 -m compileall -q myscripts/inference_tools`

Run: `bash -n myscripts/inference_tools/run_lingbot_rtc_policy_server.sh`

Expected: both exit 0.

- [ ] **Step 6: Produce the implementation handoff**

Report created files, copied-source provenance, tests run, deferred assets, and the exact reasons real execution remains locked.
