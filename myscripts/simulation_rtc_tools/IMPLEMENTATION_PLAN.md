# LingBot-VLA RoboTwin Fast-Simulation RTC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated, deterministic fast-simulation RTC evaluation path for `models/lingbot_action_step2000` without changing the official synchronous RoboTwin evaluation path.

**Architecture:** A pure step-indexed scheduler controls one asynchronous websocket worker. A copied evaluation client integrates the scheduler only at boundaries between official `TASK_ENV.take_action()` calls, while an RTC-enabled checkpoint server performs model-space guidance and returns physical 50-step RoboTwin chunks.

**Tech Stack:** Python 3.10, NumPy, PyTorch, LingBot-VLA, RoboTwin/SAPIEN, websockets, PyYAML, pytest, Bash.

**Spec:** `myscripts/simulation_rtc_tools/DESIGN.md`

## Global Constraints

- Keep `RoboTwin_lingbot/script/eval_policy_client_lingbotvla.py`, `evaluation/lingbot_action_step2000/`, and `myscripts/inference_tools/` unchanged.
- Use checkpoint `models/lingbot_action_step2000`, its exported `robotwin.yaml`, and its clean normalization statistics.
- Fix `chunk_horizon=50`, `request_index=15`, `fixed_delay_steps=5`, `execution_horizon=10`, `crossfade_steps=4`, and `ema_alpha=0.36`.
- Never call `TASK_ENV.take_action()` or advance scheduler policy steps while waiting for inference.
- Switch chunks only between completed `take_action()` calls.
- Preserve official task configs, seeds, instructions, `step_lim`, TOPP, physics, success checks, and result accounting.
- Never silently fall back from RTC to ordinary sampling after an RTC request error.
- Do not start a 50×50 evaluation during implementation or verification.

## Review Focus

- A response arriving before five policy steps must be cached but not promoted; Task 3 tests this.
- A response arriving after the eligibility boundary must pause without consuming a step; Tasks 3 and 6 test this.
- A stale response from a previous episode/request must never control the robot; Tasks 2 and 4 test this.
- Malformed/NaN/wrong-dimensional actions must fail before `take_action()`; Tasks 2 and 6 test this.
- Shutdown during a blocked websocket request must terminate cleanly and preserve the outer evaluator; Task 4 tests this.

---

### Task 1: Validated RTC configuration and exported robot action layout

**Files:**
- Create: `myscripts/simulation_rtc_tools/configs/lingbot_action_step2000_rtc_fast.yaml`
- Create: `myscripts/simulation_rtc_tools/simulation_rtc/__init__.py`
- Create: `myscripts/simulation_rtc_tools/simulation_rtc/config.py`
- Test: `myscripts/simulation_rtc_tools/tests/test_config.py`

**Interfaces:**
- Produces: `RTCConfig.from_yaml(path: Path) -> RTCConfig`
- Produces: `RobotActionLayout.from_yaml(path: Path) -> RobotActionLayout`
- Produces: `RobotActionLayout.action_dim: int` and `.gripper_indices: tuple[int, ...]`

- [ ] **Step 1: Write failing configuration tests**

Assert the shipped YAML loads exact fixed values; invalid bounds fail; the exported checkpoint config resolves `action_dim == 14` and `gripper_indices == (6, 13)`; missing/overlapping action slices fail.

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest -q myscripts/simulation_rtc_tools/tests/test_config.py`

Expected: FAIL because `simulation_rtc.config` does not exist.

- [ ] **Step 3: Implement immutable config dataclasses and strict YAML parsing**

Implement exact signatures from Interfaces. Resolve gripper indices from `actions -> action.effector.position -> origin_keys`; derive total action dimension from all action slices rather than hard-coding it.

- [ ] **Step 4: Run tests and verify pass**

Run: `pytest -q myscripts/simulation_rtc_tools/tests/test_config.py`

Expected: PASS.

### Task 2: Versioned RTC wire protocol and validation

**Files:**
- Create: `myscripts/simulation_rtc_tools/simulation_rtc/protocol.py`
- Test: `myscripts/simulation_rtc_tools/tests/test_protocol.py`

**Interfaces:**
- Consumes: `RTCConfig`, `RobotActionLayout`
- Produces: `RTCRequest`, `RTCResponse`, `ProtocolError`
- Produces: `build_bootstrap_request(...) -> RTCRequest`
- Produces: `build_rtc_request(...) -> RTCRequest`
- Produces: `validate_response(payload: dict, expected_episode_id: int, expected_request_id: int, layout: RobotActionLayout) -> RTCResponse`

- [ ] **Step 1: Write failing protocol tests**

Cover bootstrap and RTC payload fields, monotonically scoped IDs, finite `[50, 14]` responses, wrong version, stale IDs, wrong shape, and NaN/Inf.

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest -q myscripts/simulation_rtc_tools/tests/test_protocol.py`

Expected: FAIL because protocol types do not exist.

- [ ] **Step 3: Implement serialization-safe immutable protocol types**

Use protocol version `lingbot-robotwin-rtc/1`; preserve observation arrays as the existing websocket codec expects; include RTC parameters and server acknowledgement metadata.

- [ ] **Step 4: Run tests and verify pass**

Run: `pytest -q myscripts/simulation_rtc_tools/tests/test_protocol.py`

Expected: PASS.

### Task 3: Deterministic policy-step scheduler and physical-action blending

**Files:**
- Create: `myscripts/simulation_rtc_tools/simulation_rtc/scheduler.py`
- Create: `myscripts/simulation_rtc_tools/simulation_rtc/action_blender.py`
- Test: `myscripts/simulation_rtc_tools/tests/test_scheduler.py`
- Test: `myscripts/simulation_rtc_tools/tests/test_action_blender.py`

**Interfaces:**
- Consumes: `RTCConfig`, `RTCResponse`, `RobotActionLayout`
- Produces: `RTCScheduler.start_episode(episode_id: int) -> None`
- Produces: `RTCScheduler.install_bootstrap(chunk: np.ndarray) -> None`
- Produces: `RTCScheduler.before_action() -> SchedulerDecision`
- Produces: `RTCScheduler.after_action() -> None`
- Produces: `RTCScheduler.accept_response(response: RTCResponse) -> bool`
- Produces: `crossfade_chunks(old_chunk, old_index, new_chunk, steps) -> np.ndarray`
- Produces: `ema_action(previous, current, alpha, excluded_indices) -> np.ndarray`

- [ ] **Step 1: Write failing state-machine and math tests**

Assert one request at index 15, no promotion through elapsed steps 0–4, fixed promotion at index 20 when ready, WAITING_RESULT without index movement when not ready, five-step response crop, no out-of-range action reuse, stale rejection, completion cancellation, RLinf XFade4 parity, EMA0.36, and gripper exclusion `(6, 13)`.

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest -q myscripts/simulation_rtc_tools/tests/test_scheduler.py myscripts/simulation_rtc_tools/tests/test_action_blender.py`

Expected: FAIL because scheduler/blender do not exist.

- [ ] **Step 3: Implement the pure scheduler and blending functions**

Make `before_action()` side-effect free except for an explicit one-time request decision; increment execution only in `after_action()` after the caller confirms `take_action()` returned.

- [ ] **Step 4: Run tests and verify pass**

Run: same pytest command. Expected: PASS.

### Task 4: Single-owner asynchronous websocket worker

**Files:**
- Create: `myscripts/simulation_rtc_tools/simulation_rtc/async_worker.py`
- Test: `myscripts/simulation_rtc_tools/tests/test_async_worker.py`

**Interfaces:**
- Consumes: `RTCRequest`, `RTCResponse`, existing websocket client codec
- Produces: `AsyncPolicyWorker.start()`, `.submit(request)`, `.poll()`, `.wait(timeout_s)`, `.reset_episode(episode_id)`, and `.close(timeout_s)`

- [ ] **Step 1: Write failing worker tests with a fake blocking transport**

Cover single connection ownership, queue capacity one, one in-flight request, response delivery, stale response discard after reset, exception propagation, and bounded shutdown during a blocked request.

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest -q myscripts/simulation_rtc_tools/tests/test_async_worker.py`

Expected: FAIL because the worker does not exist.

- [ ] **Step 3: Implement the worker around an injected transport factory**

The worker thread alone creates and calls the websocket transport. Use bounded queues/events and do not import RoboTwin or touch `TASK_ENV`.

- [ ] **Step 4: Run tests and verify pass**

Run: same pytest command. Expected: PASS.

### Task 5: RTC-enabled exported-checkpoint server

**Files:**
- Create: `myscripts/simulation_rtc_tools/server/__init__.py`
- Create: `myscripts/simulation_rtc_tools/server/policy_server_rtc.py`
- Test: `myscripts/simulation_rtc_tools/tests/test_policy_server_contract.py`

**Interfaces:**
- Consumes: `evaluation.lingbot_action_step2000.policy_server.ExportedCheckpointServer`
- Consumes: `LingBotRTCRequest`, `install_rtc_sampler`, and `sample_actions_rtc` from `myscripts/inference_tools/lingbot_rtc/`
- Produces: `ExportedCheckpointRTCServer.infer(observation: dict) -> dict`
- Produces: CLI compatible with the existing launcher: `--model_path`, `--use_length`, `--port`, `--use_bf16`, `--use_fp32`, `--use_compile`

- [ ] **Step 1: Write failing server contract tests using fake transforms/model tensors**

Assert bootstrap uses ordinary sampling, RTC requests use `sample_actions_rtc`, the previous physical chunk is converted through the checkpoint FeatureTransform/normalizer, `[1,50,max_action_dim]` target and valid-action mask are built, physical output is `[50,14]`, and RTC acknowledgement IDs/parameters are echoed.

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest -q myscripts/simulation_rtc_tools/tests/test_policy_server_contract.py`

Expected: FAIL because the RTC server does not exist.

- [ ] **Step 3: Implement a local subclass/copy without monkey-patching official classes**

Install the RTC sampler only on the local loaded model. Reuse the exact exported robot config and normalization path loaded by the step2000 server. Keep ordinary sampling unchanged for bootstrap and compiled ordinary calls; permit RTC calls to use the uncompiled RTC denoiser.

- [ ] **Step 4: Run CPU contract tests and verify pass**

Run: same pytest command. Expected: PASS without loading the 6B checkpoint.

### Task 6: Copied RoboTwin evaluation client integration

**Files:**
- Create: `myscripts/simulation_rtc_tools/copied_eval/eval_policy_client_lingbotvla_rtc.py`
- Create: `myscripts/simulation_rtc_tools/simulation_rtc/eval_loop.py`
- Create: `myscripts/simulation_rtc_tools/simulation_rtc/metrics.py`
- Test: `myscripts/simulation_rtc_tools/tests/test_eval_loop.py`
- Test: `myscripts/simulation_rtc_tools/tests/test_official_isolation.py`

**Interfaces:**
- Consumes: scheduler, worker, protocol, blender, official `TASK_ENV`
- Produces: `run_rtc_episode(task_env, worker, config, layout, trace_writer) -> EpisodeRTCResult`
- Produces: one JSONL trace record per request plus one episode summary

- [ ] **Step 1: Snapshot SHA-256 of protected synchronous files in the isolation test fixture**

Protect the official client, step2000 server/launch scripts, and existing real-robot RTC files selected by the test manifest.

- [ ] **Step 2: Write failing fake-environment integration tests**

Use a fake `take_action_cnt`, `step_lim`, success flag, observations, and worker. Assert observation refresh at index 15, no `take_action()` while waiting at index 20, exact one-count increments per action, five-step crop, boundary-only promotion, malformed response rejection before actuation, success/limit termination, and trace fields.

- [ ] **Step 3: Run tests and verify failure**

Run: `pytest -q myscripts/simulation_rtc_tools/tests/test_eval_loop.py myscripts/simulation_rtc_tools/tests/test_official_isolation.py`

Expected: FAIL because the copied client/eval loop do not exist.

- [ ] **Step 4: Copy the official client with provenance and replace only its action policy loop**

Keep setup, seed filtering, instructions, video, success statistics, close/cache, and CLI parsing structurally identical. Import the isolated RTC package via an explicit tool-root path supplied by the launcher.

- [ ] **Step 5: Implement JSONL metrics and run tests**

Run: same pytest command. Expected: PASS and all protected hashes unchanged.

### Task 7: Isolated launchers, preflight, and user documentation

**Files:**
- Create: `myscripts/simulation_rtc_tools/scripts/run_rtc_fast_one_task.sh`
- Create: `myscripts/simulation_rtc_tools/scripts/run_rtc_fast_50x50.sh`
- Create: `myscripts/simulation_rtc_tools/scripts/preflight.sh`
- Create: `myscripts/simulation_rtc_tools/README.md`
- Test: `myscripts/simulation_rtc_tools/tests/test_entrypoints.py`

**Interfaces:**
- Consumes: checkpoint, config, copied client, RTC server
- Produces: dry-run commands and result tree under `myscripts/simulation_rtc_tools/results/`

- [ ] **Step 1: Write failing shell-entrypoint tests**

Assert executable scripts, dry-run support, unique RTC client destination, RTC server path, one-task defaults, 50×50 defaults, no video by default, no protected-file overwrite, and no episode start in preflight.

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest -q myscripts/simulation_rtc_tools/tests/test_entrypoints.py`

Expected: FAIL because scripts do not exist.

- [ ] **Step 3: Implement preflight and launchers**

If the upstream launcher cannot select a unique evaluation client, create a local launcher copy in this directory and change only client selection/copy destination. Do not patch the upstream launcher.

- [ ] **Step 4: Write README with exact safe commands and result interpretation**

Document preflight, dry-run, single-episode smoke, formal invocation, fixed-step timing semantics, JSONL audit fields, and the distinction from official synchronous baseline results.

- [ ] **Step 5: Run tests and verify pass**

Run: same pytest command. Expected: PASS.

### Task 8: Full verification and one real smoke episode

**Files:**
- Modify only as required by failures inside `myscripts/simulation_rtc_tools/`

**Interfaces:**
- Consumes: all previous tasks
- Produces: verified tool and one explicit user command for later 50×50 execution

- [ ] **Step 1: Run the complete isolated unit suite**

Run: `pytest -q myscripts/simulation_rtc_tools/tests`

Expected: all tests PASS.

- [ ] **Step 2: Run existing real-robot RTC and step2000 entrypoint regression suites**

Run: `pytest -q myscripts/inference_tools/tests evaluation/lingbot_action_step2000/tests`

Expected: all existing tests PASS.

- [ ] **Step 3: Run RTC preflight and launcher dry-run**

Run: `bash myscripts/simulation_rtc_tools/scripts/preflight.sh` and `bash myscripts/simulation_rtc_tools/scripts/run_rtc_fast_one_task.sh lift_pot --dry-run`.

Expected: assets/environments/config/provenance pass; no episode starts.

- [ ] **Step 4: Run one real `lift_pot` episode only**

Run: `bash myscripts/simulation_rtc_tools/scripts/run_rtc_fast_one_task.sh lift_pot`.

Expected: episode terminates normally; trace shows request index 15, eligibility and promotion at 20, crop_steps=5, no action-count advance during wait, and RTC server acknowledgement.

- [ ] **Step 5: Re-run protected-file hash and full isolated tests**

Expected: protected files unchanged and all tests PASS.

- [ ] **Step 6: Hand off the formal command without launching it**

Provide: `bash myscripts/simulation_rtc_tools/scripts/run_rtc_fast_50x50.sh`.
