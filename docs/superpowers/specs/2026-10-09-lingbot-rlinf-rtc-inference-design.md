# LingBot-VLA 2.0 RLinf-Style RTC Inference Design

## 1. Objective

Build a standalone real-robot inference module under:

```text
/home/lixiang/PROJECT/LingBotVLA2.0/myscripts/inference_tools
```

The module must load LingBot-VLA 2.0 weights while preserving the validated
RLinf R1 Pro control pipeline:

```text
asynchronous inference
  -> model-side RTC guidance
  -> queue-delay crop
  -> four-step SmoothStep cross-fade
  -> publish-side EMA
  -> gripper stabilizer
  -> safety limiter
  -> ROS2 publication
```

RECAP recording, dataset conversion, keyboard episode control, and VR
intervention are outside this module.

## 2. Non-Negotiable Constraints

1. The original RLinf repository remains unchanged.
2. The official `lingbot-vla-v2/` source tree remains unchanged.
3. Any official LingBot code requiring RTC integration is copied into the new
   module before modification.
4. Copied upstream files are accompanied by a provenance manifest containing:
   - source repository path;
   - source Git commit;
   - source relative path;
   - source SHA-256 before modification;
   - copied-file SHA-256 before modification.
5. The deployed model is LingBot-VLA 2.0. No OpenPI model or OpenPI checkpoint
   may be loaded by the new server.
6. The control client continues to consume complete 50-step action chunks.
7. Missing or unvalidated R1 Pro normalization assets must prevent real robot
   execution, but must not prevent model-only and protocol-level tests.

## 3. Source Baselines

### 3.1 RLinf control baseline

The behavioral reference is RLinf commit `e14297c0`, with the sealed formal
baseline:

```text
formal_async_rtc_k20_q16_d4_rtch10_xfade4_ema036_gripper_debounce7_lock7_20260826_193337
```

Formal parameters:

| Parameter | Value |
|---|---:|
| Control rate | 15 Hz |
| Model action horizon | 50 |
| Planned chunk switch, K | 20 |
| Async prefetch point, q | 16 |
| Planned handoff delay, K-q | 4 |
| RTC execution horizon | 10 |
| RTC maximum guidance weight | 10.0 |
| RTC prefix schedule | exponential |
| RTC initial delay estimate | 3 steps |
| RTC delay cap | 4 steps |
| Chunk cross-fade | 4 steps |
| Publish EMA alpha | 0.36 |
| Gripper debounce | 7 frames |
| Gripper reversal lock | 7 frames |
| Gripper thresholds | 20 / 60 |

The formal archive has SHA-256:

```text
9d501e5edfe8e6a96bdb9e71b46830c9fc78f5245cc0d5e85e7a911772c9517d
```

### 3.2 LingBot model baseline

The model implementation is taken from the local official checkout:

```text
lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/
```

The relevant official inference chain is:

```text
LingbotVlaV2Policy.sample_actions
  -> FlowMatchingV2.sample_actions
  -> FlowMatchingV2.predict_velocity
```

The official configuration predicts a 50-step action chunk with ten
flow-matching denoising steps.

## 4. Target Layout

```text
myscripts/inference_tools/
├── README.md
├── provenance.json
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
    ├── test_provenance.py
    ├── test_rtc_math.py
    ├── test_official_parity.py
    ├── test_policy_contract.py
    ├── test_async_schedule.py
    ├── test_smoothing.py
    ├── test_safety_interlock.py
    └── test_server_smoke.py
```

The target layout deliberately excludes RLinf `record_tools/` and all
experimental `*_exp*` clients.

## 5. Component Design

### 5.1 WebSocket protocol

The copied WebSocket client/server retain the RLinf/OpenPI-compatible msgpack
contract. The R1 Pro client sends:

```python
{
    "images": {
        "cam_high": uint8[H, W, 3],
        "cam_left_wrist": uint8[H, W, 3],
        "cam_right_wrist": uint8[H, W, 3],
    },
    "state": float32[20],
    "prompt": str,
    "rtc": {
        "enabled": bool,
        "prev_chunk_left_over": float32[T, 20] | None,
        "prev_chunk_left_over_length": int,
        "inference_delay": int,
        "execution_horizon": int,
        "max_guidance_weight": float,
        "prefix_attention_schedule": int,
    },
}
```

The server returns:

```python
{
    "actions": float32[50, 20],
    "policy_timing": {
        "infer_ms": float,
        "rtc_enabled": bool,
        "rtc_prefix_len": int,
        "rtc_inference_delay": int,
        "rtc_execution_horizon": int,
        "model_backend": "lingbot-vla-v2",
    },
    "server_timing": {...},
}
```

The protocol rejects missing cameras, non-finite state/actions, incorrect
dimensions, empty prompts, malformed RTC prefixes, and non-50-step output.

### 5.2 LingBot policy wrapper

`LingBotVlaV2RTCPolicy` is a copied and isolated version of the official deploy
wrapper. It is responsible for:

1. loading the LingBot training configuration and checkpoint shards;
2. loading Qwen3-VL from the configured local path;
3. constructing the robot feature transform when its assets are available;
4. converting the physical observation into LingBot model inputs;
5. converting the old physical action tail into normalized model space;
6. creating an immutable `RTCRequest`;
7. invoking the RTC-enabled LingBot sampler;
8. unnormalizing the complete output chunk to physical R1 Pro actions;
9. returning the action chunk and RTC confirmation metadata.

Model loading remains strict. Missing, unexpected, or shape-mismatched weights
are fatal.

### 5.3 Model-side RTC integration

`modeling_lingbot_vla_v2_rtc.py` provides the copied inference variant. It
reuses the official LingBot layers and weights and replaces only the sampling
entry with an RTC-aware path.

For every flow-matching denoising step:

```text
base_velocity = predict_velocity(x_t, time)
x1_estimate = x_t - time * base_velocity
error = (rtc_target - x1_estimate) * prefix_weights
correction = VJP(x1_estimate, x_t, grad_outputs=error)
guided_velocity = base_velocity - guidance_weight(time) * correction
x_t = x_t + dt * guided_velocity
```

Properties required of the implementation:

- no RTC request: numerically equivalent to official inference;
- RTC target is in the same normalized model space as `x_t`;
- the target is padded to `[B, 50, max_action_dim]`;
- guidance affects only valid action dimensions;
- hard elapsed prefix and soft overlap schedule follow the RLinf implementation;
- guidance weight is finite and clipped;
- invalid RTC input fails before sampling;
- RTC does not modify model parameters or optimizer state;
- inference does not retain autograd graphs after a denoising step.

The official prefix embedding and KV-cache construction are preserved. RTC is
introduced only between velocity prediction and the Euler update.

### 5.4 Asynchronous client and chunk lifecycle

The copied RLinf client preserves this lifecycle:

1. Bootstrap inference returns the first 50-step chunk.
2. Execute the current chunk at 15 Hz.
3. After 16 executed actions, start one background inference request.
4. Continue executing the old chunk while inference runs.
5. At action index 20, promote the new chunk if it is ready.
6. If it is not ready, continue the old chunk without blocking control.
7. On promotion, discard new actions that elapsed during inference.
8. Apply four-step SmoothStep blending at the chunk boundary.
9. Publish through EMA, gripper stabilization, and safety checks.
10. If the old 50-step horizon is exhausted, hold the last safe command until
    inference recovers.

Only one inference request may be in flight. Failures use bounded exponential
backoff and never block the ROS timer thread.

### 5.5 Cross-fade, EMA, gripper, and safety

SmoothStep cross-fade applies to arm and torso dimensions:

```text
alpha(t) = t^2 * (3 - 2t)
blended = (1-alpha) * old + alpha * new
```

Publish-side EMA applies to the same mask:

```text
published_t = 0.64 * published_(t-1) + 0.36 * action_t
```

The two gripper dimensions are excluded from interpolation and EMA. Grippers
use independent hysteresis, seven-frame debounce, and seven-frame reversal
lock. Safety limits are applied before publication. Any NaN or Inf is fatal to
the current command and must result in holding the last known safe command.

## 6. Normalization and Training-Asset Boundary

The initial implementation may omit a production R1 Pro LingBot transform,
but it must define its contract now.

The existing R1 Pro statistics contain 20-dimensional `mean`, `std`, `q01`,
and `q99` values for state and action. A later asset adapter may reorganize
keys for LingBot `FeatureTransform`, but must not change numeric values or
dimension order.

The same transformed normalization asset must be used for LingBot training and
real-robot inference. It is tied to the R1 Pro data schema and action semantics,
not to a particular checkpoint.

Until the following are supplied and validated, execution is prohibited:

- trained LingBot R1 Pro checkpoint;
- R1 Pro LingBot robot configuration;
- converted normalization asset;
- 20D schema manifest;
- successful transform round-trip test.

Mock inference, protocol tests, official-parity tests, and dry-run ROS tests may
run without these production assets.

## 7. Configuration

`formal_r1pro_rtc.yaml` stores all formal control settings. Launch scripts may
accept explicit overrides, but they must print the fully resolved configuration
and refuse inconsistent settings.

Examples of rejected configurations:

- `async_prefetch_steps >= requery_steps`;
- `requery_steps > action_horizon`;
- RTC horizon greater than action horizon;
- non-positive control rate;
- EMA alpha outside `[0, 1]`;
- smoothing requested without a valid smoothing mask;
- `--execute` without validated production assets;
- `--execute` against a server whose metadata does not identify LingBot-VLA.

The server publishes metadata before accepting inference:

```python
{
    "model_backend": "lingbot-vla-v2",
    "action_horizon": 50,
    "physical_action_dim": 20,
    "rtc_capable": True,
    "checkpoint_id": str,
    "normalization_id": str | None,
}
```

## 8. Safety and Failure Handling

### 8.1 Default mode

All launchers default to dry-run. Real ROS target publication requires an
explicit `--execute` flag and all production-asset interlocks to pass.

### 8.2 Server unavailable

The client waits during startup. During control, a failed request does not stop
the timer thread. It continues the old chunk and then holds the last safe
action if the horizon is exhausted.

### 8.3 RTC unavailable

The client initially keeps local cross-fade and EMA enabled. If the server does
not return `rtc_enabled=True` for an RTC request, execution remains prohibited
in formal RTC mode. Dry-run mode may continue and logs the mismatch.

### 8.4 Invalid model output

Wrong shape, non-finite values, missing keys, or an unrecognized action schema
prevent promotion of the new chunk. The previous chunk remains active.

### 8.5 Timing overruns

Inference latency is measured in milliseconds and converted to action steps.
The first compile/warmup request is excluded from the steady-state maximum.
The RTC estimate is capped at four steps for the formal configuration.

## 9. Verification Strategy

Verification progresses without commanding hardware.

### Level 1: Static provenance

- confirm every copied official/RLinf file matches its recorded source hash
  before modifications;
- confirm original repositories remain unchanged;
- reject OpenPI checkpoint paths and OpenPI model imports in the LingBot server.

### Level 2: RTC mathematical tests

- prefix schedules match expected linear/exp/ones/zeros values;
- guidance weight remains finite and clipped;
- disabled RTC returns the base velocity exactly;
- hard and soft overlap affect only intended time steps and dimensions;
- VJP correction does not mutate parameters.

### Level 3: Official inference parity

With fixed observations and fixed noise:

- official LingBot inference and copied RTC inference with RTC disabled must
  produce matching outputs within a defined numerical tolerance;
- BF16 and FP32 tolerances are recorded separately;
- output shape is exactly `[1, 50, model_action_dim]` before untransform.

### Level 4: Policy and protocol tests

- bootstrap request without RTC;
- replan request with RTC;
- metadata handshake;
- malformed input rejection;
- full `[50, 20]` response contract once the transform exists;
- WebSocket round trip and server health check.

### Level 5: Deterministic control simulation

Use a fake policy server with controlled delays and failures to verify:

- request starts after 16 executed steps;
- normal promotion occurs at step 20;
- late inference continues the old chunk;
- queue-delay crop uses actual elapsed steps;
- four-step cross-fade is correct;
- EMA uses alpha 0.36;
- horizon exhaustion holds the last safe action;
- only one request is in flight;
- retry backoff is bounded.

### Level 6: ROS dry-run

- subscribe to recorded or synthetic three-camera and 20D state messages;
- run at 15 Hz without target publication;
- inspect timing, chunk transitions, safety reports, and action traces;
- no `--execute` flag is used.

### Level 7: Hardware gate

Hardware execution is allowed only after the trained checkpoint,
normalization/config assets, transform round-trip, dry-run traces, and a
hold-current hardware test all pass. Initial motion tests use reduced safety
step limits and an operator-accessible emergency stop.

## 10. Acceptance Criteria

The module is ready for model integration when:

1. original RLinf and official LingBot sources are unchanged;
2. provenance of all copied files is recorded and verifiable;
3. LingBot weights load strictly through the copied server path;
4. RTC-disabled copied inference matches official inference;
5. RTC-enabled inference produces finite guided 50-step chunks;
6. the async K20/q16 control schedule passes deterministic tests;
7. queue crop, XFade4, EMA0.36, gripper stabilization, and safety behavior match
   the RLinf formal baseline;
8. real execution remains locked until trained R1 Pro assets are validated;
9. no RECAP or VR runtime dependency is present.

## 11. Deferred Work

The following work is intentionally deferred until the trained LingBot R1 Pro
checkpoint is available:

- final R1 Pro semantic feature mapping;
- conversion of the existing 20D normalization statistics into LingBot keys;
- checkpoint-specific strict-load validation;
- transform apply/unapply round-trip validation;
- real-camera preprocessing validation;
- real-robot execution.

These additions must fit behind `LingBotVlaV2RTCPolicy`; they must not change
the copied RLinf client, async schedule, smoothing chain, or ROS publication
contract.
