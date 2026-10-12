# Resilient Multi-GPU Evaluation Design

## 1. Goal

Make RTC, non-RTC, and video evaluation reliable when several inference
servers start concurrently or one inference slot fails during a run.

The change must preserve the existing model, policy, episode, seed, resume,
success-check, RTC, logging, and statistics semantics. It only changes when
simulation workers may start and how infrastructure failures are scheduled.

Success means:

- no simulation client starts before its inference server is ready;
- an unhealthy inference slot is quarantined instead of receiving more work;
- the interrupted task is reassigned to a healthy slot without consuming a
  task-level retry;
- healthy slots continue until every recoverable task is complete;
- if no healthy slot remains, the run exits nonzero and reports the unfinished
  tasks instead of silently declaring completion;
- existing commands remain valid without adding new arguments.

## 2. Current Behavior and Failure Modes

The launchers start all policy-server processes and immediately launch
RoboTwin clients. A live policy-server PID does not mean that model loading,
optional compilation, and WebSocket binding have finished. With several large
models loading concurrently, clients can connect during this gap and fail with
`ConnectionClosedError`.

The existing scheduler also treats every nonzero simulation exit as a task
failure. It retries the task on the same slot up to three times. If the slot's
inference process has died or become unavailable, this wastes every retry and
then continues assigning other tasks to the same unusable slot.

The policy server already exposes `GET /healthz`, returning HTTP 200 only after
model construction completes and the WebSocket server is serving. This is a
stronger readiness signal than process existence, CPU utilization, or a raw
TCP `LISTEN` check.

## 3. Scope

The behavior applies to both queue-scheduled launchers:

- `myscripts/simulation_rtc_tools/scripts/start_robotwin_rtc_infer_and_eval.sh`
- `evaluation/lingbot_action_step2000/video_eval_launcher/start_robotwin_video_infer_and_eval.sh`

Their higher-level RTC, non-RTC, and video wrappers will expose the same
optional timeout argument. No official RoboTwin task, success checker, or
environment implementation is modified.

The implementation may introduce one small shared shell helper for inference
health checks so the two launchers cannot drift. Scheduling state remains in
the launchers unless extracting it materially reduces duplication without
changing behavior.

## 4. Public Interface

Add one optional argument:

```text
--inference_ready_timeout SECONDS
```

The default is 180 seconds, matching the requested behavior. The value must be
a positive integer. Both hyphenated and underscored wrapper spellings may be
accepted at the user-facing entrypoints, following existing CLI conventions.

Existing commands require no changes. GPU selection through `--gpu` and
`--gpu-ids` remains unchanged.

The readiness poll interval is an internal constant of two seconds. Pending
status is printed immediately and at least once every ten seconds, without
flooding the terminal.

## 5. Startup Readiness Barrier

After all policy-server PIDs and logs have been recorded, but before copying or
starting any RoboTwin evaluation client, the launcher performs a barrier over
all active slots.

For each slot it checks, in this order:

1. the recorded policy-server PID is alive;
2. `http://127.0.0.1:<port>/healthz` returns HTTP 200 and body `OK`;
3. the check completes within a short per-request timeout so one port cannot
   block checks for the other ports.

The checks are logically parallel: every polling pass visits all pending
slots, so a slow low-numbered port cannot prevent later ports from being
reported ready.

While waiting, output identifies slot, physical GPU, port, PID, and elapsed
time. Once all slots pass, output is deterministic:

```text
All inference servers ready:
  slot 0 GPU 2 port 9330 pid 1234
  slot 1 GPU 3 port 9331 pid 1235
```

Only then may Phase 2 start.

If a server process exits before becoming healthy, the launcher fails
immediately. If the global timeout expires, it reports every pending slot,
prints the final section of each affected inference log, cleans up all child
processes through the existing trap, and exits nonzero. Simulation is never
started after a partial startup.

The readiness implementation uses Python's standard library HTTP client via an
already-resolved project/runtime Python, avoiding new dependencies on `curl`,
`wget`, or `nc`.

## 6. Runtime Slot Health and Failure Classification

Each slot has an explicit state:

```text
starting -> ready -> busy/free -> released
                         \
                          -> quarantined
```

When a simulation worker exits nonzero, the scheduler rechecks the associated
inference slot:

- **healthy PID and healthy `/healthz`:** classify as a task failure and retain
  the current three-attempt task retry behavior;
- **dead PID or failed `/healthz`:** classify as an infrastructure failure,
  quarantine the slot, and requeue the task;
- **ambiguous transient health result:** retry the health probe for a small,
  bounded grace period before classifying the slot, preventing one missed
  request from quarantining a working server.

A slot may also fail while its client has not yet exited. The scheduler checks
active inference PIDs during every scheduling pass. A dead inference PID causes
the corresponding simulation process group to be terminated, waited for, and
handled through the same infrastructure-failure path. This prevents a client
from hanging forever on a server that cannot reply.

Quarantine is permanent for the current run. The design deliberately does not
restart a model on the same GPU: hardware faults, out-of-memory conditions, and
driver failures would otherwise cause long repeated model loads and unstable
task ownership.

An adaptively released idle slot is not a failure and is never quarantined.

## 7. Requeue and Retry Semantics

The task source becomes an appendable pending queue rather than a fixed array
used only through its original length. Requeueing appends the interrupted task
to the tail, allowing other healthy tasks to continue first.

Two counters remain separate:

- `task_retries[task]`: increments only when the inference slot is healthy and
  the task process fails; maximum remains three attempts;
- `infra_requeues[task]`: increments when the task loses its inference slot and
  is moved to another slot.

Infrastructure requeues do not consume task retries. To prevent an inherently
crashing request from moving forever across GPUs, a task may be requeued for an
infrastructure failure at most the number of initially active slots. Exceeding
that limit marks it as an infrastructure skip with a distinct status and keeps
the overall launcher exit nonzero.

After quarantine:

- the slot receives no new tasks;
- its server and simulation process group are cleaned up;
- `active_inference` and healthy-slot counts are updated exactly once;
- healthy free slots immediately pull from the pending queue;
- if work remains but no healthy slot exists, scheduling stops with a clear
  unfinished-task report and nonzero exit status.

Resume overrides (`initial_episode_count`, `initial_success_count`, and
`start_seed`) remain attached to the resumed task, not to its original slot.
Reassignment therefore continues the intended episode/seed position.

## 8. Logging and Provenance

Existing per-task and per-inference logs remain in their current locations.
Retry attempt banners continue to append to task logs rather than overwrite
them.

Add scheduler events to a run-level append-only file:

```text
<run_dir>/scheduler_events.jsonl
```

Each event contains wall time and the relevant task, slot, physical GPU, port,
PID, exit code, and reason. Event types include:

- `inference_ready`
- `inference_startup_failed`
- `task_failed`
- `slot_quarantined`
- `task_requeued`
- `task_skipped`
- `run_blocked_no_healthy_slots`

The run manifest records the configured readiness timeout. The final stats add
healthy, quarantined, and infrastructure-skipped counts without altering the
existing success-rate calculation.

## 9. Exit Status

The launcher exits zero only when all selected tasks finish without skipped or
infrastructure-blocked tasks. It exits nonzero when:

- any policy server fails the startup barrier;
- every inference slot becomes unavailable while work remains;
- a task exhausts normal retries;
- a task exhausts infrastructure requeues.

This makes automation able to distinguish a complete evaluation from a run
that merely produced a partial `stats.txt`.

`--keep_inference` applies only to healthy servers at normal completion. Failed
or quarantined processes are always cleaned up.

## 10. Testing Strategy

Tests must not require a GPU or load the model. The launcher logic is exercised
with fake policy-server and simulation processes.

Required cases:

1. one server becomes healthy and simulation starts afterward;
2. several servers become healthy in different orders and the barrier waits
   for all of them;
3. one server exits during startup and the launcher fails with its log tail;
4. one server never becomes healthy and the timeout prevents simulation start;
5. a healthy slot sees a task failure and retries that task on the same slot;
6. an inference server dies during a task, its slot is quarantined, and the
   task completes on another slot without incrementing `task_retries`;
7. one slot fails while other slots finish every remaining task;
8. all slots fail and the launcher exits nonzero with pending tasks reported;
9. infrastructure requeue limit prevents infinite cross-slot bouncing;
10. resume counters and seed follow a requeued resume task;
11. adaptive idle shutdown is not misclassified as quarantine;
12. RTC, non-RTC, and video wrappers forward the timeout consistently;
13. existing GPU-ID mapping, provenance, and entrypoint tests continue to pass;
14. shell syntax checks pass for every changed shell file.

## 11. Non-Goals

This change does not:

- change action chunks, RTC timing, crossfade, EMA, or inference precision;
- automatically restart an inference server on a quarantined GPU;
- judge whether a RoboTwin task result is semantically correct;
- change official task configs, environment steps, or `check_success()`;
- make multiple servers share one model process;
- conceal incomplete runs by reporting them as successful.

