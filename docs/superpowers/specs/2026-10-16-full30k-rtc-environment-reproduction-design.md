# Full30K RTC Environment Reproduction Design

Date: 2026-10-16

## Objective

Make the Git repository sufficient to recreate the software environments and
run the verified `full30k` BF16 RTC evaluation on `demo_clean` with unseen
instructions, without editing repository code on the target server.

Large models and simulator assets remain outside Git. The repository will
instead contain portable environment definitions, exact snapshots of the
verified server environments, an asset manifest, setup and verification
scripts, and one canonical evaluation command.

## Supported and Verified Background

The authoritative source is the DRLServer installation that has already run
the full30k multi-GPU RTC pipeline successfully. The documentation will record
the observed operating system, CPU architecture, NVIDIA driver, GPU model and
memory, CUDA compatibility reported by `nvidia-smi`, Python versions, PyTorch
CUDA runtimes, important simulator package versions, and Git commit.

This is a record of the verified environment, not a promise of compatibility
with every Linux distribution, CUDA toolkit, GPU architecture, or ARM system.
The supported baseline is Linux x86_64 with NVIDIA GPUs, Conda, BF16 inference,
and a working SAPIEN/Vulkan rendering stack.

## Canonical Evaluation Contract

The reproduction target is:

```text
model: full30k
precision: bf16
mode: RTC
setting: demo_clean
instruction_type: unseen
tasks: official 50-task order
episodes: 50 per task for the documented benchmark run
request_index: 20
fixed_delay_steps: 10
execution_horizon: 20
crossfade_steps: 10
ema_alpha: 0.10
```

The RTC configuration default for `ema_alpha` will be changed to `0.10`, and
all affected tests, dry-run expectations, video path expectations, and
documentation will be updated together. The canonical command will still pass
all RTC parameters explicitly so a recorded command remains unambiguous if a
default changes later.

GPU IDs and the base port are deployment choices, not benchmark semantics.
Users select them with `--gpu-ids` (or `--gpu` for one GPU) and `--base-port`.
No physical GPU number or server-specific project path is stored in the
environment definitions.

## Repository Layout

The following tracked files will be added:

```text
environment/
├── README.md
├── system-background.md
├── lingbotvla.environment.yml
├── robotwin.environment.yml
├── snapshots/
│   ├── lingbotvla.conda-explicit.txt
│   ├── lingbotvla.pip-freeze.txt
│   ├── robotwin.conda-explicit.txt
│   ├── robotwin.pip-freeze.txt
│   └── server-runtime.txt
└── assets-full30k.json

scripts/
├── export_verified_environments.sh
├── create_eval_environments.sh
└── verify_full30k_rtc_environment.sh
```

The existing `docs/SERVER_REPRODUCTION_GUIDE.md` will be extended in place.
Existing user-authored content will be preserved; corrections and the new
full30k path will be additive or narrowly scoped replacements of stale facts.

## Environment Capture

`scripts/export_verified_environments.sh` runs on DRLServer and treats these
environment prefixes as defaults, while allowing explicit overrides:

```text
/home/hddData/User/lixiang/conda-envs/lingbotvla
/home/hddData/User/lixiang/conda-envs/RoboTwin
```

The exporter records:

- `conda list --explicit` for exact same-platform package URLs;
- `python -m pip freeze` for the complete pip state;
- sanitized portable environment YAML files;
- Python, PyTorch, CUDA runtime, SAPIEN, mplib, PyTorch3D, CuRobo, driver,
  GPU, operating-system, architecture, and repository commit information.

The exporter must fail rather than publish a portable file containing:

```text
/home/lixiang/
/home/hddData/
file://
-e <absolute local path>
```

Local editable packages are recorded separately as installation steps. CuRobo
must never be restored from an old `.pth` or editable path. It is installed
from the current checkout after its pinned source commit has been verified.

Secrets, shell history, credentials, environment variables unrelated to the
runtime, caches, and binary Conda package caches are never captured.

## Portable and Exact Layers

The environment system deliberately provides two layers:

1. `*.environment.yml` files are the supported creation interface. They use
   environment names rather than prefixes, omit machine-specific `prefix`
   fields, and pin the versions required by the verified runtime.
2. `snapshots/*` files are forensic references. Explicit Conda URLs are useful
   for exact Linux x86_64 reconstruction, and pip snapshots explain differences
   if a solver or package index changes.

The snapshots are not blindly installed when they contain local paths. The
creation script consumes the portable YAML files and then performs the
repository-local editable installs in a controlled order.

## Environment Creation

`scripts/create_eval_environments.sh` will:

1. locate Conda or use a caller-provided Conda executable;
2. refuse to overwrite existing environments unless the user explicitly asks
   for an update mode;
3. create `lingbotvla` and `RoboTwin` from the tracked YAML definitions;
4. install the LingBotVLA package from this repository where required;
5. verify the pinned CuRobo checkout and install it editable into RoboTwin;
6. print the exact environment interpreter paths for use with the launchers;
7. run the environment verifier.

It will not download model weights or RoboTwin assets and will not change any
evaluation source file. A failed Conda or pip step terminates immediately; the
script must not silently substitute a different PyTorch or CUDA build.

## External Asset Manifest

`environment/assets-full30k.json` describes every non-Git input required for
the canonical evaluation:

- Qwen3-VL-4B-Instruct;
- full30k HF checkpoint shards and index;
- the full5k training configuration, robot configuration, and normalization
  statistics reused by full30k;
- MoGe depth weights;
- LingBot depth weights;
- DINO video teacher weights and configuration;
- RoboTwin objects, embodiments, and background textures;
- the pinned CuRobo repository and commit.

Each entry contains its purpose, target path relative to the project root,
source URL when redistribution permits it (otherwise `provided_by_maintainer`),
expected key files, expected size where useful, and SHA256 for stable files or
indexes. The manifest contains metadata only; no large binary enters Git.

## Verification

`scripts/verify_full30k_rtc_environment.sh` is read-only with respect to model
weights, assets, and evaluation source. It validates:

- required interpreter paths and environment names;
- absence of a known-incompatible CUDA path in `LD_LIBRARY_PATH`;
- Python, PyTorch, CUDA availability, and BF16-capable GPU access;
- imports for SAPIEN, mplib, PyTorch3D, and CuRobo;
- CuRobo resolving to the current repository checkout;
- full30k index validity, all six referenced shards, and absence of incomplete
  shards;
- full5k configuration and normalization inputs reused by full30k;
- Qwen, depth, DINO video, and RoboTwin asset paths;
- `evaluation/prepare_runtime.sh` checks;
- `preflight.sh --model full30k --deep`.

Verification failure prints the exact failed invariant and exits nonzero. It
does not attempt an opportunistic package reinstall.

After static verification, the guide requires a one-episode single-GPU RTC
smoke test before a multi-GPU benchmark. A successful smoke test means the
episode completes and produces the manifest, episode JSONL, logs, and RTC
trace; model task success is not required for environment validation.

## Canonical Multi-GPU Launch

The guide will provide a copy-paste launch template that:

- clears the known `LD_LIBRARY_PATH` conflict;
- exports `ROBOTWIN_PYTHON` and `LINGBOTVLA_PYTHON` using caller-selected
  environment roots;
- checks only the GPUs actually listed by the user;
- verifies each selected GPU has the documented free-memory margin;
- verifies the selected consecutive inference port range is free;
- launches the existing `run_rtc_fast_50x50.sh` without modifying it.

The example may use GPU IDs `1,2,3,5` and base port `9380`, but both are clearly
marked as examples. The authoritative values are the `--gpu-ids` and
`--base-port` arguments supplied for that run.

## Documentation Corrections

The reproduction guide will be updated to:

- list `step2000`, `full5k`, `full15k`, and `full30k` accurately;
- distinguish the verified multi-GPU RTC result from the warning that each new
  server must validate SAPIEN/Vulkan capacity;
- make full30k RTC Clean-unseen the primary target;
- state that full30k reuses full5k configuration and statistics;
- document all external asset sources and checks;
- point to the new environment creation and verification workflow;
- retain full5k/full15k historical instructions without presenting them as the
  canonical target;
- update the document date and remove contradictory stale statements.

## Testing Strategy

Implementation follows test-driven development. Tests will cover:

- exporter sanitization and refusal of absolute editable paths;
- asset-manifest schema and required full30k entries;
- environment creation dry-run commands and overwrite protection;
- verifier behavior for missing model shards, stale CuRobo paths, and invalid
  CUDA library paths;
- RTC default `ema_alpha=0.10` and CLI override behavior;
- canonical command dry-run output for full30k, unseen, selected GPU IDs,
  port range, and the complete RTC parameter set;
- shell syntax and YAML/JSON parsing;
- existing evaluation and RTC entrypoint regression suites.

Tests must not create or delete the real Conda environments, download large
assets, or start a simulator episode. The final manual acceptance step runs on
DRLServer after generated environment artifacts have been exported there.

## Reproduction Success Criteria

The work is complete when a new Linux x86_64 NVIDIA server can:

1. clone the repository at a recorded commit;
2. obtain external assets using the tracked manifest and place them at the
   documented relative paths;
3. create both Conda environments from tracked definitions;
4. install the repository-local packages without editing source code;
5. pass full30k deep preflight and the one-episode RTC smoke test;
6. select arbitrary available GPU IDs and a free port range;
7. launch full30k BF16 RTC `demo_clean` unseen evaluation with
   `20/10/20/10/0.10`;
8. produce provenance, per-episode records, inference logs, evaluation logs,
   RTC traces, and resumable run state.

The repository cannot guarantee identical behavior on unverified operating
systems or drivers, or without access to the required external model and asset
files. These boundaries must remain explicit.
