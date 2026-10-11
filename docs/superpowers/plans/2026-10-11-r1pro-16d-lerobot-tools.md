# R1Pro 16D LeRobot Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the complete R1Pro ROS bag to LeRobot v2.1 data-generation, validation, plotting, and replay toolchain into an independent left-first 16D implementation.

**Architecture:** Copy the proven OpenPI 20D command workflows into `myscripts/16d`, but centralize every dimensional, naming, slicing, and required-topic decision in one dependency-light schema module. Adapt each copied command to consume that schema and strictly reject non-16D arrays instead of silently truncating old data.

**Tech Stack:** Python 3, ROS 2 bag/MCAP CLI, NumPy, pandas/pyarrow, OpenCV/Pillow, matplotlib/Plotly/Dash, pytest.

**Spec:** `docs/superpowers/specs/2026-10-11-r1pro-16d-lerobot-tools-design.md`

## Global Constraints

- State/action order is exactly `[left_arm_j0..j6, left_gripper, right_arm_j0..j6, right_gripper]`.
- State and action vectors are exactly 16D; never truncate, reorder implicitly, or zero-fill.
- Do not record, require, export, plot, or replay four-dimensional torso data.
- Preserve ROS bag keyboard start/stop/keep/discard behavior.
- Do not modify `/home/lixiang/PROJECT/openpi/myscripts/20d`, LingBot official code, simulation evaluation, or RTC tools.
- Existing bags may contain extra torso topics, which are ignored; missing required arm/gripper topics are errors.
- Output metadata records schema `r1pro_dual_arm_16d_left_first_v1`.

## Review Focus

- A 20D old intermediate array must fail clearly rather than being accepted as 16D.
- A bag containing optional torso topics must remain readable without incorporating torso values.
- Left/right synthetic sentinel values must prove no swap occurred through vector assembly.
- Feature names in LeRobot metadata must match the actual parquet vector indices.
- Tools must remain importable for tests on machines without active ROS nodes or GUI displays.

---

### Task 1: Canonical 16D schema and test foundation

**Files:**
- Create: `myscripts/16d/common/__init__.py`
- Create: `myscripts/16d/common/schema.py`
- Create: `myscripts/16d/tests/test_schema.py`
- Create: `myscripts/16d/README.md`

**Interfaces:**
- Produces: `SCHEMA_NAME`, `STATE_NAMES`, `ACTION_NAMES`, `GROUPS`, `TOPICS`, `CORE_TOPIC_KEYS`, `assemble_state()`, `assemble_action()`, and `validate_vector_array()`.

- [ ] Write tests asserting exact names, 16D length, disjoint full-range group indices, torso-free core topics, sentinel-based left/right assembly, and rejection of 20D arrays.
- [ ] Run `pytest -q myscripts/16d/tests/test_schema.py` and verify failure because the module does not exist.
- [ ] Implement the minimal schema module and package initializers.
- [ ] Run the schema tests and verify they pass.
- [ ] Document the directory, schema, dependencies, and intended command pipeline in `README.md`.

### Task 2: Keyboard-driven ROS bag recorder

**Files:**
- Create: `myscripts/16d/record_tools/record_r1pro_mcap_episode.py`
- Create: `myscripts/16d/tests/test_record_topics.py`

**Interfaces:**
- Consumes: `TOPICS` and `CORE_TOPIC_KEYS` from Task 1.
- Produces: the same operator-facing episode lifecycle as the 20D recorder, with torso-free ROS bag topic selection and metadata.

- [ ] Write tests importing the recorder and asserting all core topics are arm/gripper/camera topics, torso CLI options are absent, and metadata declares the 16D schema.
- [ ] Run the recorder tests and verify the copied/unadapted implementation fails them.
- [ ] Copy the source recorder, change paths/imports, remove torso modes and metadata, and drive topic selection from the schema.
- [ ] Preserve start, stop, keep, discard, episode numbering, cleanup, and signal-handling behavior.
- [ ] Run recorder tests plus `python .../record_r1pro_mcap_episode.py --help`.

### Task 3: MCAP to left-first 16D intermediate export

**Files:**
- Create: `myscripts/16d/convert_tools/export_r1pro_mcap_to_intermediate.py`
- Create: `myscripts/16d/tests/test_intermediate_export.py`

**Interfaces:**
- Consumes: schema assembly/validation and torso-free topic keys.
- Produces: per-episode `state.npy` and `action.npy` shaped `[N,16]`, images, timestamps, and quality metadata.

- [ ] Write tests for synthetic state/action assembly, missing required series, ignored torso series, output names/order, and 20D rejection.
- [ ] Run tests and verify expected failures.
- [ ] Copy and adapt the exporter: left-first assembly, no torso extraction/timebase repair, new defaults, and schema metadata.
- [ ] Retain image decoding, nearest timestamp matching, maximum-time-difference checks, and episode reporting.
- [ ] Run exporter tests and its `--help` smoke test.

### Task 4: LeRobot v2.1 dataset builder

**Files:**
- Create: `myscripts/16d/convert_tools/build_lerobot_from_r1pro_intermediate.py`
- Create: `myscripts/16d/tests/test_lerobot_builder.py`

**Interfaces:**
- Consumes: validated `[N,16]` intermediate episodes.
- Produces: LeRobot v2.1 metadata, parquet, and video layout with exact 16D feature names.

- [ ] Write tests for metadata shapes/names, finite-value and frame-count validation, successful 16D fixtures, and explicit failure on 20D fixtures.
- [ ] Run tests and verify expected failures.
- [ ] Copy and adapt the builder to schema constants and project-local defaults.
- [ ] Run builder tests and its `--help` smoke test.

### Task 5: Numeric audit and plotting tools

**Files:**
- Create: `myscripts/16d/lerobot_tools/check_lerobot_v21_missing_audit.py`
- Create: `myscripts/16d/lerobot_tools/check_lerobot_v21_numeric_and_plot.py`
- Create: `myscripts/16d/tests/test_audit_plot_config.py`

**Interfaces:**
- Consumes: 16D LeRobot v2.1 dataset and intermediate quality metadata.
- Produces: torso-free missing-data audit and four-group state/action plots.

- [ ] Write tests for required topic lists, group index coverage, metadata name validation, and rejection of non-16D parquet vectors.
- [ ] Run tests and verify expected failures.
- [ ] Copy/adapt both tools, remove torso report columns/plots, and use the schema for labels and indices.
- [ ] Run tests and both `--help` smoke tests.

### Task 6: Keyboard and Web LeRobot replay

**Files:**
- Create: `myscripts/16d/lerobot_tools/replay_lerobot_v21_parquet_images_keyboard.py`
- Create: `myscripts/16d/lerobot_tools/replay_lerobot_v21_web.py`
- Create: `myscripts/16d/tests/test_replay_config.py`

**Interfaces:**
- Consumes: strict 16D LeRobot v2.1 datasets.
- Produces: three-camera episode/frame replay and left-first state/action visualization.

- [ ] Write tests for strict dimension validation, feature-name checking, four-group layout, and absence of torso components/callbacks.
- [ ] Run tests and verify expected failures.
- [ ] Copy/adapt keyboard replay and Web replay, retaining navigation behavior but replacing 20D layouts with schema-driven 16D layouts.
- [ ] Run tests, `--help`, and import smoke tests without launching a GUI server.

### Task 7: Full verification and user commands

**Files:**
- Modify: `myscripts/16d/README.md`

**Interfaces:**
- Consumes: all prior commands.
- Produces: reproducible commands for recording, conversion, construction, audit, plotting, and replay.

- [ ] Run `pytest -q myscripts/16d/tests`.
- [ ] Run `python -m compileall -q myscripts/16d`.
- [ ] Run every command's `--help` path and record unavailable optional dependencies separately from code failures.
- [ ] Scan `myscripts/16d` for stale right-first order, expected dimension 20, training torso fields, and old OpenPI output defaults; fix all functional remnants.
- [ ] If a suitable real episode is locally available, run the non-destructive MCAP-to-intermediate smoke test; otherwise document the exact device-side test command.
- [ ] Complete README examples and report exact verification evidence and any hardware-only checks left to the user.
