# lingbot_action_step2000 evaluation

This package evaluates `/models/lingbot_action_step2000` directly and has no
dependency on `baseline_base_model_adapter`. The policy server reads the
checkpoint's own training recipe, robot config and clean normalization stats.

The single RTX 3090 commands use BF16 because the official FP32 reproduction
mode generally exceeds 24 GiB once the simulator is resident. Clean and
randomized must be run sequentially.

```bash
cd /home/lixiang/PROJECT/LingBotVLA2.0/evaluation/lingbot_action_step2000
bash preflight.sh
bash run_formal_clean.sh
bash run_formal_randomized.sh
```

One clean task, one episode, wall-clock-paced physics and a 30 FPS viewer:

```bash
bash run_realtime_visualization.sh lift_pot 1
```

Realtime visualization changes only wall-clock pacing and viewer refresh. It
does not change `step_lim`, task success, action chunks or SAPIEN timestep.
Inference remains synchronous in this entrypoint; RTC is not enabled here.

Headless, fast-simulation evaluation with RoboTwin's native episode video
recording enabled:

```bash
cd /home/lixiang/PROJECT/LingBotVLA2.0
bash evaluation/lingbot_action_step2000/run_video_eval.sh \
  demo_clean --task click_alarmclock --episodes 1
```

Omit `--task` to evaluate all 50 tasks. Videos and logs are isolated under
`evaluation/video_comparison/non_rtc/clean/`. Each run keeps RoboTwin's
per-task video folders under `eval_results/<task>/`. The launcher keeps
`render_freq=0`: it does not open a GUI or pace physics to wall-clock time.
Inference waits do not create duplicate video frames because RoboTwin records
frames only when a policy action is executed. This recording-only client also
refreshes RoboTwin's cached camera observation before every policy action;
without that refresh, the upstream LingBot client repeats one cached image for
most of an action chunk. The synchronous evaluation client itself is untouched.
