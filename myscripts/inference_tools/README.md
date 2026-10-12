# LingBot-VLA R1Pro RTC inference (isolated)

This directory is independent from the official RoboTwin evaluation path. It
contains exact source snapshots from RLinf and the official LingBot deploy
folder plus local RTC extensions. Nothing here is imported by
`baseline_base_model_adapter` or `lingbot-vla-v2/experiment/robotwin`.

Implemented and tested:

- LingBot flow-matching RTC guidance inserted after `predict_velocity` and
  before each Euler update; RTC-off follows the official loop exactly.
- 50-step output, 15 Hz control, async q16/K20, delay 3 (cap 4), RTC horizon
  10, exponential prefix schedule and guidance limit 10.
- SmoothStep XFade4, EMA 0.36 excluding grippers 7/15, debounce 7 and reversal
  lock 7 with thresholds 20/60.
- Strict server identity/shape handshake and fail-closed production assets.

Not yet authorized for physical execution: the trained LingBot checkpoint's
R1Pro robot config and its matching normalization statistics do not exist yet.
The server intentionally exits before serving until the checkpoint-specific
physical/model transform is sealed. Never reuse RoboTwin norm stats for R1Pro.

Run tests:

```bash
cd /home/lixiang/PROJECT/LingBotVLA2.0
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python3 -m pytest -q \
  myscripts/inference_tools/tests
```

After training and transform sealing, start the server by setting the four
variables named in `run_lingbot_rtc_policy_server.sh`. Start the ROS client in
dry-run mode with:

```bash
cd /home/lixiang/PROJECT/LingBotVLA2.0/myscripts/inference_tools
bash run_r1pro_lingbot_rtc_client.sh
```

Do not add `--execute` until server metadata, dry-run actions, joint ordering,
limits and emergency stop have been verified on the final checkpoint.
