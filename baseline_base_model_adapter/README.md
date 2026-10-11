# LingBot-VLA 2.0 foundation-model zero-shot baseline

This module evaluates the untouched `models/lingbot-vla-v2-6b` checkpoint on
the competition's 50 RoboTwin tasks. It is deliberately separate from future
post-trained baselines and never references `lingbot-vla-v2-6b-robotwin`.

The checkpoint directory contains relative symbolic links, not copied or
converted weights. `manifest.json` records each source path and SHA-256 hash.
The robot normalization file was computed from the official 2,500 clean
demonstrations. Normalization is an interface calibration step; no model
parameter was trained or updated.

## Before evaluation

Run the non-simulation preflight:

```bash
cd /home/lixiang/PROJECT/LingBotVLA2.0/baseline_base_model_adapter
bash preflight.sh
```

The strict GPU checkpoint load has already been separated into an explicit
command because it temporarily occupies most GPU memory:

```bash
source /home/lixiang/miniconda3/etc/profile.d/conda.sh
conda activate lingbotvla
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
python tools/smoke_load.py --adapter "$PWD"
```

## Run 50 tasks × 50 episodes

Clean:

```bash
cd /home/lixiang/PROJECT/LingBotVLA2.0/baseline_base_model_adapter
bash run_clean_50x50.sh
```

The default command is headless and records no video. To open a live SAPIEN
viewer for one clean episode without mixing it into the formal run:

```bash
bash run_clean_50x50.sh --visualize --task lift_pot --num-episodes 1
```

Resume the interrupted formal run from its last completed episode:

```bash
bash run_clean_50x50.sh --resume-run \
  "$PWD/results/zero_shot_clean/baseline_base_model_adapter_0k_demo_clean_20260930_143902"
```

Randomized:

```bash
cd /home/lixiang/PROJECT/LingBotVLA2.0/baseline_base_model_adapter
bash run_randomized_50x50.sh
```

Run clean and randomized sequentially on the single RTX 3090. Each command
uses BF16, one resident inference server, one simulator worker, and disables
video recording. Press `Ctrl-C` once to stop a run; the launcher cleans up its
children. Results are written below `results/zero_shot_clean/` and
`results/zero_shot_randomized/`.

BF16 is necessary for the 24 GB GPU and is not directly comparable with the
official post-trained checkpoint's published FP32 numbers. The official
competition submission standard remains 100 episodes per task; 50 is the
requested local zero-shot measurement size.
