# RoboTwin 2.0 — LeRobot v2.1 numeric state/action plots

This is a separate v2.1 adaptation of `check_lerobot_v30_numeric_and_plot.py`.
It does not modify the dataset and does not replace the v3.0 plotting tool.

## Data

Default input:
`/home/lixiang/PROJECT/LingBotVLA2.0/RoboTwin2_data/lerobot_dataset/RoboTwin_lerobot_v21`

Per-episode Parquet files: `data/chunk-000/episode_000000.parquet` etc.
`observation.state` and `action`: 14 dimensions, 15 FPS:

- 0–5: left arm joints
- 6: left gripper
- 7–12: right arm joints
- 13: right gripper

## Run

```bash
cd /home/lixiang/PROJECT/LingBotVLA2.0
conda activate lingbotvla

# Five selected episodes: static PNGs, static gallery, interactive browser HTML.
python myscripts/lerobot_tools/check_lerobot_v21_numeric_and_plot.py \
  --root "$PWD/RoboTwin2_data/lerobot_dataset/RoboTwin_lerobot_v21" \
  --out-dir "$PWD/myscripts/lerobot_tools/results/robotwin_v21_numeric" \
  --episodes 0,1,17,35,277 \
  --plot-mode all \
  --interactive-groups \
  --interactive-default-group-size 1 \
  --interactive-group-sizes 1,5,10,20 \
  --open-html

# All 2500 episodes: interactive only (avoids generating 10000 PNGs).
python myscripts/lerobot_tools/check_lerobot_v21_numeric_and_plot.py \
  --root "$PWD/RoboTwin2_data/lerobot_dataset/RoboTwin_lerobot_v21" \
  --out-dir "$PWD/myscripts/lerobot_tools/results/robotwin_v21_all" \
  --interactive-only \
  --interactive-default-group-size 1 \
  --interactive-group-sizes 1,5,10,20 \
  --open-html

# Range: episode 100 through 109.
python myscripts/lerobot_tools/check_lerobot_v21_numeric_and_plot.py \
  --start 100 --end 110 \
  --interactive-only
```

## Outputs

```
myscripts/lerobot_tools/results/robotwin_v21_numeric/
  plot_manifest.csv
  numeric_alignment_summary.csv
  plots/
    index.html
    interactive_episode_groups.html
    episode_000000/
      episode_000000_state_action_overlay.png
      episode_000000_action_rows.png
      episode_000000_state_rows.png
      episode_000000_jump_rows.png
```

## Interaction

Open `plots/interactive_episode_groups.html` in a modern browser.
Press 1/5/0/2 for group sizes 1/5/10/20; left/right to move groups;
up/down to change joint; s/a/o for state/action/overlay.

Default interactive HTML loads Plotly from the CDN. To embed Plotly JS locally,
add `--offline-plotly` (requires the Python `plotly` package); the resulting
file is larger. Modern browsers must support gzip `DecompressionStream`.

The data audit only checks numerical frame indices, timestamps,
`Action[t]` vs `State[t+1]`, and frame-to-frame numeric jumps. It does not
inspect camera video or certify video-to-state synchronization.

To test a few episodes first is recommended. The real user's full v2.1
dataset is only available on their Ubuntu machine; source code has been
syntax checked and tested on synthetic 14D v2.1-layout episode files.
