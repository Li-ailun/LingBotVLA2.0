# RoboTwin LeRobot v3 — numeric plot tool

This script reproduces the user's older LeRobot v2.1 numeric plot workflow for the
**local RoboTwin_lerobot_v30**, without changing the dataset or loading videos.

Uses original raw `observation.state` and `action`, each 14D:

```
0..5 left arm joints, 6 left gripper,
7..12 right arm joints, 13 right gripper.
```

The actual feature names come from `meta/info.json` (`names` is nested in LeRobot v3).
X-axis uses each episode's `timestamp` at 15 FPS.

## Dependencies

Use your existing `lingbotvla` environment; needs `numpy`, `pandas`,
`pyarrow`, `matplotlib`; browsers use Plotly CDN for interactive graphs.
If your environment already has Python `plotly`, add `--offline-plotly` to embed
it and view fully offline. A modern browser needs `DecompressionStream` to
unpack gzip-compressed episodes on demand.

## Examples

From the project root:

```bash
cd /home/lixiang/PROJECT/LingBotVLA2.0
conda activate lingbotvla

# First test — five representative episodes, all four static plots + HTML:
python myscripts/lerobot_tools/check_lerobot_v30_numeric_and_plot.py \
  --root "$PWD/RoboTwin2_data/lerobot_dataset/RoboTwin_lerobot_v30" \
  --out-dir "$PWD/myscripts/lerobot_tools/results/robotwin_v30_numeric" \
  --episodes 0,1,17,35,277 \
  --plot-mode all --interactive-groups \
  --interactive-default-group-size 1 \
  --interactive-group-sizes 1,5,10,20 \
  --open-html

# All 2500 episodes: faster HTML-only mode (no 10,000 static images):
python myscripts/lerobot_tools/check_lerobot_v30_numeric_and_plot.py \
  --root "$PWD/RoboTwin2_data/lerobot_dataset/RoboTwin_lerobot_v30" \
  --out-dir "$PWD/myscripts/lerobot_tools/results/robotwin_v30_all" \
  --interactive-only \
  --interactive-default-group-size 1 \
  --interactive-group-sizes 1,5,10,20 \
  --open-html

# Only episodes 100 .. 109 (exclusive end):
python myscripts/lerobot_tools/check_lerobot_v30_numeric_and_plot.py \
  --start 100 --end 110 --plot-mode range --interactive-groups
```

Output layout:

```
myscripts/lerobot_tools/results/robotwin_v30_numeric/
  numeric_alignment_summary.csv
  plot_manifest.csv
  plots/
    index.html
    interactive_episode_groups.html
    episode_000000/
      episode_000000_state_action_overlay.png
      episode_000000_action_rows.png
      episode_000000_state_rows.png
      episode_000000_jump_rows.png
```

The tool also reports per-episode frame discontinuities, timestamp-step
anomalies, and the relationship `action[t] == state[t+1]` in the numeric CSV.
This is **not** a camera-to-state synchronization audit, and does not alter the
known per-camera video-length differences.

Interactive shortcuts: `1/5/0/2` group size, left/right to navigate groups,
up/down to change dimensions, `s/a/o` state/action/overlay.
