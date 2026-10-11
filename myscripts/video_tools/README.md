# Evaluation video tools

Slow one or more RoboTwin evaluation videos without overwriting the originals:

```bash
cd /home/lixiang/PROJECT/LingBotVLA2.0
bash myscripts/video_tools/slow_eval_videos.sh --speed 0.25 \
  /path/to/rtc_episode0_success.mp4 \
  /path/to/sync_episode0_success.mp4
```

Directories are accepted and searched recursively for MP4 files. Existing
slow-motion outputs are skipped. `0.5` produces half-speed playback and `0.25`
produces quarter-speed playback. This changes only MP4 presentation timestamps;
it does not reconstruct unrecorded SAPIEN physics substeps or exact wall-clock
motion.
