# RoboTwin LeRobot v2.1 三路相机回放工具

本工具基于已可播放 AV1 视频的 `replay_lerobot_v30_videos_keyboard.py` **frame-seek** 版本适配，仅新增 `replay_lerobot_v21_videos_keyboard.py`，不覆盖现有 v3.0 工具，也不会修改原始数据集。保留四宫格、浏览器窗口、键盘/终端命令、指定帧暂停、FFmpeg 精确寻帧和缺失尾帧警告。

## v2.1 格式

基于用户本机检查到的布局：

- `meta/info.json`：版本 `v2.1`，`fps=15`，14D，三路摄像头；
- `meta/episodes.jsonl`：Episode 索引与 `length`；
- `data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet`：每 Episode 单独一个 Parquet；
- `videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4`：每 Episode、每相机独立 MP4。

相机名：`observation.images.cam_high`, `observation.images.cam_left_wrist`, `observation.images.cam_right_wrist`。

与 v3.0 不同，v2.1 不读取 `meta/episodes/*.parquet` 或共享 MP4 的时间区间。打开当前 Episode 时，`ffprobe` 会检查该 Episode 的三路 MP4 真正帧数。只有当前 Episode 的有效图像会显示；当某路 MP4 比状态轨迹短时，那一路显示红色 `MISSING TAIL FRAME`。

## 安装

将压缩包放到 `~/下载/` 后执行：

```bash
cd /home/lixiang/PROJECT/LingBotVLA2.0
unzip -o ~/下载/robotwin_v21_video_replay_tools.zip -d .
conda activate lingbotvla
```

**不需要重装 OpenCV、pyarrow 或更改 Conda 环境。** 需要 `ffmpeg`、`ffprobe`，以及 `cv2`、`numpy`。使用 `--backend browser` 可绕过不支持 `cv2.namedWindow` 的 OpenCV 发行版。

## 先检查 Episode 0 视频帧数

```bash
cd /home/lixiang/PROJECT/LingBotVLA2.0
python myscripts/lerobot_tools/replay_lerobot_v21_videos_keyboard.py \
  --root "$PWD/RoboTwin2_data/lerobot_dataset/RoboTwin_lerobot_v21" \
  --episode-index 0 --check-only
```

`--check-only` 会按 Episode 的 `length` 检查每路实际 MP4 帧数，只读取索引并调用 `ffprobe`，不打开浏览器。

如果想验证每路视频的首、中、末有效帧能否被 FFmpeg 正确解码：

```bash
python myscripts/lerobot_tools/replay_lerobot_v21_videos_keyboard.py \
  --root "$PWD/RoboTwin2_data/lerobot_dataset/RoboTwin_lerobot_v21" \
  --episode-index 0 --decode-check
```

## 回放

```bash
python myscripts/lerobot_tools/replay_lerobot_v21_videos_keyboard.py \
  --root "$PWD/RoboTwin2_data/lerobot_dataset/RoboTwin_lerobot_v21" \
  --episode-index 0 --rate 1 --backend browser
```

直接打开 Episode 0 的第 100 号帧（显示后自动暂停）：

```bash
python myscripts/lerobot_tools/replay_lerobot_v21_videos_keyboard.py \
  --root "$PWD/RoboTwin2_data/lerobot_dataset/RoboTwin_lerobot_v21" \
  --episode-index 0 --frame 100 --backend browser
```

也可以使用 `--start-paused`，启动即暂停在第 0 帧。指定帧从 **0** 开始计数，应小于该 Episode 的 `length`。网页内可拖进度条、输入帧号、按上一帧/下一帧按钮；不会自动继续播放，只有按 `p` 或点击「暂停/继续」才恢复。

终端支持：`h` 帮助、`ls` 列表、`p` 暂停/恢复、`n` 下一条、`b` 上一条、`g 35` 跳转 Episode、`i 10` 跳转列表索引、`rate 0.5` 修改播放速度、`f 64` 选帧暂停、`prevframe`/`nextframe` 单步、`r` 重播、`s` 停止、`q` 退出。

## 数值曲线绘图复用现有 v3.0 工具

现有 `check_lerobot_v30_numeric_and_plot.py` 不依赖 v3 视频目录。它按 Parquet 读取 14D State/Action，可用于 v2.1，但还需要本地运行验证：

```bash
python myscripts/lerobot_tools/check_lerobot_v30_numeric_and_plot.py \
  --root "$PWD/RoboTwin2_data/lerobot_dataset/RoboTwin_lerobot_v21" \
  --out-dir "$PWD/myscripts/lerobot_tools/results/robotwin_v21_numeric" \
  --episodes 0,1,2 --plot-mode all --interactive-groups --open-html
```

## 数据完整性与局限

- v2.1 与 v3.0 有相同的 Episode 数和总 State/Action 帧数，**不能由此推断 v2.1 视频也缺相同帧**。请使用 `--check-only` 检查实际视频。
- 本播放器不会静默读取下一 Episode 画面，不对 MP4 补帧或修改时间戳。
- 每次只对所选 Episode 的 3 个 MP4 执行 `ffprobe`，不会在启动时遍历 7500 个视频。
- 如果 `ffprobe` 读不到可靠帧数，程序会显式报错；不会把猜测帧数当成真实结果。
- `--fps` 仅覆盖播放速率的基准，不修改视频时间定位所用的原始 `info.json` FPS。
- 回放的画面帧数一致不等于已经证明图像与关节状态的物理时序同步。

## 测试

```bash
python -m unittest discover -s myscripts/lerobot_tools/tests -p 'test_replay_lerobot_v21.py' -v
python -m unittest discover -s myscripts/lerobot_tools/tests -p 'test_frame_seek_v21.py' -v
python -m unittest discover -s myscripts/lerobot_tools/tests -p 'test_ffmpeg_av1_v21.py' -v
```

这些测试会在临时目录中创建微型模拟视频，不会访问或修改 RoboTwin 官方数据集。
