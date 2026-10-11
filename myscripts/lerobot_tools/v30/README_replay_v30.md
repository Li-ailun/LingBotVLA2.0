# RoboTwin LeRobot v3.0 三路视频键盘回放

**基于用户提供的 `replay_lerobot_v21_parquet_images_keyboard.py` 功能改写。**

原脚本从 LeRobot v2.1 的 Parquet 图像字段解码；现在的 RoboTwin v3.0 使用 `meta/episodes/*.parquet` 记录 Episode、相机独立 MP4 文件及起止时间，图像保存在 `videos/observation.images.*/chunk-XXX/file-YYY.mp4`。脚本按每路相机的元信息定位视频帧；**不会读取超过 Episode 视频结束位置的帧**。

## 安装

将本压缩包解压在 `/home/lixiang/PROJECT/LingBotVLA2.0` 下，形成：

```
myscripts/lerobot_tools/replay_lerobot_v30_videos_keyboard.py
myscripts/lerobot_tools/README_replay_v30.md
myscripts/lerobot_tools/tests/test_replay_lerobot_v30.py
```

环境要求：已有的 `numpy`、`pandas`、`pyarrow` 和 OpenCV（**headless 版可用**，只用于画布合成与 JPEG 编码），以及系统命令 `ffmpeg`、`ffprobe`。**视频帧由 FFmpeg 子进程软件解码**，不再调用 OpenCV 的 AV1 寻帧。浏览器后端也不调用 OpenCV HighGUI，**无需重装 `lingbotvla`、OpenCV 或 GTK**。

## 运行

```bash
cd /home/lixiang/PROJECT/LingBotVLA2.0
conda activate lingbotvla

# 只读预检：Episode 0 的视频区间（不开窗口）
python myscripts/lerobot_tools/replay_lerobot_v30_videos_keyboard.py \
  --root "$PWD/RoboTwin2_data/lerobot_dataset/RoboTwin_lerobot_v30" \
  --episode-index 0 --check-only


# 推荐先实际解码 Episode 0 的三路首帧/中间帧/最后一帧（不打开窗口）
python myscripts/lerobot_tools/replay_lerobot_v30_videos_keyboard.py \
  --root "$PWD/RoboTwin2_data/lerobot_dataset/RoboTwin_lerobot_v30" \
  --episode-index 0 --decode-check

# 推荐：三路相机在浏览器窗口按数据集 15 FPS、1x 播放（无需 OpenCV GUI）
python myscripts/lerobot_tools/replay_lerobot_v30_videos_keyboard.py \
  --root "$PWD/RoboTwin2_data/lerobot_dataset/RoboTwin_lerobot_v30" \
  --episode-index 0 --rate 1 --backend browser

# 检查此前发现异常的 Episode 17，并减速播放
python myscripts/lerobot_tools/replay_lerobot_v30_videos_keyboard.py \
  --root "$PWD/RoboTwin2_data/lerobot_dataset/RoboTwin_lerobot_v30" \
  --episode-index 17 --rate 0.5
```

**命令**：在浏览器画面按 `h/p/n/b/r/s/q` 或点击界面按钮；在启动脚本的终端可键入 `h`、`ls`、`ls 20`、`p`、`n`、`b`、`g 35`、`i 10`、`rate 0.5`、`rate 1`、`rate 2`、`r`、`s`、`q`。默认从选定 Episode 开始播放，播完自动进入下一 Episode；`--loop` 可循环全部 Episode。

**画面布局**：左上 `cam_high`，右上 Episode 信息，左下 `cam_left_wrist`，右下 `cam_right_wrist`。维持旧脚本的 1280×840 版面，`--window-scale 0.8` 可缩小。

**缺帧标注**：假设 Episode 0 的 State 长度 139，而 `cam_high` 的视频区间只有 138 帧，在显示第 139 帧时，该相机显示红色 `MISSING TAIL FRAME`，其余相机照常播放。它**不会静默显示 Episode 1 的第一帧**。这是只读诊断显示策略，并非修复数据集。

视频帧的索引由 `meta/episodes` 的 `from_timestamp`、`to_timestamp` 与 `meta/info.json` 的 FPS 换算；`--fps` 仅修改播放节奏，不改视频索引时间基准。MP4 图像由 FFmpeg 命令行程序解码：在当前 Episode 的视频起始时间执行一次准确寻帧，随后连续解码到该 Episode 的视频末尾，**不再依赖 OpenCV 的 `CAP_PROP_POS_FRAMES`**。对 AV1 视频采用软件解码，并禁止硬件加速。大型视频的首次 seek 可能需要数秒，三个相机会分别启动解码进程。若发生解码错误，会显示 `FFMPEG DECODE ERROR`，不会将相邻 Episode 图像当作有效尾帧。

**限制**：长度差异和跨 Episode 越界由播放器主动阻断；无法仅凭帧数证明每个 Episode 内部图像和关节姿态严格同步。本工具也不修改 LeRobot 数据、Norm Stats 或模型配置。

## 快速自检

```bash
cd /home/lixiang/PROJECT/LingBotVLA2.0
python -m unittest discover -s myscripts/lerobot_tools/tests -v
```

## 修复 OpenCV HighGUI (`cv2.namedWindow`) 报错

如果出现 `OpenCV ... The function is not implemented ... cvNamedWindow`，说明 Python 的 OpenCV 没有图形窗口支持，**不是 LeRobot MP4 损坏**。新版支持：

- `--backend browser`：强制使用本地浏览器，**推荐**。内部采用 `127.0.0.1` HTTP MJPEG 视频流，浏览器键盘/按钮命令通过 localhost 发给原播放器；仍可从终端输入命令。
- `--backend auto`（默认）：优先使用具备 HighGUI 的 OpenCV；若发现 `GUI: NONE`、无 DISPLAY，或捕获到 `cv2.namedWindow()` 错误，则切换到浏览器。
- `--backend opencv`：保留原 OpenCV 原生窗口行为，只有安装带 GUI 后端的 OpenCV 时使用。
- `--no-browser`：只打印 `http://127.0.0.1:<port>/` 地址，不尝试自动打开浏览器。
- `--browser-port 8765`：可指定浏览器服务器端口，默认随机空闲端口。

新后端使用 Python 标准库 `http.server`、FFmpeg 解码和 OpenCV JPEG 编码，不新增 pip 依赖。不会打开对外网络端口，也不会修改数据集。关闭程序在终端按 `q` 回车，或浏览器按 `q`；仅关闭浏览器标签不会终止终端中的播放器。

例如：

```bash
cd /home/lixiang/PROJECT/LingBotVLA2.0
conda activate lingbotvla
python myscripts/lerobot_tools/replay_lerobot_v30_videos_keyboard.py \
  --root "$PWD/RoboTwin2_data/lerobot_dataset/RoboTwin_lerobot_v30" \
  --episode-index 0 --rate 1 --backend browser
```

若自动打开失败，手动把终端打印的 `http://127.0.0.1:<port>/` 复制到 Ubuntu 本机浏览器地址栏。首次测试可使用 `--check-only` 确认元信息读取没问题。

浏览器窗口提供键盘、播放/暂停、上一条/下一条 Episode、跳转 Episode、修改倍速、重播、停止、退出。原始四宫格像素布局仍为 1280×840（网页自适应缩放），缺帧显示红色 `MISSING TAIL FRAME`。


## 修复 AV1 视频播放全黑和 SEEK MISMATCH

旧版本有时在播放时显示 `SEEK MISMATCH (-9007199254740991!=63)`，终端出现 `Missing Sequence Header`、`Failed to get pixel format`、`AV1 hardware accelerated decoding` 等告警。这是 OpenCV/AV1 的帧位置查询或解码兼容性问题：负数不是有效的视频帧编号，也不是数据集缺失了第 63 帧。

**本版本完全不使用 OpenCV 的 `VideoCapture` 解码**，交给系统 FFmpeg 软件解码。安装 FFmpeg 4.4+ 通常即可（可运行 `ffmpeg -version` 和 `ffprobe -version` 检查），并保持原来的浏览器界面。视频本身不会被重新编码或修改。

正式打开网页前建议执行上述 `--decode-check` 命令。成功时显示三路相机的 `[OK] ... shape=(H,W,3)` 和 `[DECODE CHECK] PASSED`。已确认的视频尾部缺帧会显示 `[OK] ... missing tail frame guarded`，属于预期行为。若实际 FFmpeg 无法解码 AV1，执行：

```bash
ffmpeg -hide_banner -decoders | grep -E 'av1|dav1d'
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,avg_frame_rate,width,height \
  -of default=noprint_wrappers=1 \
  '/home/lixiang/PROJECT/LingBotVLA2.0/RoboTwin2_data/lerobot_dataset/RoboTwin_lerobot_v30/videos/observation.images.cam_high/chunk-000/file-000.mp4'
```

**验证范围**：已使用人工制作的 AV1 MP4 在多个不同帧索引（包括非关键帧）上核对实际解码数值，证明不再出现 OpenCV 的非法 seek 值；尚不能在当前环境直接读取你 Ubuntu 机器上的真实 MP4。请以本地 `--decode-check` 结果为准。

## 新增：在当前 Episode 内手动选帧（自动暂停）

**帧索引为 0-based。** 若某 Episode 有 139 个 state/action 帧，可查看 `frame_index=0..138`。浏览器播放页面四宫格下新增「选帧查看」区域：

- 拖动滑块到目标位置，**松开鼠标后**定位对应帧并自动暂停；拖动时不持续解码，避免大量 AV1 随机寻帧。
- 输入整数帧索引并点击「查看该帧并暂停」（或按 Enter）。
- 点击「上一帧」或「下一帧」，逐帧浏览、始终保持暂停；也支持网页的 `[` / `]`、方向键（需避免输入框获得焦点）。
- 显示当前 `Episode`、`frame_index` 和状态；只有主动点击「暂停/继续 p」才能恢复播放。
- 若目标位置恰是已知缺失尾帧，相机显示红色 `MISSING TAIL FRAME`，**不取下一 Episode 图像**。
- 终端可以输入 `f 64` / `frame 64` / `seek 64`、`prevframe`、`nextframe`。超范围帧索引将被拒绝。

可从指定帧直接启动，**无需自动播放**：

```bash
cd /home/lixiang/PROJECT/LingBotVLA2.0
conda activate lingbotvla

python myscripts/lerobot_tools/replay_lerobot_v30_videos_keyboard.py \
  --root "$PWD/RoboTwin2_data/lerobot_dataset/RoboTwin_lerobot_v30" \
  --episode-index 0 \
  --frame 100 \
  --backend browser
```

若想从该 Episode 第 0 帧启动并暂停：将 `--frame 100` 改为 `--start-paused`。

**注意**：从页面选择帧会暂停但不会改变 MP4、Parquet 或 meta/episodes；它读取现有的真实图像，不是生成/插值图像。FFmpeg 对 AV1 视频在某些较远帧位置精确定位可能需要数秒，期间页面仍显示上一张画面和“正在定位”提示。
