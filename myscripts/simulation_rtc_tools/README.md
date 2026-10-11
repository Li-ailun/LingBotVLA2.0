# LingBot-VLA 快速仿真 RTC 工具

这是 `lingbot_action_step2000` 的独立 RTC 仿真评测入口。它不会覆盖官方同步客户端，也不会改动 RoboTwin 环境、成功判定、`step_lim`、TOPP 或物理设置。

固定策略为：模型输出 50 步，第 10 步用新观测异步请求，第 20 步固定交接，RTC execution horizon 10，RLinf 风格 XFade4，EMA0.36。新结果先按 10 步交接延迟裁剪，再用 `old[20:24]` 与对齐后的 `new_raw[10:14]` 混合。结果未就绪时暂停仿真，因此等待不推进 SAPIEN、不调用 `take_action()`、不消耗动作预算。Crossfade 和 EMA 均排除夹爪维度。

## 预检

```bash
cd /home/lixiang/PROJECT/LingBotVLA2.0
bash myscripts/simulation_rtc_tools/scripts/preflight.sh
```

预检只检查配置、资产、Python 语法、隔离哈希和 dry-run，不启动 episode。

## 单任务 smoke test

先查看命令：

```bash
bash myscripts/simulation_rtc_tools/scripts/run_rtc_fast_one_task.sh lift_pot --dry-run
```

正式执行一个 episode：

```bash
bash myscripts/simulation_rtc_tools/scripts/run_rtc_fast_one_task.sh lift_pot
```

### 临时覆盖 RTC 参数

参数只对本次进程生效，不修改 YAML。默认配置即为模型输出 50 步、第 10 步申请、第 20 步固定交接：

```bash
bash myscripts/simulation_rtc_tools/scripts/run_rtc_fast_one_task.sh \
  --task lift_pot \
  --episodes 1 \
  --request-index 10 \
  --execution-horizon 10 \
  --fixed-delay-steps 10 \
  --crossfade-steps 4 \
  --ema-alpha 0.36
```

`fixed-delay-steps` 决定仿真的固定交接点，因此交接位置为 `request-index + fixed-delay-steps`。`execution-horizon` 是模型侧 RTC 在动作时间轴上的约束范围，不是模型输出长度，也不是客户端等待步数。脚本启动时会打印 `RTC effective config`，非法组合会在加载模型前退出。同样的五个覆盖参数也适用于 `run_rtc_fast_50x50.sh`。

## 50任务 × 每任务50次

只有单任务 trace 验证通过后再由用户执行：

```bash
bash myscripts/simulation_rtc_tools/scripts/run_rtc_fast_50x50.sh
```

结果位于 `myscripts/simulation_rtc_tools/results/clean/`。每个任务的 `rtc_trace.jsonl` 记录 request、eligible step、wait、promotion、推理耗时、剩余旧动作和 episode 结果。RTC 结果属于推理策略实验，不能与同步 baseline 混写或冒充同步 baseline。

## 无界面快速仿真录像

按 RoboTwin 原生规则保存每个 episode 的视频，不打开 GUI，也不按现实时间限速：

```bash
cd /home/lixiang/PROJECT/LingBotVLA2.0
bash myscripts/simulation_rtc_tools/scripts/run_rtc_video_eval.sh \
  --task click_alarmclock --episodes 1
```

省略 `--task` 即运行全部 50 个任务。视频与日志独立保存在
`evaluation/video_comparison/rtc/<setting>/<参数标签>/`，例如：

```text
evaluation/video_comparison/rtc/clean/
└── request_20-delay_20-switch_40-horizon_20-xfade_10-ema_0p21/
    └── <run>/eval_results/<task>/*.mp4
```

参数标签根据最终生效配置自动生成；同一配置的多次运行继续由时间戳隔离。
可以继续传入
`--request-index`、`--fixed-delay-steps`、`--execution-horizon`、
`--crossfade-steps` 和 `--ema-alpha` 临时覆盖 RTC 参数。

RTC 等待期间没有调用 `take_action()`，因此不会推进物理仿真，也不会向
MP4 写入表示等待时间的静止帧。录像模式会在每个策略动作前刷新相机缓存，
避免上游客户端在整个 action chunk 内重复同一张旧画面。保存的视频展示逐
策略动作轨迹，不展示推理的墙钟等待；无录像评估路径不受影响。

完整边界与时间语义见 [DESIGN.md](DESIGN.md)。
