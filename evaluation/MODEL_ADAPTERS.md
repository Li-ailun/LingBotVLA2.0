# LingBot 双模型评测适配

评测适配层只负责解析模型路径、资产路径、精度和结果目录，不修改或复制模型权重。

## 模型名称

| `--model` | 权重目录 | 配置及归一化目录 |
|---|---|---|
| `step2000` | `models/lingbot_action_step2000/` | 同一目录 |
| `full5k` | `models/lingbot_full_5k_6_60/checkpoints/global_step_5000/hf_ckpt/` | `models/lingbot_full_5k_6_60/` |

解析逻辑位于 `evaluation/model_registry.sh`。`full5k` 直接读取原始 FP32
safetensors；选择 BF16 时是在加载/推理阶段使用 BF16，不会生成或覆盖权重文件。

精度参数：

- `--precision bf16`：`use_bf16=True, use_fp32=False`（默认）。
- `--precision fp32`：`use_bf16=False, use_fp32=True`。

FP32 显存占用明显高于 BF16，单张 RTX 3090 可能无法容纳完整模型；这不影响
适配接口，但实际启动前应确认 GPU 显存。

## 非 RTC 评测

```bash
cd /home/lixiang/PROJECT/LingBotVLA2.0

# 新 full5k 模型，BF16
bash evaluation/lingbot_action_step2000/run_eval.sh \
  demo_clean --model full5k --precision bf16 --episodes 50

# 旧 step2000 模型，FP32
bash evaluation/lingbot_action_step2000/run_eval.sh \
  demo_clean --model step2000 --precision fp32 --episodes 50
```

结果按模型和精度隔离：

```text
evaluation/lingbot_action_step2000/results/<model>/<precision>/<setting>/
```

## RTC 评测

```bash
cd /home/lixiang/PROJECT/LingBotVLA2.0

bash myscripts/simulation_rtc_tools/scripts/run_rtc_fast_50x50.sh \
  --model full5k \
  --precision bf16 \
  --episodes 50 \
  --request-index 20 \
  --fixed-delay-steps 10 \
  --execution-horizon 20 \
  --crossfade-steps 10 \
  --ema-alpha 0.21
```

单任务 RTC：

```bash
bash myscripts/simulation_rtc_tools/scripts/run_rtc_fast_one_task.sh \
  --task click_alarmclock --episodes 1 \
  --model full5k --precision bf16
```

结果按模型和精度隔离：

```text
myscripts/simulation_rtc_tools/results/<model>/<precision>/<setting>/
```

## 精确断点续评

断点续评必须复用原来的运行目录，并提供中断任务的已完成 episode 数、成功数和
下一个 seed。日志使用追加模式，已经产生的日志不会被覆盖。

非 RTC：

```bash
bash evaluation/lingbot_action_step2000/run_eval.sh demo_clean \
  --model full5k --precision bf16 --episodes 50 \
  --resume-run-dir /绝对路径/到/原运行目录 \
  --resume-task place_object_stand \
  --initial-episode-count 10 \
  --initial-success-count 2 \
  --start-seed 100011
```

RTC：

```bash
bash myscripts/simulation_rtc_tools/scripts/run_rtc_fast_50x50.sh \
  --model full5k --precision bf16 --episodes 50 \
  --resume-run-dir /绝对路径/到/原运行目录 \
  --resume-task place_object_stand \
  --initial-episode-count 10 \
  --initial-success-count 2 \
  --start-seed 100011 \
  --request-index 20 --fixed-delay-steps 10 \
  --execution-horizon 20 --crossfade-steps 10 --ema-alpha 0.21
```

续跑时必须保持原模型、精度、setting 和 RTC 参数不变。`--resume-task` 从指定任务
继续，并在完成后按官方任务顺序处理后续任务。

## 录像比较

非 RTC 与 RTC 录像入口同样接受 `--model`、`--precision` 和
`--resume-run-dir`：

```bash
bash evaluation/lingbot_action_step2000/run_video_eval.sh \
  demo_clean --task click_alarmclock --episodes 1 \
  --model full5k --precision bf16

bash myscripts/simulation_rtc_tools/scripts/run_rtc_video_eval.sh \
  --setting demo_clean --task click_alarmclock --episodes 1 \
  --model full5k --precision bf16
```

录像结果按模型、精度、推理类型、setting、RTC 参数和任务分别保存：

```text
evaluation/video_comparison/<model>/<precision>/non_rtc/<setting>/<task>/<run>/
evaluation/video_comparison/<model>/<precision>/rtc/<setting>/<rtc_params>/<task>/<run>/
```
