# 评估过程记录

该模块只旁路记录评估元数据，不参与模型推理、动作执行、环境推进或
`check_success()` 判定。RTC和非RTC评估均使用项目自己的隔离客户端，官方
LingBot-VLA与RoboTwin评测客户端保持不变。

每个新的 `Run directory` 会生成：

- `run_manifest.json`：模型、权重索引指纹、BF16/FP32、RTC参数、任务配置、
  action chunk、compile设置、GPU/服务数量、代码commit以及每次恢复会话；
- `eval_results/<task>/episodes.jsonl`：逐episode的环境seed、指令、成功状态、
  策略步数、推理次数、端口、录像路径；
- `successful_episodes.jsonl`：所有成功episode的汇总索引；
- RTC原有的 `eval_results/<task>/rtc_trace.jsonl` 保持不变。

断点续评会向同一目录追加记录。记录器以
`task + episode_index + environment_seed` 去重，不会重复写入同一episode。

查看全部成功episode：

```bash
jq -r '[.task, .episode_index, .environment_seed, .policy_steps] | @tsv' \
  "$RUN/successful_episodes.jsonl"
```

只查看某个任务：

```bash
jq -r 'select(.task == "lift_pot") |
  [.task, .episode_index, .environment_seed, .policy_steps] | @tsv' \
  "$RUN/successful_episodes.jsonl"
```

环境seed可以复现同一仿真布局。LingBot-VLA动作生成噪声按推理服务从初始
seed 42连续推进，因此单独复跑某个环境seed能够复现场景，但不保证逐动作或
成功结果完全一致。
