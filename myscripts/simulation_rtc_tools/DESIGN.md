# LingBot-VLA RoboTwin 快速仿真 RTC 设计规格

## 1. 目标

在不修改 RoboTwin 任务、环境、成功判定、随机化、动作预算和现有同步评测入口的前提下，为 `models/lingbot_action_step2000` 新增一个独立的快速仿真 RTC 评测工具。

RTC 工具需要实现：

- LingBot-VLA 每次仍生成 50 步 action chunk；
- 仿真执行旧 chunk 时异步请求新 chunk；
- RTC 模型空间前缀引导；
- 新旧 chunk 交叉淡化；
- 动作 EMA 平滑；
- 用确定性的“策略动作步延迟”表达推理延迟，而不是把结果绑定到 GPU 墙钟速度；
- 正式同步评测入口和已有实机推理工具保持原样。

首版只实现快速仿真 RTC。现实一倍速可视化 RTC 不属于本规格。

## 2. 不变量与官方评测边界

RTC 入口不得改变：

- `TASK_ENV.setup_demo()` 的参数、seed 和测试样本选择；
- clean/randomized task config；
- `TASK_ENV.step_lim`；
- `TASK_ENV.take_action()`；
- `TASK_ENV.check_success()` 和 `TASK_ENV.eval_success`；
- 成功/失败统计和 episode 数量；
- SAPIEN 物理步长、TOPP 轨迹和控制器；
- 原始同步客户端 `RoboTwin_lingbot/script/eval_policy_client_lingbotvla.py`；
- 正式同步评测目录 `evaluation/lingbot_action_step2000/`；
- 实机 RTC 目录 `myscripts/inference_tools/`。

新工具只能改变模型如何产生动作、何时预取下一段动作以及模型输出进入 `take_action()` 前的动作组合方式。

等待推理时不得调用 `TASK_ENV.take_action()`，因此不得增加 `take_action_cnt`，不得推进 SAPIEN，也不得消耗 `step_lim`。

## 3. 采用方案

采用“真实异步计算 + 固定仿真步可用门控”方案：

1. 独立 worker 持有 websocket 连接并进行模型推理；
2. 主仿真线程继续执行旧 action chunk；
3. 新结果即使提前返回，也必须等待固定策略步延迟结束后才能生效；
4. 固定延迟结束而推理仍未返回时，主线程暂停仿真等待；
5. 因此 GPU 性能只改变总墙钟耗时，不改变动作切换的最早仿真步。

不采用以下方案：

- 墙钟驱动的快速仿真 RTC：同一模型在不同 GPU 或负载下会得到不同控制轨迹；
- 同步推理后伪造延迟：可重复但不具备真实的推理/仿真并行；
- 在 `take_action()` 内部切换：会打断官方 TOPP 轨迹语义。

## 4. 固定参数

首版配置如下：

```yaml
chunk_horizon: 50
request_index: 10
fixed_delay_steps: 10
execution_horizon: 10

rtc:
  max_guidance_weight: 10.0
  prefix_attention_schedule: exp

smoothing:
  crossfade_steps: 4
  ema_alpha: 0.36
  exclude_gripper_from_ema: true

fallback:
  inference_not_ready: pause_simulation
  request_failure: fail_episode
```

参数含义：

- `request_index=10`：旧 chunk 已执行 10 步后，在下一次动作执行前采集新观测并发起请求；
- `fixed_delay_steps=10`：请求后必须再执行至少 10 个旧策略动作，新结果才有资格生效；
- 固定在旧 chunk 索引 20 处切换；若结果未就绪，暂停仿真且不消耗动作预算；
- `execution_horizon=10`：RTC 引导约束旧计划中尚未执行的 10 步前缀；
- `crossfade_steps=4` 和 `ema_alpha=0.36` 沿用现有 RLinf 风格平滑设置。

所有索引均为策略动作索引，不是 SAPIEN physics step，也不是墙钟毫秒。

## 5. 目录与隔离方式

```text
myscripts/simulation_rtc_tools/
├── DESIGN.md
├── README.md
├── configs/
│   └── lingbot_action_step2000_rtc_fast.yaml
├── simulation_rtc/
│   ├── __init__.py
│   ├── config.py
│   ├── protocol.py
│   ├── scheduler.py
│   ├── async_worker.py
│   ├── action_blender.py
│   └── metrics.py
├── copied_eval/
│   └── eval_policy_client_lingbotvla_rtc.py
├── server/
│   └── policy_server_rtc.py
├── scripts/
│   ├── run_rtc_fast_one_task.sh
│   └── run_rtc_fast_50x50.sh
└── tests/
```

`copied_eval/eval_policy_client_lingbotvla_rtc.py` 是官方客户端的带来源说明副本。启动器将它复制为 RoboTwin 中一个唯一的新文件名，例如：

```text
RoboTwin_lingbot/script/eval_policy_client_lingbotvla_rtc.py
```

不得覆盖 `eval_policy_client_lingbotvla.py` 或 `script/deploy/websocket_client_policy.py`。

## 6. 组件职责

### 6.1 配置加载器

`config.py` 负责加载并严格验证 RTC 参数。非法值必须在启动任何模型或 episode 前失败。至少验证：

- `chunk_horizon == 50`；
- `0 <= request_index < chunk_horizon`；
- `0 <= fixed_delay_steps <= execution_horizon`；
- `execution_horizon <= chunk_horizon - request_index`；
- `crossfade_steps >= 0`；
- `0 < ema_alpha <= 1`。

### 6.2 协议

`protocol.py` 定义带版本号的请求和响应元数据。每个请求包含：

- `protocol_version`；
- `episode_id`；
- 单调递增 `request_id`；
- 新鲜 observation；
- 旧 chunk 的物理动作；
- 请求时旧 chunk 的执行索引；
- RTC 参数。

响应包含：

- 对应的 `episode_id` 和 `request_id`；
- 50 步物理动作；
- server timing；
- RTC 是否实际启用及服务器使用的参数。

响应 shape、dtype、有限值、ID 或协议版本不正确时必须拒绝，禁止静默降级为普通同步动作。

### 6.3 异步 worker

`async_worker.py` 使用单一后台线程。websocket 连接仅由该线程拥有，避免跨线程共享非线程安全连接。

约束：

- 同时最多一个请求在途；
- 请求队列容量为 1；
- 响应槽容量为 1；
- worker 不接触 `TASK_ENV`；
- episode reset 会递增 `episode_id` 并清除未消费结果；
- 迟到的旧 episode 或旧 request 响应必须丢弃；
- reset、关闭、网络异常必须有限时退出，不能留下后台线程阻塞下一个任务。

首次 chunk 也通过 worker 请求，但主线程同步等待它完成后才开始动作执行。

### 6.4 调度状态机

`scheduler.py` 是不依赖 SAPIEN、websocket 和 PyTorch 的纯状态机。状态为：

```text
BOOTSTRAP -> EXECUTING -> REQUEST_INFLIGHT -> READY_GATED
                       -> WAITING_RESULT
任意状态 -> EPISODE_DONE / FAILED
```

状态机跟踪：

- 当前 chunk；
- 下一个要执行的 chunk 索引；
- 当前 episode/request ID；
- 发起请求时的官方 `take_action_cnt`；
- 已经过的虚拟策略步；
- 新结果是否到达；
- 上一次实际发送给 `take_action()` 的动作。

每次 `TASK_ENV.take_action(action)` 正常返回后，主线程才通知状态机“成功执行一步”。请求等待、网络轮询和结果处理不得推进策略步计数。

### 6.5 动作融合

服务器中的 RTC guidance 在归一化模型动作空间执行，复用：

```text
myscripts/inference_tools/lingbot_rtc/rtc_request.py
myscripts/inference_tools/lingbot_rtc/rtc_sampler.py
```

旧物理动作必须通过当前 checkpoint 的同一个 `FeatureTransform` 转为归一化 RTC target。禁止使用另一份 norm stats 或手工常量转换。

服务器返回反归一化后的 50 步 RoboTwin 动作。客户端在切换时执行：

1. 请求时只发送从索引 10 开始的旧轨迹尾部作为模型侧 RTC target；
2. 固定交接时裁掉新 chunk 中已过去的 5 步，并用末动作补齐为 50 步接口；
3. 对 `old[20:24]` 与对齐后的 `new_raw[5:9]` 执行 RLinf SmoothStep crossfade；
4. Crossfade 和 EMA 均排除夹爪维度；
5. 每个实际执行动作与上一个实际动作做 EMA；
6. 调用一次官方 `TASK_ENV.take_action()`。

仿真动作预计为双臂 14 维，但实现不得硬编码维度或夹爪索引。夹爪索引必须从 checkpoint 导出的 `configs/robot_configs/robotwin.yaml`/FeatureTransform 映射解析，并在启动时验证与模型输出维度一致。解析不唯一时启动失败。

### 6.6 复制版评测客户端

复制版保留官方客户端的：

- episode 构造和 expert seed 筛选；
- instruction 生成；
- 视频处理；
- 成功统计；
- close/cache 行为；
- CLI 配置解析。

仅替换官方循环中“同步推理完整 chunk 后执行”的 policy 部分。新循环：

1. reset 服务和本地 scheduler；
2. 同步等待 bootstrap chunk；
3. 获取 scheduler 的下一个动作并调用 `take_action()`；
4. 在旧 chunk 到达索引 10 时，先调用 `get_obs()` 构造新鲜观测，再异步提交；
5. 每个动作边界轮询响应；
6. 同时满足“固定 5 步门控”和“结果已就绪”才切换；
7. 门控已满足但结果未就绪时暂停仿真并等待 worker；
8. 每个动作后检查原始 `eval_success` 和 `step_lim`。

## 7. 时间语义

RTC 同时记录三个时间量，但控制决策只使用第一个：

| 时间量 | 用途 | 是否决定动作切换 |
|---|---|---|
| 策略动作步 | 固定延迟、chunk 对齐、step budget | 是 |
| SAPIEN 物理时间 | 诊断 TOPP/物理推进 | 否 |
| 墙钟时间 | 推理性能与等待耗时统计 | 否 |

一次 `take_action()` 内可能执行多个 physics step。后台结果即使在这段时间返回，也只能在该次 `take_action()` 完成后的策略动作边界被接纳。

## 8. 失败策略

- bootstrap 推理失败：当前 episode 明确失败并记录原因；
- RTC 请求网络/服务器失败：当前 episode 明确失败，不回退同步普通采样；
- 新结果迟于固定门控：暂停仿真直至返回或达到配置的墙钟故障超时；
- 故障超时：当前 episode 失败，关闭 worker/连接；
- shape、NaN/Inf、ID、版本或动作映射不匹配：立即拒绝并失败；
- 用户中断：停止提交请求，关闭 worker，沿用外层评测器现有的结果/恢复机制；
- episode 已成功或达到 `step_lim`：停止使用任何迟到响应并结束 episode。

墙钟故障超时只用于识别服务故障，不参与正常 RTC 动作对齐。

## 9. 日志与可审计性

每个 episode 记录 JSONL RTC trace，至少包含：

- task、seed、episode_id；
- request_id；
- request policy step 和 chunk index；
- eligible policy step；
- response-ready policy step；
- promotion policy step；
- inference wall latency；
- pause wall duration；
- promotion 时剩余旧动作数；
- crossfade/EMA/RTC 参数；
- stale response、校验失败和异常；
- 最终 success/failure 与官方 `take_action_cnt`。

普通控制台继续保留官方 success-rate 输出，便于现有汇总脚本读取。

## 10. 启动入口

首版提供两个入口：

```bash
# 单任务、单 episode smoke test
bash myscripts/simulation_rtc_tools/scripts/run_rtc_fast_one_task.sh lift_pot

# clean 50任务正式批量评估；默认每任务50次，允许显式改为100
bash myscripts/simulation_rtc_tools/scripts/run_rtc_fast_50x50.sh
```

启动脚本固定使用：

```text
checkpoint: models/lingbot_action_step2000
inference env: lingbotvla
simulation env: RoboTwin
chunk size: 50
one inference server / one simulation slot / one RTX 3090
```

所有结果写入新目录，不与同步结果混合：

```text
myscripts/simulation_rtc_tools/results/
```

## 11. 验证标准

### 11.1 纯单元测试

- index 10 只提交一次请求；
- 请求后的第 0～4 步禁止 promotion；
- 第 5 步且响应 ready 时允许 promotion；
- 第 5 步响应未 ready 时进入 WAITING_RESULT；
- WAITING_RESULT 不增加策略步和官方动作计数；
- 提前到达的结果被缓存而不是提前使用；
- stale episode/request 响应被丢弃；
- chunk 耗尽前无结果时不会越界复用动作；
- success/step limit 会取消未消费结果；
- crossfade 和 EMA 数值与既有实机实现一致；
- 夹爪维度不做 EMA；
- NaN、shape 和协议错误被拒绝。

### 11.2 服务测试

- checkpoint 严格加载；
- bootstrap 请求返回 `[50, D]`；
- RTC 请求确实走 `sample_actions_rtc`，而不是普通 `sample_actions`；
- 物理旧 chunk 经当前 FeatureTransform 归一化后成为 RTC target；
- server response 带正确 request/episode ID 和 RTC 元数据。

### 11.3 隔离与一致性测试

- 对同步评测关键文件计算实施前后 SHA-256，必须完全相同；
- 新启动器 dry-run 指向唯一 RTC 客户端和 RTC server；
- RTC 关闭的复制客户端在相同 seed 下保持官方同步执行结构；
- 不覆盖 `websocket_client_policy.py`；
- 无仿真 episode 的 preflight 能验证配置、checkpoint、端口、环境和文件来源。

### 11.4 分阶段 smoke test

1. fake worker + fake environment 验证完整调度时间线；
2. 真模型服务单请求；
3. `lift_pot` 一个 episode、关闭可视化；
4. 检查 trace：请求在 10、promotion 固定在 20、裁剪 10 步、等待不消耗 step；
5. 小批量多 seed；
6. 通过后才运行 50×50。

## 12. 完成定义

同时满足以下条件才视为工具准备完成：

- 所有单元、协议、隔离测试通过；
- 现有同步入口关键文件哈希未变化；
- `lift_pot` 真实 smoke test 能完成一个 episode；
- trace 能证明固定步门控和实际异步重叠生效；
- 推理等待没有推进 SAPIEN 或消耗 `step_lim`；
- README 明确 RTC 属于独立推理策略实验，不能与官方同步 baseline 结果混为一谈；
- 尚未自动启动 50×50 正式评估，由用户显式执行正式命令。
