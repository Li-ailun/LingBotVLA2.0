下面是结合当前模型文件、训练配置、权重信息以及绝对关节动作语义后的最新版完整总结。

# 一、`lingbot_full_5k_6_60` 完整训练配置

## 1. 模型基本信息

| 项目 | 配置 |
|---|---|
| 模型名称 | `lingbot_full_5k_6_60` |
| 训练起点 | 官方 `lingbot-vla-v2-6b` 基座 |
| 训练方式 | 全参数后训练 |
| LoRA | 未使用 |
| 最终Checkpoint | `global_step_5000` |
| 可训练参数 | 6,375,906,359，约63.76亿 |
| 冻结学生参数 | 无 |
| 数据 | RoboTwin 50任务 clean-only |
| 随机种子 | 42 |

配置依据：

- [lingbotvla_cli.yaml](/home/lixiang/PROJECT/LingBotVLA2.0/models/lingbot_full_5k_6_60/lingbotvla_cli.yaml)
- [full_train_parameters.json](/home/lixiang/PROJECT/LingBotVLA2.0/models/lingbot_full_5k_6_60/full_train_parameters.json)
- [robotwin.yaml](/home/lixiang/PROJECT/LingBotVLA2.0/models/lingbot_full_5k_6_60/configs/robot_configs_clean/robotwin.yaml)

## 2. 训练规模

| 参数 | 配置 |
|---|---:|
| GPU数量 | 6 |
| 每卡 micro batch | 60 |
| 梯度累积 | 1 |
| Global batch | 360 |
| 最大训练步数 | 5,000 |
| 实际训练步数 | 5,000 |
| 数据并行 | FSDP2 |
| FSDP shard数量 | 6 |
| Tensor Parallel | 1 |
| Pipeline Parallel | 1 |
| Context Parallel | 1 |
| Expert Parallel | 1 |

Global batch计算：

\[
60\times6\times1=360
\]

即每次参数更新理论上使用360个训练样本。

配置里的：

```yaml
num_train_epochs: 29000
```

只是一个很大的epoch上限。真正终止训练的是：

```yaml
max_steps: 5000
```

## 3. 折算训练Epoch

clean-only数据集规模：

```text
548,893 frames
```

单个epoch所需更新步数：

\[
548893/360\approx1524.7
\]

5000步的样本曝光量：

\[
5000\times360=1,800,000
\]

理论epoch数：

\[
1800000/548893\approx3.28
\]

因此，这个模型大约训练了：

```text
3.28 epoch
```

该结果是根据全局batch和数据帧数计算的理论值。如果训练加载器存在过滤、重复采样或动态组批，实际有效epoch可能略有差异。

## 4. 学习率

| 参数 | 配置 |
|---|---:|
| Warmup初始学习率 | 0 |
| 峰值学习率 | `1e-5` |
| 最小学习率 | `1e-6` |
| Warmup比例 | 5% |
| Warmup步数 | 约250步 |
| 调度方式 | Cosine |
| 衰减范围 | 完整训练过程 |

配置：

```yaml
lr_start: 0.0
lr: 1.0e-5
lr_min: 1.0e-6
lr_warmup_ratio: 0.05
lr_decay_style: cosine
lr_decay_ratio: 1.0
```

学习率过程大致是：

```text
step 0～250：
0 → 1e-5

step 250～5000：
1e-5 → 1e-6，Cosine衰减
```

## 5. 优化器和梯度配置

| 参数 | 配置 |
|---|---|
| 优化器 | `anyprecision_adamw` |
| Weight decay | 0 |
| 最大梯度范数 | 1.0 |
| 梯度累积 | 1 |
| 梯度检查点 | 开启 |
| 激活卸载 | 关闭 |
| FSDP CPU offload | 关闭 |
| `torch.compile`训练 | 关闭 |

模型是全量参数训练，不是LoRA或只训练动作头。

## 6. 训练精度

配置：

```yaml
enable_fp32: true
enable_mixed_precision: true
action_fp32: false
```

训练产物统计：

```text
torch.float32: 6,375,906,359 parameters
```

含义是：

- 最终checkpoint以FP32保存；
- 训练框架启用了混合精度执行；
- 没有单独强制动作头使用额外FP32路径；
- 评估时可以临时将权重转换成BF16；
- BF16评估不会修改磁盘上的原始FP32权重。

## 7. 输入与Action Chunk

| 参数 | 配置 |
|---|---|
| 相机数量 | 3 |
| 相机 | 顶部、左腕、右腕 |
| 图像尺寸 | 256 |
| 图像增强 | 关闭 |
| 当前机器人状态 | 14D |
| 模型动作输出 | 50步×14D |
| Action chunk | 50 |
| 状态归一化 | `none` |
| 动作归一化 | `bounds_99_woclip` |
| 使用未来图像监督 | 是 |
| DataLoader workers | 4 |
| Prefetch factor | 4 |
| Drop last | 是 |

模型统一接口设置为：

```yaml
max_state_dim: 55
max_action_dim: 55
```

但RoboTwin真实有效状态和动作仍为14D，其余位置是统一模型空间中的填充维度。

## 8. 绝对关节动作

这个模型不是关节增量动作，而是绝对关节目标。

配置：

```yaml
actions:
- action.arm.position:
    subtract_state: false

- action.effector.position:
    subtract_state: false
```

动作顺序为：

```text
左臂6关节
左夹爪1维
右臂6关节
右夹爪1维
```

共14维。

动作语义：

| 部分 | 表示方式 |
|---|---|
| 左臂6关节 | 绝对目标位置 |
| 左夹爪 | 绝对目标位置 |
| 右臂6关节 | 绝对目标位置 |
| 右夹爪 | 绝对目标位置 |

即：

```text
双臂绝对关节 + 双夹爪绝对位置
```

这里的：

```yaml
model:
  incremental_training: false
```

表示没有启用增量式训练模式，和动作是否使用增量关节是两个不同概念。

## 9. 深度和视频特征辅助监督

当前模型启用了：

```yaml
use_future_depth: true
use_future_video: true
```

辅助监督包括：

- 当前深度特征对齐；
- 未来深度特征预测；
- 当前视频特征对齐；
- 未来视频特征预测。

教师模型：

- MoGe；
- LingBot-Depth / MoRGBD；
- DINO Video Teacher。

损失权重：

| 损失 | 权重 |
|---|---:|
| 当前深度损失 | 0.004 |
| 未来深度损失 | 0.004 |
| 未来视频损失 | 0.004 |
| 视频Cosine损失 | 0.2 |
| 视频MSE损失 | 1.0 |

这些教师模型在训练时保持冻结，负责生成监督目标；学生LingBot-VLA本身进行全量参数更新。

## 10. MoE配置

| 参数 | 配置 |
|---|---:|
| MoE层 | 0～35，共36层 |
| 每层专家数量 | 32 |
| 每个Token激活专家 | Top-4 |
| 路由激活 | Sigmoid |
| Routed scaling factor | 4.0 |
| Token expert intermediate size | 512 |
| Shared intermediate size | 704 |
| Router Z loss | `1e-4` |
| Sequence-wise辅助损失 | `1e-3` |

# 二、`full5k`与官方93.52%模型对比

| 项目 | 我们的`full5k` | 官方93.52%模型 |
|---|---:|---:|
| 基座 | `lingbot-vla-v2-6b` | `lingbot-vla-v2-6b` |
| 参数训练方式 | 全量训练 | 全量训练 |
| 数据 | clean-only | clean + randomized |
| 任务数量 | 50 | 50 |
| 动作表示 | 绝对关节+绝对夹爪 | 绝对关节+绝对夹爪 |
| Action chunk | 50 | 50 |
| Micro batch | 60/GPU | 32/GPU |
| GPU数量 | 6 | 32 |
| Global batch | 360 | 1024 |
| 梯度累积 | 1 | 1 |
| Steps | 5,000 | 50,000 |
| 样本曝光量 | 180万 | 5120万 |
| 优化器 | AnyPrecision AdamW | Muon |
| 峰值学习率 | `1e-5` | `1e-4` |
| 最小学习率 | `1e-6` | `5e-5` |
| Warmup | 5%，约250步 | 未显式启用 |
| LR调度 | Cosine | Cosine |
| Weight decay | 0 | 0 |
| 梯度检查点 | 开启 | 关闭 |
| 训练`torch.compile` | 关闭 | 开启 |
| FP32参数 | 是 | 是 |
| 混合精度框架 | 是 | 是 |
| 当前深度监督 | 开启 | 开启 |
| 未来深度监督 | 开启 | 开启 |
| 当前/未来视频监督 | 开启 | 开启 |
| MoGe教师 | 使用 | 使用 |
| LingBot-Depth教师 | 使用 | 使用 |
| DINO Video Teacher | 使用 | 使用 |
| Clean结果 | 当前本地评估中 | 93.52% |
| Randomized结果 | 尚待完整评估 | 92.80% |

# 三、样本曝光量差异

我们的训练量：

\[
5000\times360=1,800,000
\]

官方训练量：

\[
50000\times1024=51,200,000
\]

比例：

\[
51,200,000/1,800,000\approx28.44
\]

官方训练过程的样本曝光量约是我们的：

```text
28.4倍
```

但这里不能直接解释成官方在同一份数据上训练了28.4倍epoch，因为官方训练集合并了规模更大的randomized数据。

# 四、主要相同点

我们和官方模型相同的核心设置包括：

1. 都从 `lingbot-vla-v2-6b` 开始；
2. 都进行全量参数后训练；
3. 都使用三路RGB；
4. 都使用50步action chunk；
5. 都使用绝对关节动作；
6. 都使用绝对夹爪动作；
7. 都启用MoE；
8. 都启用当前深度监督；
9. 都启用未来深度预测；
10. 都启用当前/未来视频特征监督；
11. 都使用MoGe、LingBot-Depth和DINO教师；
12. 都使用FP32模型参数及混合精度训练框架。

# 五、主要差异

真正影响训练结果的核心差异是：

1. 数据不同

```text
我们：clean-only
官方：clean + randomized
```

这是最关键的泛化差异。官方模型在训练期间直接见过randomized场景，而比赛禁止我们使用randomized训练。

2. 训练步数不同

```text
我们：5,000步
官方：50,000步
```

3. Global batch不同

```text
我们：360
官方：1024
```

4. 学习率不同

```text
我们：峰值1e-5，最低1e-6
官方：峰值1e-4，最低5e-5
```

官方峰值学习率是我们的10倍。

5. 优化器不同

```text
我们：AnyPrecision AdamW
官方：Muon
```

6. 工程性能配置不同

```text
我们：梯度检查点开启，torch.compile关闭
官方：梯度检查点关闭，torch.compile开启
```

这主要影响显存和训练速度，通常不是成功率差异的第一来源。

# 六、关于绝对动作与RTC

我们的模型和官方模型都采用绝对关节动作，所以RTC没有改变模型动作语义。

当前RTC做的是：

```text
绝对关节旧chunk
        ↓
提前请求新chunk
        ↓
按时间索引裁剪并对齐
        ↓
绝对目标轨迹Crossfade
        ↓
EMA平滑
        ↓
官方RoboTwin take_action()
```

这能够缓解绝对目标在chunk切换时的跳变，但没有把绝对动作转换成增量动作。

# 七、最终概括

我们的模型：

> 从官方`lingbot-vla-v2-6b`基座开始，使用RoboTwin 50任务clean-only数据，在6张GPU上进行FSDP2全量参数训练；每卡batch 60、global batch 360、无梯度累积，使用AnyPrecision AdamW，峰值学习率`1e-5`，前250步warmup后Cosine衰减至`1e-6`，总训练5000步，约合clean数据3.28 epoch。模型使用50步action chunk，输出14维绝对关节和夹爪目标，同时启用当前深度、未来深度以及当前/未来视频特征辅助监督。

与官方93.52%模型相比：

> 网络主体、绝对动作表示、MoE和深度/视频监督方向基本相同；主要差异是我们只能使用clean数据，global batch为360而非1024，训练5000步而非50000步，优化器为AnyPrecision AdamW而非Muon，峰值学习率为`1e-5`而非`1e-4`。官方还直接使用了randomized训练数据，因此官方93.52%/92.80%的结果不能作为clean-only比赛方案可直接复现的目标。