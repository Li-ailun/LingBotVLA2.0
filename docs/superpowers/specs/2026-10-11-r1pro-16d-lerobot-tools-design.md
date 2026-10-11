# R1Pro 16D LeRobot 工具链设计

## 目标

在 `LingBotVLA2.0/myscripts/16d/` 中建立一套独立的 R1Pro 数据工具链，复用现有
`openpi/myscripts/20d/` 的成熟工作流，但只处理双臂七关节和双夹爪，不录制或导出四维腰部。

所有状态和动作统一为以下 16 维顺序：

```text
[left_arm_j0 .. left_arm_j6, left_gripper,
 right_arm_j0 .. right_arm_j6, right_gripper]
```

原始 `openpi/myscripts/20d/`、LingBot 官方代码、仿真评估和 RTC 工具均不得修改。

## 目录结构

```text
myscripts/16d/
├── README.md
├── common/
│   └── schema.py
├── record_tools/
│   └── record_r1pro_mcap_episode.py
├── convert_tools/
│   ├── export_r1pro_mcap_to_intermediate.py
│   └── build_lerobot_from_r1pro_intermediate.py
├── lerobot_tools/
│   ├── check_lerobot_v21_missing_audit.py
│   ├── check_lerobot_v21_numeric_and_plot.py
│   ├── replay_lerobot_v21_parquet_images_keyboard.py
│   └── replay_lerobot_v21_web.py
└── tests/
```

`common/schema.py` 是16维顺序、名称、切片和核心话题集合的唯一事实来源，避免各脚本重复硬编码。

## ROS bag 录制

录制器继续使用 ROS 2 bag/MCAP，并保留现有键盘交互语义：

- 用户通过键盘开始一个 episode。
- 用户通过键盘结束当前 episode。
- 结束后明确选择保留或丢弃。
- 丢弃的 episode 不进入后续数据转换；保留的 episode 连同元数据落盘。
- 继续保留任务名、episode 编号、时间戳和相机录制能力。

核心录制话题为三路图像，加以下8类机器人话题：

```text
/hdas/feedback_arm_left
/hdas/feedback_gripper_left
/hdas/feedback_arm_right
/hdas/feedback_gripper_right
/motion_target/target_joint_state_arm_left
/motion_target/target_position_gripper_left
/motion_target/target_joint_state_arm_right
/motion_target/target_position_gripper_right
```

不得把 `/hdas/feedback_torso` 或 `/motion_target/target_joint_state_torso` 放入任何16D必要话题集合，
也不提供 `--torso-mode` 等会暗示腰部属于训练数据的参数。录制器不发布机器人控制命令，只记录已有话题。

## MCAP 到中间格式

每个保留 episode 输出三路图像以及：

```text
state.npy   shape [N, 16]
action.npy  shape [N, 16]
```

组装顺序严格为左臂7维、左夹爪1维、右臂7维、右夹爪1维。转换器按相机时间轴匹配机器人状态与动作，
沿用原20D工具经过验证的时间对齐、最大时间差检查、坏帧报告和 episode 元数据，但删除所有腰部提取与腰部时间基修复逻辑。

输入 bag 即使含有旧腰部话题，也只忽略这些额外话题；左右臂或左右夹爪任一必要话题缺失时必须报告失败，不能填零或悄悄降维。

## LeRobot v2.1 构建

构建器读取16D中间格式并输出 LeRobot v2.1 数据集。元数据必须声明：

- `observation.state`: shape `[16]`，名称为16D左先右后状态名称。
- `action`: shape `[16]`，名称为16D左先右后目标动作名称。
- 三路相机字段、FPS、episode/task 索引和视频编码方式沿用原20D工作流。

构建前验证所有 episode 的帧数一致性、数组有限值和精确维度。20D输入必须显式报错，不能自动截断最后四维，
以免把旧版右先左后的20D数组错误解释成新版16D数组。

## 检查、绘图与回放

数值检查与绘图按四组展示：左臂7维、左夹爪、右臂7维、右夹爪。所有默认索引和名称来自公共 schema，
不显示腰部图表。缺失审计检查三路图像、16D数值、源话题覆盖率及时间匹配报告。

键盘回放和 Web 回放保留现有 episode/frame 导航、三路画面及 state/action 曲线能力，默认按左侧组后右侧组显示。
加载时必须验证 `meta/info.json` 与 parquet 中的实际向量均为16D，维度或名称不一致时给出清晰错误。

## 兼容性与错误处理

- Python和ROS依赖沿用原20D脚本，不新增不必要依赖。
- 所有默认路径改为 `LingBotVLA2.0` 下的16D数据与报告目录，同时允许命令行覆盖。
- 禁止隐式左右交换、隐式截断、补零和用腰部数据填充缺失值。
- 每个错误包含 episode、字段/话题、期望维度和实际维度。
- 所有输出元数据记录 schema 名称和版本，例如 `r1pro_dual_arm_16d_left_first_v1`。

## 测试与验收

自动化测试至少覆盖：

1. 公共 schema 恰好16维且左先右后。
2. 合成左右臂/夹爪消息组装后的 state/action 索引完全正确。
3. 录制核心话题不包含任何 torso 话题。
4. 16D中间数组可构建 LeRobot 元数据，20D或错误维度会失败。
5. 绘图和回放分组索引覆盖0到15且不重复、不越界。
6. 数据转换时，旧 bag 中含额外腰部话题会被忽略，必要双臂/夹爪话题缺失时失败。
7. 所有迁移脚本至少通过 `--help` 和 Python 语法编译检查。

最终验收还包括用一个真实保留 episode 做端到端冒烟测试：录制/读取 MCAP、导出 `[N,16]`、构建 LeRobot v2.1、
生成曲线并分别用键盘和 Web 工具读取。若当前机器没有连接ROS设备，录制部分只验证命令和话题集合，真实采集留给设备端执行。
