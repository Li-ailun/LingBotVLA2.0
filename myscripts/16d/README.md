# R1Pro 16D 数据工具链

这套工具独立于 OpenPI 的20D脚本，供 LingBot-VLA 使用。机器人向量固定为：

```text
[left_arm_j0..j6, left_gripper, right_arm_j0..j6, right_gripper]
```

schema 名称为 `r1pro_dual_arm_16d_left_first_v1`。录制、转换、构建、检查和回放均不使用腰部数据，
也不会把旧20D数组自动截断成16D。

## 1. 录制 ROS bag/MCAP

先加载机器人对应的 ROS 2 环境，再运行：

```bash
cd /home/lixiang/PROJECT/LingBotVLA2.0
source /opt/ros/humble/setup.bash

/usr/bin/python3 myscripts/16d/record_tools/record_r1pro_mcap_episode.py \
  --root "$PWD/datasets/r1pro_mcap_raw" \
  --task "pick_up_object"
```

### 三类盘子/杯子任务

脚本启动时先选择本次录制人员，只接受 `li`、`liu`、`yu`、`xing`。人员在本次程序运行期间保持不变。随后选择三条固定英文指令之一。也可以在命令行同时指定人员和初始指令：

```bash
/usr/bin/python3 myscripts/16d/record_tools/record_r1pro_mcap_episode.py \
  --root "$PWD/datasets/r1pro_mcap_raw" \
  --operator li \
  --instruction-id 1
```

三个编号分别表示：

1. 蓝、粉盘子；蓝、红、黄杯子。
2. 蓝、粉盘子；红、黄、蓝杯子。
3. 粉、蓝盘子；黄、蓝、红杯子。

颜色顺序表示最终从下到上的堆叠顺序。英文指令只约束盘子与杯子的最终上下关系，以及杯子堆最终位于盘子堆上；不限定左侧、右侧或中间区域，也不强制某一种中间运动路径。

等待开始下一条episode时，可输入 `1`、`2` 或 `3` 切换指令。数据自动分类为：

```text
datasets/r1pro_mcap_raw/stack_plates_and_cups/
├── li/
│   ├── stack_plates_cups_01/episode_*/
│   ├── stack_plates_cups_02/episode_*/
│   └── stack_plates_cups_03/episode_*/
├── liu/
├── yu/
└── xing/
```

每个episode的 `record_config.json` 同时记录 `operator_id`、`operator_order`、`task_id`、`instruction_id`、完整 `instruction_text`、`language`和原MCAP episode序号；`task.txt` 保存模型实际读取的完整英文指令。

固定录制11个话题：三路RGB图像、左右臂关节反馈、左右夹爪反馈、左右臂关节目标和左右夹爪目标。

交互流程：

- 准备好后按 `Enter` 启动一个 episode。
- 录制中按 `Enter` 停止并保存。
- 录制中输入 `q` 后按 `Enter`，取消并删除当前 episode；编号不会增加，可以重录。
- 等待下一次提示继续录制；在待机提示输入 `q` 退出程序。

如果只想直接录制、不启用录前时间戳稳定性检查：

```bash
/usr/bin/python3 myscripts/16d/record_tools/record_r1pro_mcap_episode.py \
  --root "$PWD/datasets/r1pro_mcap_raw" \
  --task "pick_up_object" \
  --enable-timestamp-monitor false
```

## 2. MCAP 转16D中间格式

```bash
cd /home/lixiang/PROJECT/LingBotVLA2.0
source /opt/ros/humble/setup.bash

/usr/bin/python3 myscripts/16d/convert_tools/export_r1pro_mcap_to_intermediate.py \
  --input "$PWD/datasets/r1pro_mcap_raw/pick_up_object" \
  --output "$PWD/datasets/r1pro_intermediate_16d/pick_up_object" \
  --fps 15 \
  --max-image-dt 0.12 \
  --max-vec-dt 0.20 \
  --overwrite
```

输出会镜像“人员/子任务/原MCAP episode”目录层级，并在`manifest.json`和每条`episode_meta.json`中保留人员、原MCAP序号以及源/中间目录对应关系。

每个 episode 生成 `state.npy [N,16]`、`action.npy [N,16]`、`timestamps.npy`、三路图像、
`episode_meta.json` 和逐帧拒绝记录。旧 bag 中即使包含腰部话题也不会被读取。

盘子/杯子任务可以一次递归转换全部三个指令目录，并保留分类层级：

```bash
/usr/bin/python3 myscripts/16d/convert_tools/export_r1pro_mcap_to_intermediate.py \
  --input "$PWD/datasets/r1pro_mcap_raw/stack_plates_and_cups" \
  --output "$PWD/datasets/r1pro_intermediate_16d/stack_plates_and_cups" \
  --fps 15 \
  --max-image-dt 0.12 \
  --max-vec-dt 0.20
```

## 3. 构建 LeRobot v2.1

该脚本沿用原20D工具所使用的 OpenPI Python/LeRobot v2.1环境：

```bash
cd /home/lixiang/PROJECT/LingBotVLA2.0
export HF_LEROBOT_HOME="$PWD/datasets/lerobot"

/home/lixiang/PROJECT/openpi/.venv/bin/python \
  myscripts/16d/convert_tools/build_lerobot_from_r1pro_intermediate.py \
  --input "$PWD/datasets/r1pro_intermediate_16d/pick_up_object" \
  --repo-id lixiang/pick_up_object_16d \
  --fps 15 \
  --overwrite
```

构建器会严格检查16维、有限值、state/action/timestamp帧数一致性，并在数据集 `meta/` 下写入
`r1pro_schema.json`。传入20D旧数据会直接报错。

构建器会递归读取三个指令子目录，将每个episode的完整英文 `instruction_text` 写入LeRobot任务字段：

```bash
/home/lixiang/PROJECT/openpi/.venv/bin/python \
  myscripts/16d/convert_tools/build_lerobot_from_r1pro_intermediate.py \
  --input "$PWD/datasets/r1pro_intermediate_16d/stack_plates_and_cups" \
  --repo-id lixiang/stack_plates_and_cups_16d \
  --fps 15 \
  --overwrite \
  --episodes-per-worker 10
```

大数据集转换会按默认每10条episode重启子进程以释放图像和Arrow内存。中断后可在输入集合及排序不变的前提下使用`--resume`继续；不要同时使用`--overwrite`和`--resume`。

四个人的数据会合并到同一个LeRobot数据集，固定顺序为：

```text
子任务01：li -> liu -> yu -> xing
子任务02：li -> liu -> yu -> xing
子任务03：li -> liu -> yu -> xing
```

每个人内部按原MCAP episode序号排列。LeRobot使用标准零基连续编号`episode_000000...`。数据集同时生成：

```text
meta/episode_source_map.jsonl
meta/episode_source_map.csv
```

映射包含LeRobot零基episode编号、从1开始的人工查看序号、人员、子任务、原MCAP编号和路径、intermediate路径及帧数。发现坏数据时先查映射，删除对应原MCAP，然后用`--overwrite`依次重建intermediate和LeRobot；新LeRobot编号会自动压紧，映射也会重新生成。不能只删除MCAP后复用旧intermediate。

## 4. 数据审计与曲线

完整映射/缺帧审计：

```bash
/home/lixiang/PROJECT/openpi/.venv/bin/python \
  myscripts/16d/lerobot_tools/check_lerobot_v21_missing_audit.py \
  --raw-root "$PWD/datasets/r1pro_mcap_raw/pick_up_object" \
  --inter-root "$PWD/datasets/r1pro_intermediate_16d/pick_up_object" \
  --lerobot-root "$PWD/datasets/lerobot/lixiang/pick_up_object_16d" \
  --out-dir "$PWD/reports/lerobot_quality_reports/pick_up_object_16d_missing_audit" \
  --expected-dim 16
```

绘制 `state/action` 曲线：

```bash
MPLCONFIGDIR=/tmp/matplotlib-r1pro16d \
/home/lixiang/PROJECT/openpi/.venv/bin/python \
  myscripts/16d/lerobot_tools/check_lerobot_v21_numeric_and_plot.py \
  --root "$PWD/datasets/lerobot/lixiang/pick_up_object_16d" \
  --out-dir "$PWD/reports/lerobot_quality_reports/pick_up_object_16d_numeric" \
  --expected-dim 16 \
  --plot-mode all \
  --interactive-groups
```

## 5. 回放 LeRobot

键盘/OpenCV回放：

```bash
/home/lixiang/PROJECT/openpi/.venv/bin/python \
  myscripts/16d/lerobot_tools/replay_lerobot_v21_parquet_images_keyboard.py \
  --root "$PWD/datasets/lerobot/lixiang/pick_up_object_16d" \
  --episode-index 0 \
  --rate 1
```

Web回放（三路图像、四组16D曲线和当前帧数值）：

```bash
/home/lixiang/PROJECT/openpi/.venv/bin/python \
  myscripts/16d/lerobot_tools/replay_lerobot_v21_web.py \
  --root "$PWD/datasets/lerobot/lixiang/pick_up_object_16d" \
  --episode-index 0 \
  --rate 1 \
  --host 127.0.0.1 \
  --port 8050 \
  --open-browser
```

## 6. 自动验证

```bash
cd /home/lixiang/PROJECT/LingBotVLA2.0
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/lixiang/PROJECT/openpi/.venv/bin/python -m pytest -q myscripts/16d/tests
```
