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
  --max-vec-dt 0.20
```

每个 episode 生成 `state.npy [N,16]`、`action.npy [N,16]`、`timestamps.npy`、三路图像、
`episode_meta.json` 和逐帧拒绝记录。旧 bag 中即使包含腰部话题也不会被读取。

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
