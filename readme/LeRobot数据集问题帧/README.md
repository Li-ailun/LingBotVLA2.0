# RoboTwin LeRobot v3.0 视频问题帧说明

## 1. 数据集

- 路径：`RoboTwin2_data/lerobot_dataset/RoboTwin_lerobot_v30`
- 版本：LeRobot v3.0
- Episode：2,500
- State/Action 总帧数：548,893
- 帧率：15 FPS
- 相机：`cam_high`、`cam_left_wrist`、`cam_right_wrist`

本目录只记录数据质量检查结果，没有修改原始 LeRobot 数据、State、Action、视频或 metadata。

## 2. 已验证正常的部分

对全部 25 个 Parquet 文件、2,500 个 episode 和 548,893 帧的检查结果：

- `episode_index` 连续覆盖 `0..2499`；
- episode 内 `frame_index` 没有断点；
- episode 内 `timestamp` 按 15 FPS 连续；
- 每个 episode 均从 `frame_index=0, timestamp=0` 开始；
- 全部 546,393 组 episode 内相邻样本均严格满足 `Action[t] == State[t+1]`；
- State/Action 维度和左右臂、夹爪映射未发现错位。

因此，当前问题只指向 MP4 视频帧，不指向 State/Action 时序。

## 3. 视频帧数问题

使用 `ffprobe -count_frames` 对全部 25 个 MP4 进行实际解码计数，结果如下：

| 相机 | State/Action 样本数 | MP4 实际解码帧数 | 少帧数 | 受影响 episode |
|---|---:|---:|---:|---:|
| `observation.images.cam_high` | 548,893 | 548,198 | 695 | 625 |
| `observation.images.cam_left_wrist` | 548,893 | 548,753 | 140 | 138 |
| `observation.images.cam_right_wrist` | 548,893 | 548,789 | 104 | 101 |

三路相机合计少 939 个“样本×相机”视频帧。去重后，807 个训练样本的至少一路相机受影响，涉及 732 个 episode，占全部训练样本约 0.147%。

每个异常 episode 的单路相机少 1–2 帧：

- 657 个 episode 至少有一路相机影响最后 1 个样本；
- 75 个 episode 至少有一路相机影响最后 2 个样本。

## 4. 问题机制

LeRobot 0.4.2 的视频读取器不是显式复制上一帧。它会：

1. 根据样本和 episode 视频起点计算目标时间戳；
2. 解码目标时间附近的视频帧；
3. 选择与目标时间最接近的帧；
4. 只要时间距离未超过容差，就不报错。

由于多个 episode 被连续编码在同一 MP4 中，当当前 episode 尾部缺少 1–2 帧时，读取器可能：

- 返回当前 episode 的最后一张有效图像；或
- 跨过 episode 视频边界，返回下一 episode 的开头图像。

官方 `LeRobotDataset(video_backend="pyav")` 对抽查样本均能成功返回图像，因此这是静默的错误图像问题，不是索引越界异常。

## 5. 已验证的跨 episode 实例

通过对 LingBot 环境中 LeRobot/PyAV 实际返回的图像张量进行逐像素比较，已确认：

- episode 0 `cam_high` 最后一个样本的图像与 episode 1 `cam_high` 第一帧完全相同；
- episode 1 `cam_high` 最后一个样本的图像与 episode 2 `cam_high` 第一帧完全相同；
- episode 17 左、右腕相机最后一个样本与 episode 18 对应相机第一帧完全相同；
- episode 35 `cam_high` 和左腕相机最后一个样本与 episode 36 对应相机第一帧完全相同。

这些实例证明了跨 episode 图像污染确实存在。清单中的“问题帧”是根据全量 metadata 与真实 MP4 帧数缺口定位的风险样本；未对 807 个样本全部进行逐像素跨 episode 相等验证。

## 6. 清单文件

### `问题帧清单.csv`

每行对应一个应排除的训练样本，共 807 行，不包含表头。同一样本的多路异常相机已合并。

### `相机缺帧清单.csv`

每行对应一个“训练样本×异常相机”记录，共 939 行，不包含表头。

关键字段：

- `episode_index`：episode 编号；
- `frame_index`：episode 内帧编号；
- `dataset_index`：LeRobot 全局样本索引；
- `camera`：受影响的相机；
- `episode_length`：State/Action 样本数；
- `available_video_frames`：metadata 时间区间按 15 FPS 对应的实际视频帧数；
- `missing_tail_frames`：该 episode 该相机少帧数；
- `tail_offset`：从 episode 尾部倒数，`1` 表示最后一个样本；
- `recommended_action`：建议处理方式。

## 7. 训练影响与建议

- 不应因为 1–2 帧尾帧问题删除整个 episode；
- 建议保留官方 2,500 条 clean 示教，仅在训练 sampler/dataset wrapper 中跳过 `问题帧清单.csv` 中的 807 个全局索引；
- 不要直接覆盖官方原始数据集；
- 如采用补帧，应复制当前 episode 最后一张有效图像，不应使用下一 episode 的图像；
- 建议在比赛复现报告中披露检测方法、过滤数量和代码位置。

## 8. 本报告的边界

本次检查能确定 MP4 帧数不足以及尾部读取风险，但不包含对所有视频中间帧的黑帧、花屏、视觉重复或内容跳变检测。
