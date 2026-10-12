# LingBotVLA2.0 服务器评测复现手册

本文以本地项目 `/home/lixiang/PROJECT/LingBotVLA2.0` 当前代码为准，目标是在
服务器 `/home/hddData/User/lixiang/LingBotVLA2.0_new` 中复现 full5k 的
RoboTwin 非 RTC、RTC、断点续评、逐 episode 记录与录像功能。

本文只覆盖评测；不覆盖训练和实机数据录制。

## 1. 当前 Git 状态

生成本文时：

```text
HEAD: fd34838 conda环境只依赖lingbotvla和RoboTwin
工作区：clean
```

最近十个提交：

```text
fd34838 conda环境只依赖lingbotvla和RoboTwin
a51becd 指令增强的规则
b414b8d 比赛实机demo数据录制即处理指令
8c956ea 比赛实机demo数据录制即处理指令
c135c32 评估配置与成功 seed 记录功能
c032608 评估配置与成功 seed 记录功能
fd2ff4b 数据增强的规则
7c5d17f 数据增强的规则
33c929f 5k full 的成功率提升百分之二十,rtc比非rtc更可以发挥模型学到的能力
a87e24f Add LingBot-VLA model adapters and improve evaluation scripts
```

服务器检查命令：

```bash
cd /home/hddData/User/lixiang/LingBotVLA2.0_new
git log --oneline -10
git status --short
git rev-parse HEAD
```

服务器必须至少更新到 `fd34838`，否则不会包含取消 `point-flow-sim` 依赖的
运行时路径解析。

## 2. 项目结构与 Git 边界

与评测有关的主干结构：

```text
LingBotVLA2.0/
├── evaluation/
│   ├── model_registry.sh
│   ├── runtime_paths.sh
│   ├── prepare_runtime.sh
│   ├── runtime_assets/             # 小型配置与外部资产清单
│   ├── eval_provenance/
│   ├── MODEL_ADAPTERS.md
│   └── lingbot_action_step2000/
│       ├── policy_server.py
│       ├── run_eval.sh
│       ├── run_video_eval.sh
│       ├── copied_eval/
│       └── tests/
├── myscripts/
│   ├── simulation_rtc_tools/
│   │   ├── configs/
│   │   ├── copied_eval/
│   │   ├── scripts/
│   │   ├── server/
│   │   ├── simulation_rtc/
│   │   └── tests/
│   └── video_tools/
├── lingbot-vla-v2/             # Git中直接跟踪，不是submodule
├── RoboTwin_lingbot/           # Git中直接跟踪，不是submodule
├── models/                     # Git忽略，需要额外复制/下载
├── RoboTwin2_data/             # Git忽略
└── docs/
```

Git包含：

```text
evaluation/
myscripts/simulation_rtc_tools/
myscripts/video_tools/
lingbot-vla-v2/ 的源码
RoboTwin_lingbot/ 的源码
evaluation/runtime_assets/ 中可恢复的任务配置与下载清单
```

Git不包含：

```text
models/
RoboTwin2_data/
RoboTwin_lingbot/assets/ 下下载的大型仿真资产
RoboTwin_lingbot/envs/curobo/
任何 **/results/ 和 eval_results/
evaluation/video_comparison/
日志、PID文件和视频
Conda环境、CUDA和驱动
```

仓库没有 `.gitmodules`，clone后不需要执行 `git submodule update`。

### 大型依赖下载边界

以下内容不上传Git：模型权重、约16GB RoboTwin资产和约241MB CuRobo源码。
固定来源记录在 `evaluation/runtime_assets/manifest.json`：

```text
RoboTwin 2.0资产：
https://huggingface.co/datasets/TianxingChen/RoboTwin2.0

CuRobo源码：
https://github.com/NVlabs/curobo.git
commit d64c4b005459db10c5dd867d8b30a87d5bda9bdb
```

官方资产页提供 `background_texture.zip`、`embodiments.zip`、`objects.zip`。
解压后必须形成下面的直接层级，不能出现 `assets/assets/`：

```text
RoboTwin_lingbot/assets/background_texture/
RoboTwin_lingbot/assets/embodiments/aloha-agilex/
RoboTwin_lingbot/assets/objects/objaverse/list.json
```

准备脚本默认只复制几十KB任务配置并验证已有资产，不会自动下载大型文件：

```bash
bash evaluation/prepare_runtime.sh
```

若服务器已有共享资产，可用绝对路径建立三个细粒度软链接：

```bash
bash evaluation/prepare_runtime.sh \
  --assets-source /absolute/path/to/shared/assets
```

缺少CuRobo时，脚本会打印固定仓库和commit。确认CUDA编译环境正常后，才显式执行：

```bash
bash evaluation/prepare_runtime.sh --install-curobo
```

也可自行按manifest固定commit安装；不要安装未固定版本的 `nvidia-curobo` 来冒充
本项目已验证源码。

## 3. 模型注册系统

注册文件：

```text
evaluation/model_registry.sh
```

当前支持两个模型别名：

| `--model` | 权重目录 | 配置资产根目录 |
|---|---|---|
| `step2000` | `models/lingbot_action_step2000/` | 同一目录 |
| `full5k` | `models/lingbot_full_5k_6_60/checkpoints/global_step_5000/hf_ckpt/` | `models/lingbot_full_5k_6_60/` |

注册器还解析：

```text
LINGBOT_MODEL_ID
LINGBOT_MODEL_PATH
LINGBOT_MODEL_ASSET_ROOT
LINGBOT_TRAINING_CONFIG
LINGBOT_ROBOT_CONFIG
LINGBOT_NORM_STATS
LINGBOT_MODELS_ROOT
```

选择模型的正式方式只有命令行参数：

```bash
--model full5k
```

当前不支持：

```bash
export LINGBOT_MODEL=full5k       # 无效
--model-path /arbitrary/path      # 高层包装脚本不接受
```

不应手工修改脚本中的 `MODEL_NAME`。新增更高训练步数checkpoint时，在
`model_registry.sh` 增加一个新别名即可，RTC、非RTC、录像和记录功能都可复用。

支持的精度：

```text
--precision bf16   # 默认；加载/推理时转换，不改磁盘权重
--precision fp32   # 3090上可能显存不足
```

注册器启动前会检查以下文件：

```text
<model_path>/model.safetensors.index.json
<model_path>/config.json
<asset_root>/assets/norm_stats/robotwin_clean.json
<robot_config>
<training_config>
```

## 4. 非 RTC 入口

入口：

```text
evaluation/lingbot_action_step2000/run_eval.sh
```

语法：

```bash
bash evaluation/lingbot_action_step2000/run_eval.sh SETTING [参数]
```

`SETTING`只能是：

```text
demo_clean
demo_randomized
```

支持参数：

| 参数 | 含义 | 默认值 |
|---|---|---|
| `--model` | `step2000` 或 `full5k` | `step2000` |
| `--precision` | `bf16` 或 `fp32` | `bf16` |
| `--task` | 单任务；省略则按官方顺序跑50任务 | 空 |
| `--episodes` | 每任务episode数 | `100` |
| `--resume-run-dir` | 原Run目录 | 空 |
| `--resume-task` | 从指定任务继续 | 空 |
| `--initial-episode-count` | 当前任务已有episode数 | `0` |
| `--initial-success-count` | 当前任务已有成功数 | `0` |
| `--start-seed` | 下一个环境seed | 空 |
| `--realtime` | 单任务GUI现实时间观察 | 关闭 |
| `--dry-run` | 打印最终命令，不启动 | 关闭 |

full5k、clean、每任务50次：

```bash
cd /home/hddData/User/lixiang/LingBotVLA2.0_new

bash evaluation/lingbot_action_step2000/run_eval.sh \
  demo_clean \
  --model full5k \
  --precision bf16 \
  --episodes 50
```

单任务：

```bash
bash evaluation/lingbot_action_step2000/run_eval.sh \
  demo_clean \
  --model full5k \
  --precision bf16 \
  --task click_alarmclock \
  --episodes 1
```

精确断点续评：

```bash
RUN=/absolute/path/to/original/run

bash evaluation/lingbot_action_step2000/run_eval.sh \
  demo_clean \
  --model full5k \
  --precision bf16 \
  --episodes 50 \
  --resume-run-dir "$RUN" \
  --resume-task place_object_stand \
  --initial-episode-count 10 \
  --initial-success-count 2 \
  --start-seed 100011
```

高层 `run_eval.sh` 当前固定：

```text
num_gpus=1
num_per_gpu=1
start_port=9330（来自底层launcher默认值）
use_length=50
use_compile=True
output=evaluation/lingbot_action_step2000/results/<model>/<precision>/<setting>/
```

高层入口没有 `--gpu`、`--start-port`、`--output-base` 参数。

## 5. RTC入口

### 5.1 单任务

入口：

```text
myscripts/simulation_rtc_tools/scripts/run_rtc_fast_one_task.sh
```

示例：

```bash
cd /home/hddData/User/lixiang/LingBotVLA2.0_new

bash myscripts/simulation_rtc_tools/scripts/run_rtc_fast_one_task.sh \
  --task click_alarmclock \
  --episodes 1 \
  --model full5k \
  --precision bf16 \
  --request-index 20 \
  --fixed-delay-steps 10 \
  --execution-horizon 20 \
  --crossfade-steps 10 \
  --ema-alpha 0.21
```

也支持把任务作为第一个位置参数：

```bash
bash myscripts/simulation_rtc_tools/scripts/run_rtc_fast_one_task.sh \
  click_alarmclock --model full5k --precision bf16
```

### 5.2 全量任务

入口：

```text
myscripts/simulation_rtc_tools/scripts/run_rtc_fast_50x50.sh
```

```bash
cd /home/hddData/User/lixiang/LingBotVLA2.0_new
unset ROBOTWIN_REALTIME_VISUALIZATION

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

支持参数：

| 参数 | 含义 |
|---|---|
| `--model` | 模型注册名 |
| `--precision` | `bf16`/`fp32` |
| `--episodes` | 每任务episode数 |
| `--request-index` | 当前chunk执行到该索引时异步申请新chunk |
| `--fixed-delay-steps` | 从申请点到固定交接点的动作步数 |
| `--execution-horizon` | 模型RTC对新动作前缀施加约束的范围 |
| `--crossfade-steps` | 交接后进行SmoothStep交叉淡化的步数 |
| `--ema-alpha` | 双臂关节EMA的新动作权重 |
| `--resume-run-dir` | 原Run目录 |
| `--resume-task` | 中断任务 |
| `--initial-episode-count` | 中断任务已有episode数 |
| `--initial-success-count` | 中断任务已有成功数 |
| `--start-seed` | 下一环境seed |
| `--dry-run` | 只解析和打印命令 |

默认RTC配置来自：

```text
myscripts/simulation_rtc_tools/configs/lingbot_action_step2000_rtc_fast.yaml
```

默认值：

```yaml
chunk_horizon: 50
request_index: 10
fixed_delay_steps: 10
execution_horizon: 10
crossfade_steps: 4
ema_alpha: 0.36
request_timeout_s: 120.0
inference_not_ready: pause_simulation
```

注意：当前full5k正式实验使用的是命令行覆盖值：

```text
request=20, delay=10, switch=30, horizon=20, xfade=10, EMA=0.21
```

它们不是YAML默认值。复现实验时必须显式传入。

RTC断点续评：

```bash
RUN=/absolute/path/to/original/rtc/run

bash myscripts/simulation_rtc_tools/scripts/run_rtc_fast_50x50.sh \
  --model full5k \
  --precision bf16 \
  --episodes 50 \
  --request-index 20 \
  --fixed-delay-steps 10 \
  --execution-horizon 20 \
  --crossfade-steps 10 \
  --ema-alpha 0.21 \
  --resume-run-dir "$RUN" \
  --resume-task place_object_stand \
  --initial-episode-count 10 \
  --initial-success-count 2 \
  --start-seed 100011
```

## 6. Conda环境

本地实际相关环境：

```text
lingbotvla  /home/lixiang/miniconda3/envs/lingbotvla
RoboTwin    /home/lixiang/miniconda3/envs/RoboTwin
```

用途：

```text
lingbotvla：LingBot-VLA模型加载、WebSocket推理服务
RoboTwin：SAPIEN仿真、评测客户端、RTC配置解析
```

`point-flow-sim`从 `fd34838` 起已不再是运行依赖。即使本地Conda仍保留旧环境，
当前评测入口也不会使用它。

服务器检查：

```bash
conda env list
conda run -n lingbotvla python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
conda run -n RoboTwin python -c 'import torch, sapien, mplib, curobo, pytorch3d; print("RoboTwin env OK", torch.__version__, torch.version.cuda)'
```

## 7. `runtime_paths.sh`机制

文件：

```text
evaluation/runtime_paths.sh
```

默认流程：

```text
CONDA_EXE（若已设置）
  或 command -v conda
        ↓
conda info --base
        ↓
<base>/etc/profile.d/conda.sh
        ↓
conda run -n RoboTwin / lingbotvla 查询真实Python路径
```

可覆盖变量：

```bash
export LINGBOT_CONDA_SH=/path/to/conda/etc/profile.d/conda.sh
export ROBOTWIN_PYTHON=/path/to/RoboTwin/bin/python
export LINGBOTVLA_PYTHON=/path/to/lingbotvla/bin/python
export LINGBOT_RTC_CONFIG_PY=/optional/config/parser/python
```

通常只需要确保shell中能找到 `conda`。注意：Python路径覆盖只负责预检和RTC配置
解析；底层launcher正式启动时仍会执行：

```bash
conda activate RoboTwin
conda activate lingbotvla
```

因此两个环境必须能用上述名称激活。若服务器只有未注册的prefix环境，应先通过
`conda env list`确认Conda能够识别它们；否则需要把环境加入服务器Conda配置，或
直接使用底层launcher并把 `--sim_env`/`--inference_env` 设为可激活的名称或prefix。

## 8. full5k完整模型结构

应放置为：

```text
models/lingbot_full_5k_6_60/
├── lingbotvla_cli.yaml
├── full_train_parameters.json
├── configuration.json
├── assets/
│   └── norm_stats/
│       └── robotwin_clean.json
├── configs/
│   └── robot_configs_clean/
│       └── robotwin.yaml
└── checkpoints/
    └── global_step_5000/
        └── hf_ckpt/
            ├── config.json
            ├── model.safetensors.index.json
            ├── model-00001-of-00006.safetensors
            ├── model-00002-of-00006.safetensors
            ├── model-00003-of-00006.safetensors
            ├── model-00004-of-00006.safetensors
            ├── model-00005-of-00006.safetensors
            ├── model-00006-of-00006.safetensors
            ├── tokenizer.json
            ├── tokenizer_config.json
            └── 其他HF处理器文件
```

此外，`policy_server.py`还会从项目公共模型目录读取：

```text
models/Qwen3-VL-4B-Instruct/
models/lingbot-vla-v2-6b/depth/model.pt
models/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth
models/lingbot-vla-v2-6b/dino_video/config.yaml
models/depth/moge-2-vitb-normal/model.pt
```

这些辅助资产是否在推理前向中实际走到对应分支，由模型配置决定；当前服务端会
根据训练配置解析并设置这些路径，因此服务器应保持本地相同结构。

## 9. 权重加载调用链

非RTC：

```text
run_eval.sh
  ↓ source evaluation/model_registry.sh
resolve_lingbot_model(full5k)
  ↓
start_robotwin_rtc_infer_and_eval.sh
  ↓ conda activate lingbotvla
python -m evaluation.lingbot_action_step2000.policy_server
  ↓
ExportedCheckpointServer.load_vla()
  ├── 读取 lingbotvla_cli.yaml
  ├── 读取 robotwin_clean.json
  ├── 读取 robotwin.yaml
  ├── 合并Qwen3-VL配置和processor
  ├── 创建 LingBotVlaV2InferencePolicy
  └── strict=True加载hf_ckpt权重
```

RTC：

```text
run_rtc_fast_50x50.sh / run_rtc_fast_one_task.sh
  ↓ source model_registry.sh + runtime_paths.sh
start_robotwin_rtc_infer_and_eval.sh
  ↓
python -m myscripts.simulation_rtc_tools.server.policy_server_rtc
  ↓
复用 ExportedCheckpointServer
  ↓
RTCServerCore处理previous_actions、RTC约束和协议
```

客户端：

```text
非RTC：evaluation/lingbot_action_step2000/copied_eval/
       eval_policy_client_lingbotvla_recording.py

RTC：myscripts/simulation_rtc_tools/copied_eval/
    eval_policy_client_lingbotvla_rtc.py
```

两者都是项目独立复制版，不覆盖Git中的官方同步客户端。launcher运行时会将所需
客户端复制到RoboTwin工作目录的独立文件名。

## 10. 输出目录与记录文件

非RTC：

```text
evaluation/lingbot_action_step2000/results/
└── <model>/<precision>/<clean|randomized>/<run>/
```

RTC：

```text
myscripts/simulation_rtc_tools/results/
└── <model>/<precision>/<clean>/<run>/
```

每个Run目录通常包含：

```text
run_manifest.json
successful_episodes.jsonl
stats.txt
inference_pids.txt
eval_pids.txt
inference_logs/
│   └── qwenpi_slot*_gpu*_port*.log
eval_logs/
│   └── <task>.log
eval_results/
    └── <task>/
        ├── episodes.jsonl
        └── rtc_trace.jsonl       # 仅RTC
```

`episodes.jsonl`记录成功和失败；`successful_episodes.jsonl`只汇总成功episode。
`run_manifest.json`记录模型、精度、RTC参数、代码commit、权重索引指纹及恢复会话。

结果目录被Git忽略。要跨机器断点续评，必须单独复制完整Run目录。

## 11. GPU与端口机制

底层launcher：

```text
myscripts/simulation_rtc_tools/scripts/start_robotwin_rtc_infer_and_eval.sh
```

参数：

```text
--num_gpus N
--num_per_gpu M
--start_port PORT
--output_base PATH
```

slot规则：

```text
num_slots = min(num_tasks, num_gpus × num_per_gpu)
gpu_id    = slot % num_gpus
port      = start_port + slot
```

launcher会为每个slot设置：

```bash
CUDA_VISIBLE_DEVICES=$gpu_id
```

高层 `run_eval.sh`、`run_video_eval.sh`、`run_rtc_fast_one_task.sh`、
`run_rtc_fast_50x50.sh` 和 `run_rtc_video_eval.sh` 默认保持：

```text
1 GPU × 1 server/GPU
物理GPU 0
起始端口9330
```

不指定参数时，原有命令和行为完全不变。需要显式选择单张物理GPU时追加：

```bash
--gpu 8
```

需要任意多卡并行（包括非连续编号）时追加：

```bash
--gpu-ids 2,5,8
```

此时默认每张GPU启动一个推理服务，任务按slot排队分配到GPU 2、5、8。底层参数名为
`--gpu_ids`，高层同时兼容 `--gpu-ids` 与 `--gpu_ids`。正式启动前会通过
`nvidia-smi -L` 检查物理编号，重复、非数字或越界编号会立即报错。所选编号也会写入
`run_manifest.json` 的 `inference.physical_gpu_ids`。

不建议再依赖外层 `CUDA_VISIBLE_DEVICES=8`，因为launcher会为每个slot设置自己的
`CUDA_VISIBLE_DEVICES`。旧任务仍占用端口9330时，新任务即使换GPU也必须避免端口冲突。

在服务器上并行启动前必须检查：

```bash
nvidia-smi
ss -ltnp | grep ':9330\|:9340'
ps -ef | grep -E 'policy_server|start_robotwin' | grep -v grep
```

若需要指定物理GPU、改变端口或多GPU并发，当前应直接调用底层launcher并传完整
参数以改变端口；GPU选择已经可由高层的 `--gpu`/`--gpu-ids` 完成。不要在不清楚
端口映射时同时启动两个使用9330的运行。

## 12. 当前服务器迁移状态

以下状态来自当前服务器迁移对话，未由本机直接验证：

```text
服务器项目：/home/hddData/User/lixiang/LingBotVLA2.0_new

据报告已有：
models/Qwen3-VL-4B-Instruct
models/lingbot_action_step2000
models/lingbot_full_5k_6_60
lingbotvla环境
RoboTwin环境
RoboTwin代码和资产
```

服务器仍需现场确认：

```bash
cd /home/hddData/User/lixiang/LingBotVLA2.0_new

git rev-parse --short HEAD
git status --short

test -f models/lingbot_full_5k_6_60/checkpoints/global_step_5000/hf_ckpt/model.safetensors.index.json
test -f models/lingbot_full_5k_6_60/lingbotvla_cli.yaml
test -f models/lingbot_full_5k_6_60/assets/norm_stats/robotwin_clean.json
test -f models/lingbot_full_5k_6_60/configs/robot_configs_clean/robotwin.yaml
test -d models/Qwen3-VL-4B-Instruct
test -d RoboTwin_lingbot/assets/objects

conda env list
```

如果服务器Conda能够按名称激活环境、但自动Python路径探测失败，可以设置：

```bash
export LINGBOT_CONDA_SH=/usr/local/anaconda3/etc/profile.d/conda.sh
export ROBOTWIN_PYTHON=/home/hddData/User/lixiang/conda-envs/RoboTwin/bin/python
export LINGBOTVLA_PYTHON=/home/hddData/User/lixiang/conda-envs/lingbotvla/bin/python
```

这些变量不替代 `conda activate RoboTwin/lingbotvla`。必须先验证：

```bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate RoboTwin
conda deactivate
conda activate lingbotvla
conda deactivate
```

不需要创建 `point-flow-sim` 环境或兼容软链接。

## 13. 一键复现流程

### 13.1 首次验证

```bash
cd /home/hddData/User/lixiang/LingBotVLA2.0_new

# 只有自动发现Conda失败时才设置这三项。
export LINGBOT_CONDA_SH=/usr/local/anaconda3/etc/profile.d/conda.sh
export ROBOTWIN_PYTHON=/home/hddData/User/lixiang/conda-envs/RoboTwin/bin/python
export LINGBOTVLA_PYTHON=/home/hddData/User/lixiang/conda-envs/lingbotvla/bin/python

bash evaluation/prepare_runtime.sh
bash myscripts/simulation_rtc_tools/scripts/preflight.sh --model full5k

# GPU可见时可额外执行PyTorch3D CUDA算子检查；仍不会启动episode。
bash myscripts/simulation_rtc_tools/scripts/preflight.sh --model full5k --deep

bash myscripts/simulation_rtc_tools/scripts/run_rtc_fast_one_task.sh \
  --task click_alarmclock \
  --episodes 1 \
  --model full5k \
  --precision bf16 \
  --request-index 20 \
  --fixed-delay-steps 10 \
  --execution-horizon 20 \
  --crossfade-steps 10 \
  --ema-alpha 0.21
```

### 13.2 RTC全量评测

确认端口9330空闲且GPU 0可用后：

```bash
cd /home/hddData/User/lixiang/LingBotVLA2.0_new
unset ROBOTWIN_REALTIME_VISUALIZATION

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

### 13.3 非RTC全量评测

RTC和非RTC不要同时占用同一GPU/端口：

```bash
cd /home/hddData/User/lixiang/LingBotVLA2.0_new

bash evaluation/lingbot_action_step2000/run_eval.sh \
  demo_clean \
  --model full5k \
  --precision bf16 \
  --episodes 50
```

### 13.4 只检查命令，不启动episode

```bash
bash evaluation/lingbot_action_step2000/run_eval.sh \
  demo_clean --model full5k --precision bf16 --episodes 50 --dry-run

bash myscripts/simulation_rtc_tools/scripts/run_rtc_fast_50x50.sh \
  --model full5k --precision bf16 --episodes 50 \
  --request-index 20 --fixed-delay-steps 10 \
  --execution-horizon 20 --crossfade-steps 10 --ema-alpha 0.21 \
  --dry-run
```

## 14. 复现边界

代码一致、权重一致、任务配置一致和环境seed一致，是复现评测的必要条件，但单独
重放某个环境seed不保证逐动作完全一致。模型内部采样随机数从推理服务初始seed 42
开始连续推进；改变任务顺序、断点位置、服务数量或先前推理调用次数，都会改变后续
采样状态。

正式比较RTC与非RTC时，应固定：

```text
Git commit
模型及精度
clean/randomized setting
任务顺序
episode数量和环境seed区间
服务数量
RTC参数
torch.compile设置
```

并保留各自的 `run_manifest.json`、`episodes.jsonl`、日志和统计文件。
