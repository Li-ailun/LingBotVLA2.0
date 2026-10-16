# LingBotVLA2.0 服务器评测完整复现手册

更新时间：2026-10-16。

本文说明如何在已经具备兼容 NVIDIA 驱动、CUDA 和 Conda 的服务器上，从Git仓库和本地大型资产复现 LingBotVLA2.0 的 RoboTwin 单GPU非RTC、RTC、断点续评、逐episode记录和录像功能。

本文只覆盖评测，不覆盖训练和实机数据录制。单GPU非RTC和多GPU RTC已经在现有服务器实际运行；任意GPU编号选择已经支持。每台新服务器仍必须先用单任务验证SAPIEN/Vulkan资源，再逐步提高并发数。

## 1. 最终结论

仅有 `Conda环境 + Git代码 + full5k权重` 还不够。完整复现至少需要：

```text
1. Git仓库代码
2. lingbotvla与RoboTwin两个Conda环境
3. Qwen3-VL-4B-Instruct基础模型
4. full5k模型、配置和归一化统计
5. 深度/未来视频辅助模型文件
6. RoboTwin大型仿真资产
7. 固定commit的CuRobo源码及其可编辑安装
8. 可用的NVIDIA驱动、CUDA和SAPIEN/Vulkan渲染环境
```

上述内容齐全后，服务器无需修改项目代码，只需把大型资产放到规定的相对路径，执行准备和预检脚本，即可使用相同入口评测。

## 2. Git仓库边界

```bash
git clone https://github.com/Li-ailun/LingBotVLA2.0.git
cd LingBotVLA2.0
git log -1 --oneline
git status --short
```

Git包含：

```text
evaluation/                         评测入口、模型注册、路径解析、记录功能
myscripts/simulation_rtc_tools/     RTC服务端、客户端、配置和启动脚本
myscripts/video_tools/              录像处理工具
lingbot-vla-v2/                     LingBotVLA源码（不是submodule）
RoboTwin_lingbot/                   RoboTwin源码（不是submodule）
evaluation/runtime_assets/          小型任务配置和大型资产来源清单
docs/                               复现说明
```

Git不包含：

```text
models/
datasets/
reports/
RoboTwin2_data/
RoboTwin_lingbot/assets/ 下的大型资产
RoboTwin_lingbot/envs/curobo/
任何 results/、eval_results/、日志、PID和视频
Conda环境、CUDA和NVIDIA驱动
```

仓库没有 `.gitmodules`，clone后不需要执行 `git submodule update`。正式复现实验时，应把实际 `git rev-parse HEAD` 写入实验记录，不要依赖本文中的某个历史commit号。

## 3. 必需大型资产

以下路径以本机项目根目录 `/home/lixiang/PROJECT/LingBotVLA2.0` 为基准。服务器可以使用任意项目根目录，但目录内部相对层级必须一致。

### 3.1 Qwen3-VL基础模型（约8.3 GB）

官方来源：`https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct`。

```text
本地：/home/lixiang/PROJECT/LingBotVLA2.0/models/Qwen3-VL-4B-Instruct
目标：<SERVER_ROOT>/models/Qwen3-VL-4B-Instruct
```

### 3.2 full5k模型（约24 GB）

```text
本地：/home/lixiang/PROJECT/LingBotVLA2.0/models/lingbot_full_5k_6_60
目标：<SERVER_ROOT>/models/lingbot_full_5k_6_60
```

必须保留整个目录，至少包含：

```text
models/lingbot_full_5k_6_60/
├── lingbotvla_cli.yaml
├── assets/norm_stats/robotwin_clean.json
├── configs/robot_configs_clean/robotwin.yaml
└── checkpoints/global_step_5000/hf_ckpt/
    ├── config.json
    ├── model.safetensors.index.json
    ├── model-00001-of-00006.safetensors
    ├── ...
    ├── model-00006-of-00006.safetensors
    └── tokenizer及processor文件
```

### 3.3 推理配置引用的辅助模型（约3.1 GB）

full5k训练配置启用了深度、未来深度和未来视频相关配置。当前评测服务会将路径重写到项目 `models/`，服务器应保持：

```text
<SERVER_ROOT>/models/depth/moge-2-vitb-normal/model.pt
<SERVER_ROOT>/models/lingbot-vla-v2-6b/depth/model.pt
<SERVER_ROOT>/models/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth
<SERVER_ROOT>/models/lingbot-vla-v2-6b/dino_video/config.yaml
```

对应本地源：

```text
/home/lixiang/PROJECT/LingBotVLA2.0/models/depth/moge-2-vitb-normal/model.pt
/home/lixiang/PROJECT/LingBotVLA2.0/models/lingbot-vla-v2-6b/depth/model.pt
/home/lixiang/PROJECT/LingBotVLA2.0/models/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth
/home/lixiang/PROJECT/LingBotVLA2.0/models/lingbot-vla-v2-6b/dino_video/config.yaml
```

不需要为了评测复制完整的27 GB `models/lingbot-vla-v2-6b/`。

这些辅助模型均有公开官方来源：

```text
MoGe-2-vitb-normal:
https://huggingface.co/Ruicheng/moge-2-vitb-normal

LingBot-Depth与DINO-VIDEO:
https://huggingface.co/robbyant/lingbot-vla-v2-6b
```

### 3.4 RoboTwin资产（约16.3 GB）

```text
/home/lixiang/PROJECT/LingBotVLA2.0/RoboTwin_lingbot/assets/background_texture  约11 GB
/home/lixiang/PROJECT/LingBotVLA2.0/RoboTwin_lingbot/assets/embodiments         约901 MB
/home/lixiang/PROJECT/LingBotVLA2.0/RoboTwin_lingbot/assets/objects             约4.4 GB
```

服务器目标：

```text
<SERVER_ROOT>/RoboTwin_lingbot/assets/background_texture
<SERVER_ROOT>/RoboTwin_lingbot/assets/embodiments
<SERVER_ROOT>/RoboTwin_lingbot/assets/objects
```

必须形成直接层级，不能出现 `assets/assets/`：

```text
RoboTwin_lingbot/assets/background_texture/
RoboTwin_lingbot/assets/embodiments/aloha-agilex/
RoboTwin_lingbot/assets/objects/objaverse/list.json
```

官方来源：`https://huggingface.co/datasets/TianxingChen/RoboTwin2.0`，固定清单见 `evaluation/runtime_assets/manifest.json`。

### 3.5 CuRobo源码（约241 MB）

```text
本地：/home/lixiang/PROJECT/LingBotVLA2.0/RoboTwin_lingbot/envs/curobo
目标：<SERVER_ROOT>/RoboTwin_lingbot/envs/curobo
仓库：https://github.com/NVlabs/curobo.git
commit：d64c4b005459db10c5dd867d8b30a87d5bda9bdb
```

复制后必须在服务器RoboTwin环境重新执行可编辑安装，不能沿用旧机器 `.pth` 的绝对路径：

```bash
conda activate RoboTwin
python -m pip install -e "$PWD/RoboTwin_lingbot/envs/curobo" --no-build-isolation
```

### 3.6 可选step2000模型（约12 GB）

只复现full5k时不需要：

```text
本地：/home/lixiang/PROJECT/LingBotVLA2.0/models/lingbot_action_step2000
目标：<SERVER_ROOT>/models/lingbot_action_step2000
```

### 3.7 最小传输量

```text
Qwen3-VL       8.3 GB
full5k        24.0 GB
辅助模型       3.1 GB
RoboTwin资产  16.3 GB
CuRobo         0.24 GB
---------------------
合计约        51.9 GB
```

## 4. 官方公开资产下载

除参赛者自行训练或另行取得的`full30k`权重及其配套full5k运行配置外，其余基础模型、
辅助模型、RoboTwin资产和CuRobo均可从官方公开仓库下载。安装Hugging Face命令行：

```bash
python3 -m pip install --upgrade huggingface_hub
hf --help
```

在项目根目录下载Qwen、MoGe以及LingBot辅助模型：

```bash
cd <SERVER_ROOT>

hf download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir models/Qwen3-VL-4B-Instruct

hf download Ruicheng/moge-2-vitb-normal model.pt \
  --local-dir models/depth/moge-2-vitb-normal

hf download robbyant/lingbot-vla-v2-6b \
  depth/model.pt \
  dino_video/teacher_step_10000.pth \
  dino_video/config.yaml \
  --local-dir models/lingbot-vla-v2-6b
```

只下载RoboTwin运行所需的三个官方资产压缩包并解压：

```bash
cd <SERVER_ROOT>/RoboTwin_lingbot/assets

hf download TianxingChen/RoboTwin2.0 \
  background_texture.zip embodiments.zip objects.zip \
  --repo-type dataset \
  --local-dir .

unzip -q background_texture.zip
unzip -q embodiments.zip
unzip -q objects.zip
```

CuRobo必须使用本项目固定的commit，不能直接使用当前main：

```bash
cd <SERVER_ROOT>
mkdir -p RoboTwin_lingbot/envs
git clone https://github.com/NVlabs/curobo.git \
  RoboTwin_lingbot/envs/curobo
git -C RoboTwin_lingbot/envs/curobo checkout --detach \
  d64c4b005459db10c5dd867d8b30a87d5bda9bdb
```

下载完成后不要凭目录大小判断完整性，统一运行第20.2节的验证脚本。
Codex可以代为执行这些命令，但必须保留仓库ID、相对目标路径与CuRobo固定commit。

## 5. 已有机器之间的推荐传输命令

在本机项目根目录执行，并替换服务器信息：

```bash
cd /home/lixiang/PROJECT/LingBotVLA2.0
SERVER="用户名@服务器地址"
REMOTE_ROOT="/home/hddData/User/lixiang/LingBotVLA2.0"

rsync -aH --info=progress2 models/Qwen3-VL-4B-Instruct \
  "$SERVER:$REMOTE_ROOT/models/"

rsync -aH --info=progress2 models/lingbot_full_5k_6_60 \
  "$SERVER:$REMOTE_ROOT/models/"

ssh "$SERVER" "mkdir -p \
  '$REMOTE_ROOT/models/depth/moge-2-vitb-normal' \
  '$REMOTE_ROOT/models/lingbot-vla-v2-6b/depth' \
  '$REMOTE_ROOT/models/lingbot-vla-v2-6b/dino_video'"

rsync -aH --info=progress2 models/depth/moge-2-vitb-normal/model.pt \
  "$SERVER:$REMOTE_ROOT/models/depth/moge-2-vitb-normal/"

rsync -aH --info=progress2 models/lingbot-vla-v2-6b/depth/model.pt \
  "$SERVER:$REMOTE_ROOT/models/lingbot-vla-v2-6b/depth/"

rsync -aH --info=progress2 \
  models/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth \
  models/lingbot-vla-v2-6b/dino_video/config.yaml \
  "$SERVER:$REMOTE_ROOT/models/lingbot-vla-v2-6b/dino_video/"

rsync -aH --info=progress2 \
  RoboTwin_lingbot/assets/background_texture \
  RoboTwin_lingbot/assets/embodiments \
  RoboTwin_lingbot/assets/objects \
  "$SERVER:$REMOTE_ROOT/RoboTwin_lingbot/assets/"

rsync -aH --info=progress2 RoboTwin_lingbot/envs/curobo \
  "$SERVER:$REMOTE_ROOT/RoboTwin_lingbot/envs/"
```

若服务器已有共享RoboTwin资产，可不重复复制：

```bash
bash evaluation/prepare_runtime.sh \
  --assets-source /absolute/path/to/shared/assets
```

该路径必须直接包含 `objects/`、`embodiments/` 和 `background_texture/`。

## 6. Conda与系统环境

只需要两个Conda环境：

```text
lingbotvla：模型加载和WebSocket推理服务
RoboTwin：SAPIEN仿真、评测客户端、CuRobo、PyTorch3D和RTC配置解析
```

`point-flow-sim`不再是依赖。环境必须能按名称激活：

```bash
conda activate lingbotvla
conda deactivate
conda activate RoboTwin
conda deactivate
```

基本检查：

```bash
conda run -n lingbotvla python -c \
  'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'

conda run -n RoboTwin python -c \
  'import torch, sapien, mplib, curobo, pytorch3d; print(torch.__version__, torch.version.cuda)'
```

若Conda不能自动发现，可设置：

```bash
export LINGBOT_CONDA_SH=/path/to/conda/etc/profile.d/conda.sh
export ROBOTWIN_PYTHON=/path/to/RoboTwin/bin/python
export LINGBOTVLA_PYTHON=/path/to/lingbotvla/bin/python
```

项目通过 `evaluation/runtime_paths.sh` 解析运行路径，不要求服务器复刻本机 `/home/lixiang/...` 根路径。

服务器上已经验证可用的环境设置形式如下。路径必须替换为服务器上的真实位置：

```bash
unset LD_LIBRARY_PATH

export CONDA_ENVS_PATH=/path/to/conda-envs
export LINGBOT_CONDA_SH=/path/to/conda/etc/profile.d/conda.sh
export ROBOTWIN_PYTHON=/path/to/conda-envs/RoboTwin/bin/python
export LINGBOTVLA_PYTHON=/path/to/conda-envs/lingbotvla/bin/python
```

建议在安装CuRobo、运行预检和启动评测之前执行这些设置。`unset LD_LIBRARY_PATH`只对当前Shell及其子进程生效，不会修改系统CUDA安装。

`nvidia-smi`显示的CUDA版本是驱动支持的最高CUDA API版本，不等于当前PyTorch wheel使用的CUDA运行时。例如，驱动显示CUDA 13.0而PyTorch为`2.4.1+cu121`是允许的；关键是不要让系统CUDA 12.0动态库覆盖PyTorch/Conda需要的CUDA 12.1动态库。

遇到 `libcusparse.so.12`、`libnvJitLink.so.12`或`__nvJitLinkAddData_12_1`符号错误时，先检查：

```bash
echo "${LD_LIBRARY_PATH-}"
"$ROBOTWIN_PYTHON" -c \
  'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

若`LD_LIBRARY_PATH`包含与PyTorch不匹配的系统CUDA目录，先执行`unset LD_LIBRARY_PATH`，不要立即重装PyTorch、CUDA或整个Conda环境。

CuRobo源码复制到新项目目录后，需要在RoboTwin环境中重新注册editable安装：

```bash
"$ROBOTWIN_PYTHON" -m pip install -e \
  "$PWD/RoboTwin_lingbot/envs/curobo" \
  --no-build-isolation
```

这会更新Conda环境中的包注册位置，不会修改模型权重或评测算法。首次运行时，启动器可能提示：

```text
Curobo yml paths stale -> regenerating for <SERVER_ROOT>/RoboTwin_lingbot
```

这是将CuRobo模板中的`${ASSETS_PATH}`展开为当前服务器路径，属于正常的部署路径适配。

## 7. 首次部署与预检

```bash
cd <SERVER_ROOT>

# 恢复Git跟踪的小型任务配置，并验证资产和CuRobo。
bash evaluation/prepare_runtime.sh

# 普通预检，不启动episode。
bash myscripts/simulation_rtc_tools/scripts/preflight.sh --model full5k

# GPU可用时执行CUDA/PyTorch3D深度预检。
bash myscripts/simulation_rtc_tools/scripts/preflight.sh --model full5k --deep
```

期望出现：

```text
TASK CONFIG VALID
ASSET LAYOUT VALID
CUROBO SOURCE VALID
MODEL ADAPTER VALID
RUNTIME IMPORTS VALID
RTC CONFIG VALID
RTC PREFLIGHT OK
```

预检成功只说明静态文件、依赖导入和RTC配置满足启动条件；它不会执行仿真episode。完成预检后仍需运行第16节的非RTC和RTC单任务冒烟测试。

关键文件检查：

```bash
test -f models/lingbot_full_5k_6_60/checkpoints/global_step_5000/hf_ckpt/model.safetensors.index.json
test -f models/lingbot_full_5k_6_60/lingbotvla_cli.yaml
test -f models/lingbot_full_5k_6_60/assets/norm_stats/robotwin_clean.json
test -f models/lingbot_full_5k_6_60/configs/robot_configs_clean/robotwin.yaml
test -d models/Qwen3-VL-4B-Instruct
test -f models/depth/moge-2-vitb-normal/model.pt
test -f models/lingbot-vla-v2-6b/depth/model.pt
test -f models/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth
test -f RoboTwin_lingbot/assets/objects/objaverse/list.json
```

## 8. 模型、精度与GPU

模型注册位于 `evaluation/model_registry.sh`，支持：

```text
--model step2000 | full5k | full15k | full30k
--precision bf16 | fp32
```

BF16只改变加载和推理精度，不修改磁盘FP32权重。full5k FP32通常需要显存更大的GPU，RTX 3090不保证能装下。

默认使用物理GPU 0。指定单卡：

```text
--gpu 8
```

指定任意非连续GPU列表：

```text
--gpu-ids 2,5,8
```

GPU列表调度已实现，但多路SAPIEN仿真并发仍需在目标服务器验证渲染资源稳定性。

## 9. 非RTC评测

单任务冒烟测试：

```bash
cd <SERVER_ROOT>
unset ROBOTWIN_REALTIME_VISUALIZATION

bash evaluation/lingbot_action_step2000/run_eval.sh \
  demo_clean \
  --model full5k \
  --precision bf16 \
  --task click_alarmclock \
  --episodes 1
```

全量50×50：

```bash
bash evaluation/lingbot_action_step2000/run_eval.sh \
  demo_clean \
  --model full5k \
  --precision bf16 \
  --episodes 50
```

指定GPU只需在命令末尾追加 `--gpu 8`。

### 9.1 比赛语言指令设置

赛事评测入口默认对 `demo_clean` 和 `demo_randomized` 都使用
`instruction_type=unseen`。`seen/unseen` 只选择发送给模型的语言描述集合，
不会改变场景、seed、物体、相机、step limit或成功判定。正式提交时建议在命令中
显式写出 `--instruction-type unseen`，便于日志审计；不写时默认值同样是
`unseen`。研究语言指令影响时可显式使用 `--instruction-type seen`，但该结果应与
正式 unseen 结果分开标记。

天池正式统计要求每个任务100次；本文中的50次命令用于阶段性评测，正式提交时将
`--episodes 50` 改为 `--episodes 100`。

## 10. RTC评测

当前full5k正式参数：

```text
request_index=20
fixed_delay_steps=10
固定切换点=30
execution_horizon=20
crossfade_steps=10
ema_alpha=0.21
```

单任务冒烟测试：

```bash
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

全量50×50：

```bash
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

需要时追加 `--gpu 8` 或 `--gpu-ids 2,5,8`。

全量RTC入口默认使用 `demo_clean` 和起始端口9330。评估 randomized 时使用
`--setting demo_randomized`；如果另一轮评测仍占用9330开始的端口，必须用
`--base-port` 选择不冲突的端口。例如，在GPU 3、5上从9340开始运行：

```bash
bash myscripts/simulation_rtc_tools/scripts/run_rtc_fast_50x50.sh \
  --setting demo_randomized \
  --base-port 9340 \
  --model full5k \
  --precision bf16 \
  --episodes 50 \
  --gpu-ids 3,5 \
  --request-index 20 \
  --fixed-delay-steps 10 \
  --execution-horizon 20 \
  --crossfade-steps 10 \
  --ema-alpha 0.21
```

该命令的结果进入
`myscripts/simulation_rtc_tools/results/full5k/bf16/randomized/<run>/`。
GPU显存是否足够仍应以启动前的 `nvidia-smi` 为准，不能仅根据利用率选择GPU。

## 11. 断点续评

每个Run目录中的 `eval_results/<task>/episodes.jsonl` 记录成功和失败episode。下一个seed应取最后记录的 `environment_seed + 1`，不能简单按episode数量推算，因为无效场景seed可能被跳过。

非RTC：

```bash
RUN=/absolute/path/to/non_rtc/run

bash evaluation/lingbot_action_step2000/run_eval.sh \
  demo_clean --model full5k --precision bf16 --episodes 50 \
  --resume-run-dir "$RUN" \
  --resume-task place_object_stand \
  --initial-episode-count 10 \
  --initial-success-count 2 \
  --start-seed 100011
```

RTC：

```bash
RUN=/absolute/path/to/rtc/run

bash myscripts/simulation_rtc_tools/scripts/run_rtc_fast_50x50.sh \
  --model full5k --precision bf16 --episodes 50 \
  --request-index 20 --fixed-delay-steps 10 \
  --execution-horizon 20 --crossfade-steps 10 --ema-alpha 0.21 \
  --resume-run-dir "$RUN" \
  --resume-task place_object_stand \
  --initial-episode-count 10 \
  --initial-success-count 2 \
  --start-seed 100011
```

跨机器继续旧实验必须额外复制完整Run目录；只复制代码和模型无法恢复旧日志。

## 12. 输出与可追溯记录

```text
非RTC：evaluation/lingbot_action_step2000/results/<model>/<precision>/<setting>/<run>/
RTC：myscripts/simulation_rtc_tools/results/<model>/<precision>/<setting>/<run>/
```

Run目录通常包含：

```text
run_manifest.json                     模式、模型、精度、commit、GPU和RTC参数
successful_episodes.jsonl             成功episode汇总
stats.txt                             最终统计
inference_logs/                       推理日志
eval_logs/<task>.log                  任务进度与成功率
eval_results/<task>/episodes.jsonl    成功和失败episode
eval_results/<task>/rtc_trace.jsonl   仅RTC
```

结果目录被Git忽略，需要自行归档。

## 13. 录像入口

非RTC：

```bash
bash evaluation/lingbot_action_step2000/run_video_eval.sh \
  demo_clean --model full5k --precision bf16 \
  --task click_alarmclock --episodes 1
```

RTC：

```bash
bash myscripts/simulation_rtc_tools/scripts/run_rtc_video_eval.sh \
  --model full5k --precision bf16 \
  --setting demo_clean --task click_alarmclock --episodes 1 \
  --request-index 20 --fixed-delay-steps 10 \
  --execution-horizon 20 --crossfade-steps 10 --ema-alpha 0.21
```

## 14. 常见故障

### `No module named evaluation`

非RTC推理服务必须从项目根目录启动。当前入口已使用 `--inference_workdir <SERVER_ROOT>`，不要改回 `<SERVER_ROOT>/lingbot-vla-v2`。

### 缺少`task_config/demo_clean.yml`

```bash
bash evaluation/prepare_runtime.sh
```

### `No module named curobo`

```bash
"$ROBOTWIN_PYTHON" -m pip install -e \
  "$PWD/RoboTwin_lingbot/envs/curobo" \
  --no-build-isolation
```

### 找不到`objaverse/list.json`

正确位置是 `RoboTwin_lingbot/assets/objects/objaverse/list.json`，不能出现 `assets/assets/`。

### `libcusparse`或`nvJitLink`符号错误

检查PyTorch CUDA与 `LD_LIBRARY_PATH`。如果系统CUDA目录覆盖了Conda/PyTorch运行库，执行`unset LD_LIBRARY_PATH`后重新验证导入。不要仅根据`nvidia-smi`顶部显示的CUDA版本重装环境。

### preflight中的SHA256静态校验失败

先确认工作区文件和当前Git提交中的文件是否一致。以`policy_server.py`为例：

```bash
sha256sum evaluation/lingbot_action_step2000/policy_server.py
git show HEAD:evaluation/lingbot_action_step2000/policy_server.py | sha256sum
git status --short -- evaluation/lingbot_action_step2000/policy_server.py
```

- 两个哈希不同：工作区文件被改动，先查明改动来源，不要盲目更新预检哈希。
- 两个哈希一致，但预检仍报告旧哈希：当前Git版本的预检常量可能陈旧，应在主仓库修正`preflight.sh`并提交，然后由服务器`git pull`。
- 不要为了迎合旧哈希直接执行`git checkout -- policy_server.py`，否则可能回退当前评测适配代码。

当前仓库的静态校验值应随对应Git提交一起使用；不要在服务器上维护私有哈希补丁。

### `RuntimeError: cannot create buffer`

通常与多个SAPIEN实例并发创建相机/渲染资源有关。先用单GPU、单slot确认环境，再增加并发。

### WebSocket连接失败

```bash
ss -ltnp | grep 9330
ps -ef | grep -E 'policy_server|start_robotwin' | grep -v grep
tail -n 200 <RUN>/inference_logs/*.log
```

## 15. 不需要迁移的内容

全新评测不需要：

```text
datasets/
reports/
evaluation/**/results/
myscripts/simulation_rtc_tools/results/
evaluation/video_comparison/
*.log
*.pid
*.mp4
__pycache__/
```

只有断点续跑旧实验时，才复制对应的完整Run目录。

## 16. 最短复现流程

```bash
# 1. 克隆代码
git clone https://github.com/Li-ailun/LingBotVLA2.0.git
cd LingBotVLA2.0

# 2. 按第3、4节放置约51.9GB大型资产

# 3. 解析服务器运行环境；以下路径按服务器实际位置修改
unset LD_LIBRARY_PATH
export CONDA_ENVS_PATH=/path/to/conda-envs
export LINGBOT_CONDA_SH=/path/to/conda/etc/profile.d/conda.sh
export ROBOTWIN_PYTHON=/path/to/conda-envs/RoboTwin/bin/python
export LINGBOTVLA_PYTHON=/path/to/conda-envs/lingbotvla/bin/python

# 4. 重新注册CuRobo editable源码
"$ROBOTWIN_PYTHON" -m pip install -e \
  "$PWD/RoboTwin_lingbot/envs/curobo" \
  --no-build-isolation

# 5. 恢复任务配置并预检
bash evaluation/prepare_runtime.sh
bash myscripts/simulation_rtc_tools/scripts/preflight.sh --model full5k --deep

# 6. 单任务非RTC冒烟测试
bash evaluation/lingbot_action_step2000/run_eval.sh \
  demo_clean --model full5k --precision bf16 \
  --task click_alarmclock --episodes 1 --gpu 0

# 7. 单任务RTC冒烟测试
bash myscripts/simulation_rtc_tools/scripts/run_rtc_fast_one_task.sh \
  --task click_alarmclock --episodes 1 \
  --model full5k --precision bf16 --gpu 0 \
  --request-index 20 --fixed-delay-steps 10 \
  --execution-horizon 20 --crossfade-steps 10 --ema-alpha 0.21
```

将`--gpu 0`替换为实际要使用的物理GPU编号。单任务测试必须显式提供`--task click_alarmclock`；只指定`--episodes 1`并不会自动把任务数量限制为1。

非RTC冒烟测试至少需要同时出现：

```text
[done] click_alarmclock ... (progress 1/1)
Result: 1 done, 0 skipped
All tasks fully executed 1 episodes
All inference servers stopped
Cleanup done
```

成功率可以是`1/1`或`0/1`：这里验证的是完整评测链路，而不是要求模型必须完成任务。RTC冒烟测试还应生成对应任务的`rtc_trace.jsonl`并包含`request`、`wait`或`promotion`等RTC事件。

两个冒烟测试都完成episode并生成 `run_manifest.json`、`episodes.jsonl` 和任务日志后，再开始50×50全量评测。

## 17. 复现边界

代码、权重、配置和环境seed一致是必要条件，但单独重放某个环境seed不保证逐动作完全一致。推理模型随机数从服务初始seed 42开始连续推进；任务顺序、断点位置、服务数量及先前推理调用次数都会改变后续随机状态。

正式比较RTC与非RTC时必须固定：

```text
Git commit
模型与精度
clean/randomized设置
任务顺序
episode数量和环境seed区间
推理服务数量
RTC参数
torch.compile设置
```

同时保留 `run_manifest.json`、`episodes.jsonl`、推理日志和最终统计文件。



总结：
可以快速复现，但需要区分“代码快速复现”和“完整环境一键复现”。

目前已经实现：

```text
Git拉取代码
+ 复制约51.9GB外部资产
+ 准备两个Conda环境
+ 注册CuRobo
+ 执行preflight
+ 单任务冒烟测试
= 完整评测复现
```

已经在新服务器实际验证通过：

- full5k权重加载
- Qwen3-VL加载
- RoboTwin、SAPIEN、CuRobo、PyTorch3D
- BF16单GPU非RTC
- BF16三GPU RTC并行
- GPU编号映射
- 日志、成功率与RTC trace

但当前还不是“新机器执行一个命令就自动安装所有依赖”。复现者仍需：

1. 建立兼容的`lingbotvla`和`RoboTwin`环境。
2. 复制模型、RoboTwin assets和CuRobo源码。
3. 避免系统`LD_LIBRARY_PATH`污染PyTorch CUDA运行库。
4. 运行CuRobo editable安装。
5. 执行preflight和冒烟测试。

按照当前指南，在已经有两个兼容Conda环境的服务器上，代码和运行链路可以很快恢复；主要耗时来自约51.9GB资产传输，而不是代码适配。

最短流程是：

```bash
git clone https://github.com/Li-ailun/LingBotVLA2.0.git
cd LingBotVLA2.0

# 将指南列出的模型及资产放入指定相对路径

unset LD_LIBRARY_PATH
export CONDA_ENVS_PATH=/path/to/conda-envs
export LINGBOT_CONDA_SH=/path/to/conda/etc/profile.d/conda.sh
export ROBOTWIN_PYTHON=/path/to/conda-envs/RoboTwin/bin/python
export LINGBOTVLA_PYTHON=/path/to/conda-envs/lingbotvla/bin/python

"$ROBOTWIN_PYTHON" -m pip install -e \
  "$PWD/RoboTwin_lingbot/envs/curobo" \
  --no-build-isolation

bash evaluation/prepare_runtime.sh
bash myscripts/simulation_rtc_tools/scripts/preflight.sh \
  --model full5k \
  --deep
```

然后单任务验证：

```bash
bash evaluation/lingbot_action_step2000/run_eval.sh \
  demo_clean \
  --model full5k \
  --precision bf16 \
  --task click_alarmclock \
  --episodes 1 \
  --gpu 0
```

因此准确结论是：

> 如果Conda环境和大型资产已经准备好，当前仓库可以快速、无需修改代码地复现；如果面对完全空白的新服务器，仍需安装两个Conda环境和CUDA相关编译依赖，尚未达到完全一键部署。

## 18. full30k权重评测

最新继续训练权重注册名为`full30k`，并兼容`30k`和`lingbot_30K`别名。权重放在：

```text
models/lingbot_30K/
```

注册器自动支持以下两种布局：

```text
models/lingbot_30K/config.json
models/lingbot_30K/model.safetensors.index.json
```

或：

```text
models/lingbot_30K/checkpoints/global_step_30000/hf_ckpt/config.json
models/lingbot_30K/checkpoints/global_step_30000/hf_ckpt/model.safetensors.index.json
```

full30k与full15k一样复用`models/lingbot_full_5k_6_60/`中的归一化统计、
机器人配置和训练配置。权重下载未完成、关键文件缺失或目录中仍有
`.incomplete`分片时，评测入口会直接报错，不会加载半成品权重。

下载完成后的预检：

```bash
bash myscripts/simulation_rtc_tools/scripts/preflight.sh \
  --model full30k \
  --deep
```

本地单卡非RTC冒烟测试（历史BF16示例）：

```bash
bash evaluation/lingbot_action_step2000/run_eval.sh \
  demo_clean \
  --model full30k \
  --precision bf16 \
  --task click_alarmclock \
  --episodes 1 \
  --gpu 0 \
  --instruction-type unseen
```

服务器任意三卡RTC全量评测示例（当前标准为FP32、Clean-unseen）：

```bash
bash myscripts/simulation_rtc_tools/scripts/run_rtc_fast_50x50.sh \
  --setting demo_clean \
  --model full30k \
  --precision fp32 \
  --episodes 50 \
  --gpu-ids 1,3,4 \
  --instruction-type unseen \
  --request-index 20 \
  --fixed-delay-steps 10 \
  --execution-horizon 20 \
  --crossfade-steps 10 \
  --ema-alpha 0.10
```

本地与服务器使用同一套代码；单卡和多卡的差异仅是`--gpu`与
`--gpu-ids`参数。结果分别写入`results/full30k/<precision>/<setting>/`，
不会与full5k或full15k结果混在一起。

## 19. 服务器多卡自动断点续跑

RTC全量评测中断后，推荐使用`--resume-all`自动恢复整个运行目录。它会读取：

```text
<RUN>/eval_results/<task>/episodes.jsonl
```

并为每个任务独立恢复已完成数、成功数和下一个环境seed。完成目标episode数的任务
不会重新运行，未开始和未完成的任务按官方任务顺序进入多卡队列。

示例：使用GPU 1、3、4恢复full30k FP32 RTC评测：

```bash
cd <SERVER_ROOT>
unset LD_LIBRARY_PATH
unset ROBOTWIN_REALTIME_VISUALIZATION

export CONDA_ENVS_PATH=/path/to/conda-envs
export ROBOTWIN_PYTHON=/path/to/conda-envs/RoboTwin/bin/python
export LINGBOTVLA_PYTHON=/path/to/conda-envs/lingbotvla/bin/python
export RUN=/绝对路径/到/原运行目录

bash myscripts/simulation_rtc_tools/scripts/run_rtc_fast_50x50.sh \
  --setting demo_clean \
  --model full30k \
  --precision fp32 \
  --episodes 50 \
  --gpu-ids 1,3,4 \
  --instruction-type unseen \
  --resume-run-dir "$RUN" \
  --resume-all \
  --request-index 20 \
  --fixed-delay-steps 10 \
  --execution-horizon 20 \
  --crossfade-steps 10 \
  --ema-alpha 0.10
```

启动前会把当前参数与`run_manifest.json`比较。模型、精度、RTC/non-RTC模式、
clean/randomized、seen/unseen、episode目标或RTC参数不一致时直接停止，避免污染
已有结果。每个任务的`episode_index`还必须从0连续且不重复。

旧实验如果日志中已有成功率，但缺少`episodes.jsonl`，自动模式不会根据日志猜测
断点；此时继续使用原来的单任务手动恢复参数。`--resume-all`不能与以下参数混用：

```text
--resume-task
--initial-episode-count
--initial-success-count
--start-seed
```

多个worker写入成功episode汇总时由provenance文件锁保护。任务进程异常重试前会
重新读取该任务的最新`episodes.jsonl`，因此已经在失败前落盘的episode不会被当成
未完成进度重新计算。

## 20. full30k FP32 RTC Clean-unseen标准复现

当前标准目标固定为：

```text
model=full30k
precision=fp32
mode=RTC
setting=demo_clean
instruction_type=unseen
episodes=50/task
request_index=20
fixed_delay_steps=10
execution_horizon=20
crossfade_steps=10
ema_alpha=0.10
```

### 20.1 环境来源与创建

权威环境来自已经运行该链路的DRLServer，而不是本地工作站。先在DRLServer导出：

```bash
cd <SERVER_ROOT>
bash scripts/export_verified_environments.sh
```

检查并提交生成的纯文本`environment/*.environment.yml`和
`environment/snapshots/*`后，新服务器执行：

```bash
bash scripts/create_eval_environments.sh \
  --conda /path/to/conda \
  --lingbot-prefix /path/to/conda-envs/lingbotvla \
  --robotwin-prefix /path/to/conda-envs/RoboTwin
```

完整外部资产清单位于`environment/assets-full30k.json`。模型和仿真资产不进入Git，
必须放在清单规定的项目相对路径；full30k继续复用full5k的训练配置、机器人配置和
归一化统计。

### 20.2 只读预检

FP32需要显著高于BF16的显存。以下`55000 MiB`是保守启动门槛，可按实测峰值调整；
不能继续使用BF16评测时期的`21000 MiB`检查值。

```bash
cd <SERVER_ROOT>
unset LD_LIBRARY_PATH
export LINGBOTVLA_PYTHON=/path/to/conda-envs/lingbotvla/bin/python
export ROBOTWIN_PYTHON=/path/to/conda-envs/RoboTwin/bin/python

bash scripts/verify_full30k_rtc_environment.sh \
  --gpu-ids 1,2,3,5 \
  --base-port 9380 \
  --min-free-mib 55000
```

### 20.3 正式运行

先审计最终命令，不启动服务：

```bash
bash scripts/run_full30k_rtc_clean_unseen.sh \
  --gpu-ids 1,2,3,5 \
  --base-port 9380 \
  --episodes 50 \
  --min-free-mib 55000 \
  --dry-run
```

确认输出后去掉`--dry-run`正式运行。GPU编号和端口只是示例，复现者可以选择任意
满足显存条件且未占用的GPU和端口；模型、FP32、Clean-unseen及RTC参数由包装入口
固定，无需修改任何项目代码。

录像仍使用现有入口，并显式保持同一参数：

```bash
bash myscripts/simulation_rtc_tools/scripts/run_rtc_video_eval.sh \
  --setting demo_clean --model full30k --precision fp32 \
  --task click_alarmclock --episodes 1 --gpu 0 \
  --instruction-type unseen \
  --request-index 20 --fixed-delay-steps 10 \
  --execution-horizon 20 --crossfade-steps 10 --ema-alpha 0.10
```

中断后的多GPU自动续跑仍使用第19节的`--resume-run-dir "$RUN" --resume-all`，但
必须保持`--precision fp32`、`demo_clean`、`unseen`和全部RTC参数一致，否则Manifest
一致性检查会拒绝写入原结果目录。
