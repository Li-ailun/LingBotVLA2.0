# 已验证系统背景

本项目的权威软件环境来自用于full30k FP32多GPU RTC复现的DRLServer。运行`bash scripts/export_verified_environments.sh`后，精确系统、GPU、Python、PyTorch/CUDA和关键包信息写入`snapshots/server-runtime.txt`。

复现保证范围为Linux x86_64、具备足够显存的NVIDIA GPU、Conda、FP32推理以及可用的SAPIEN/Vulkan渲染环境。其他操作系统、ARM、不同驱动和不同CUDA组合可能兼容，但不属于已经验证的环境背景。
