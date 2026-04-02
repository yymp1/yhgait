# GPU / NCCL 可行性报告

## 机器现场

- GPU：`NVIDIA GeForce RTX 2070 SUPER`
- 驱动：`470.256.02`
- `nvidia-smi` 显示 CUDA：`11.4`
- 本机 `nvcc`：`/usr/local/cuda-11.4/bin/nvcc`
- 本机 toolkit：`CUDA 11.4`

本地检查材料：

- `reports/linux_first_round/linux_gpu_compat_local.json`
- `reports/linux_second_round/linux_gpu_compat_gpu_env.json`

## 兼容性判断

### 建议尝试的组合

- `Python 3.10`
- `torch==2.5.1`
- `torchvision==0.20.1`
- `cu118` 官方轮子

选择理由：

- 当前机器驱动在 `470` 线，继续往 `cu121+` 走风险更高
- `cu118` 是 PyTorch 官方仍提供的历史稳定组合
- 本机实测这套环境可以做到：
  - `torch.cuda.is_available() = True`
  - `torch.version.cuda = 11.8`
  - `torch.distributed.is_nccl_available() = True`
  - OpenGait 官方入口 smoke 可通过

### 不建议优先尝试的组合

- `CPU-only torch`
  - 原因：会再次触发 `Distributed package doesn't have NCCL built in`
- `cu121 / cu124 / cu126 / cu128`
  - 原因：这些组合与当前 `470` 驱动相比风险更高，没有必要在第二轮先走高风险路线
- 继续在当前 CPU 环境上追官方入口
  - 原因：这条路第一轮已经证伪

## 独立 GPU 环境实测

环境：

- `/home/bb/gait/yhgait/.venvs/opengait-gpu-cu118`

核心版本：

- `torch 2.5.1+cu118`
- `torchvision 0.20.1+cu118`

实测结果：

- `torch.cuda.is_available() = True`
- `torch.cuda.device_count() = 1`
- `torch.distributed.is_nccl_available() = True`
- `device_name = NVIDIA GeForce RTX 2070 SUPER`

在这套环境里，官方入口辅助 smoke 已通过：

- `reports/linux_second_round/opengait_smoke_test.md`

## 当前阻塞点

GPU/NCCL 不是当前主阻塞点。

当前真正阻塞正式推进的是：

- Linux 机器上没有真实 `CASIA-B-pkl`

## 建议

1. 保留当前 GPU 环境，不要删除
2. 优先把真实 `CASIA-B-pkl` 同步到 Linux
3. 用同一套 GPU 环境重跑真实数据 smoke
4. 如果真实数据 smoke 通过，再进入更正式的训练配置
