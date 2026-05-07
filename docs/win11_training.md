# Win11 单卡训练工作流

这份文档默认你已经把项目迁到了 `Win11 + NVIDIA GPU` 电脑上，并且已经完成基础环境部署。

如果你还没完成环境部署，先看：

- `docs/win11_deployment.md`

当前仓库现在把 Win11 视为主平台，训练和评估统一走仓库自己的包装入口：

- `scripts/run_opengait_main.py`

这个入口的定位是：

1. 统一使用单机单卡
2. 在 Win11 上自动避开 `NCCL`
3. 让训练 / 评估 / smoke / probe / gallery 构建都走同一套运行时

## 当前建议的 Win11 运行范围

Win11 主线现在推荐做这些事：

- OpenGait smoke test
- 单卡 probe / short-run / baseline
- `formal_conservative_real` 单卡训练
- `formal_conservative_real` 单卡评估
- 构建 gallery cache
- 启动私有步态库 GUI

当前不再把 Linux 当成默认主线。  
但仍要说明一点：当前仓库的 Win11 训练目标是**单卡稳定运行**，不是多卡扩展。

## 先准备什么

### 1. 环境

先完成：

```powershell
.\scripts\start_win11_demo.ps1 -SkipLaunch
```

这一步会把训练需要的依赖也装好。

### 2. 训练数据

如果你要跑训练或正式评估，必须有：

```text
datasets/processed/CASIA-B-pkl/
```

### 3. OpenGait 代码

必须有：

```text
external/OpenGait/
```

## 最推荐的训练入口

以后在 Win11 上，优先用：

```powershell
.\scripts\start_win11_training.ps1
```

这个脚本会先跑环境检查，再根据你选择的模式执行不同动作。

## 最常用的模式

### 1. smoke

先确认训练链路是不是最小可用：

```powershell
.\scripts\start_win11_training.ps1 -Mode smoke
```

### 2. probe

跑一个非常短的小探针：

```powershell
.\scripts\start_win11_training.ps1 -Mode probe
```

### 3. short-run

比 probe 稍长一点：

```powershell
.\scripts\start_win11_training.ps1 -Mode short-run
```

### 4. baseline

跑 `baseline_small_real`：

```powershell
.\scripts\start_win11_training.ps1 -Mode baseline
```

### 5. formal-train

启动 `formal_conservative_real`：

```powershell
.\scripts\start_win11_training.ps1 -Mode formal-train
```

默认会按当前项目状态尝试从 `1000` iter 继续。

如果你想显式指定恢复迭代：

```powershell
.\scripts\start_win11_training.ps1 -Mode formal-train -ResumeIter 1000
```

如果你想从头开始：

```powershell
.\scripts\start_win11_training.ps1 -Mode formal-train -ResumeIter 0
```

### 6. formal-eval

对某个 checkpoint 做正式评估：

```powershell
.\scripts\start_win11_training.ps1 -Mode formal-eval -CheckpointIter 1000
```

### 7. build-gallery

构建本地检索 demo 的 gallery 缓存：

```powershell
.\scripts\start_win11_training.ps1 -Mode build-gallery -CheckpointIter 1000
```

## 如果你想手工执行底层命令

### 训练

```powershell
.\.venvs\win11-demo\Scripts\python.exe .\scripts\run_opengait_main.py `
  --cfg-path .\configs\opengait_casiab_formal_conservative.yaml `
  --phase train `
  --opengait-root .\external\OpenGait `
  --master-port 29531 `
  --num-workers 0 `
  --log-to-file
```

### 评估

```powershell
.\.venvs\win11-demo\Scripts\python.exe .\scripts\run_opengait_main.py `
  --cfg-path .\configs\opengait_casiab_formal_conservative_eval.yaml `
  --phase test `
  --opengait-root .\external\OpenGait `
  --master-port 29531 `
  --iter 1000 `
  --num-workers 0 `
  --log-to-file
```

## num_workers 建议

Win11 上第一次跑时，建议先用：

- `0`
- 或 `1`

等确认稳定之后，再考虑调大。

## 训练输出去哪看

OpenGait 的训练产物仍然会写到：

```text
external/OpenGait/output/CASIA-B/GaitSet/<save_name>/
```

你最常看的通常是：

- `checkpoints/`
- `logs/`
- `summary/`

## 当前 Win11 训练方案和旧主线的区别

当前仓库不再直接让你调用：

```text
external/OpenGait/opengait/main.py
```

而是统一改成：

```text
scripts/run_opengait_main.py
```

原因很简单：

- 旧入口写死了 `nccl`
- 旧入口默认按 Linux 风格理解分布式
- Win11 单卡更需要“少意外、少手工补丁”的包装层

## 一条最短可执行路径

如果你后续都只在 Win11 上操作，我建议你记住这一条：

1. `.\scripts\start_win11_demo.ps1 -SkipLaunch`
2. `.\scripts\start_win11_training.ps1 -Mode smoke`
3. `.\scripts\start_win11_training.ps1 -Mode short-run`
4. `.\scripts\start_win11_training.ps1 -Mode formal-train -ResumeIter 1000`
5. `.\scripts\start_win11_training.ps1 -Mode formal-eval -CheckpointIter 1000`
6. `.\scripts\start_win11_training.ps1 -Mode build-gallery -CheckpointIter 1000`
7. `.\scripts\start_win11_demo.ps1`
