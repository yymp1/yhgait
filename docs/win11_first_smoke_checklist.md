# Win11 首轮 Smoke Checklist

这份清单用于把 `feature/win11-single-gpu-mainline` 这条分支在 Win11 新电脑上做第一轮落地验证。

它的目标不是一次跑完所有正式训练，而是先确认下面四件事：

1. Win11 环境可用
2. OpenGait 单卡训练入口可用
3. 现成 checkpoint 的评估 / gallery / GUI 可用
4. 这条分支具备继续合并和继续开发的条件

## 0. 先拉对分支

如果 Win11 电脑上还没有代码：

```powershell
git clone git@github.com:yymp1/yhgait.git D:\gait\yhgait
cd D:\gait\yhgait
git fetch origin
git switch --track origin/feature/win11-single-gpu-mainline
```

如果 Win11 电脑上已经有仓库：

```powershell
cd D:\gait\yhgait
git fetch origin
git switch feature/win11-single-gpu-mainline
git pull --ff-only
```

确认当前分支：

```powershell
git branch --show-current
```

预期输出：

```text
feature/win11-single-gpu-mainline
```

## 1. 先检查机器本身

先跑：

```powershell
nvidia-smi
python --version
git --version
```

通过标准：

- `nvidia-smi` 能正常显示 RTX 3050 Laptop GPU
- Python 是 `3.10` 或 `3.11`
- Git 可用

## 2. 准备 Win11 运行环境

先执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\start_win11_demo.ps1 -RecreateVenv -SkipLaunch
```

这一步的目标：

- 创建 `.venvs\win11-demo`
- 安装 `torch / torchvision`
- 安装 GUI 和训练依赖
- 生成环境检查报告

重点检查：

```text
reports/windows_env_check.json
```

通过标准：

- 没有阻塞项
- `torch.cuda.is_available() = true`
- 能看到 GPU 名称

## 3. 跑训练 smoke test

```powershell
.\scripts\start_win11_training.ps1 -Mode smoke
```

重点看：

- `reports/opengait_smoke_test.md`
- `reports/opengait_smoke_test.json`

通过标准：

- `import_check = passed`
- `artifact_check = passed`
- `data_smoke = passed`
- `entry_smoke = passed`

## 4. 跑单卡 probe

```powershell
.\scripts\start_win11_training.ps1 -Mode probe
```

重点看：

- `reports/real_gpu_probe_2iter.md`
- 或脚本当前生成的 probe 报告文件

通过标准：

- 脚本返回成功
- `external/OpenGait/output/...` 下出现日志和 checkpoint

## 5. 跑单卡 short-run

```powershell
.\scripts\start_win11_training.ps1 -Mode short-run
```

通过标准：

- 能完整跑完
- 没有 `nccl` 相关报错
- 没有 `Distributed package doesn't have NCCL built in`

## 6. 如果已有 1000 iter checkpoint，跑正式评估

如果 Win11 电脑上已经复制了：

```text
external/OpenGait/output/CASIA-B/GaitSet/formal_conservative_real/checkpoints/formal_conservative_real-01000.pt
```

就执行：

```powershell
.\scripts\start_win11_training.ps1 -Mode formal-eval -CheckpointIter 1000
```

通过标准：

- 评估命令执行成功
- 能在日志里看到 `Restore Parameters`
- 能正常输出 Rank-1 结果

## 7. 构建 gallery cache

同样在已有 checkpoint 的前提下执行：

```powershell
.\scripts\start_win11_training.ps1 -Mode build-gallery -CheckpointIter 1000
```

通过标准：

- 生成：

```text
reports/gait_demo_cache_formal_conservative_real_iter01000.npz
```

- 没有 CUDA / 分布式初始化报错

## 8. 跑 GUI smoke

执行：

```powershell
.\scripts\start_win11_demo.ps1
```

打开页面后，至少验证：

1. 页面能正常打开
2. “样本录入”页能加载
3. “视频识别”页能加载
4. 如果已有 checkpoint，处理视频时不会因为 OpenGait 初始化直接报错

## 9. 首轮通过标准

如果下面这些都成立，就可以认为这条分支在 Win11 上已经具备继续推进和合并的条件：

- 环境部署成功
- `start_win11_training.ps1 -Mode smoke` 成功
- `start_win11_training.ps1 -Mode short-run` 成功
- 现成 checkpoint 的评估成功（如果 checkpoint 已复制）
- gallery cache 构建成功（如果 checkpoint 已复制）
- GUI 可以正常打开，并能进入处理流程

## 10. 如果首轮失败，优先回看什么

按这个顺序排查最省时间：

1. `reports/windows_env_check.json`
2. `reports/opengait_smoke_test.md`
3. `external/OpenGait/output/.../logs/*.txt`
4. `nvidia-smi`
5. 当前分支是不是：

```text
feature/win11-single-gpu-mainline
```
