# Win11 新电脑部署指南

这份文档给“把当前项目迁到另一台 `Windows 11 + NVIDIA GPU` 电脑，并把私有步态库 demo 跑起来”的场景用。

如果你只是想尽快启动，不想先理解训练细节，建议直接用仓库里的：

- `scripts/start_win11_demo.bat`
- `scripts/start_win11_demo.ps1`

它们会做：

1. 检查 `Python / PyTorch / CUDA / OpenGait / checkpoint`
2. 创建本地虚拟环境
3. 安装 demo 依赖
4. 生成环境检查报告
5. 启动 Gradio 页面

## 先说清楚：Win11 这条路主要解决什么

当前仓库在 Win11 上最适合完成的是：

- 启动“私有步态库录入 / 识别” GUI
- 上传视频，录入本地身份样本
- 上传新视频，和本地私有步态库做 Top-K 识别

当前仓库在 Win11 上**不建议**作为主目标的是：

- OpenGait 官方正式训练
- 长时间训练 / 正式 benchmark
- Linux + NCCL 那套正式训练工作流

原因是当前 `external/OpenGait/opengait/main.py` 仍然是 Linux / NCCL 风格的官方训练入口。  
本仓库已经为“本地视频推理 / 私有库 demo”补了 Windows 可用的包装，但这不等于“Windows 已经变成正式训练主机”。

## 新电脑最少要带走什么

如果你想在另一台 Win11 电脑上直接跑 GUI，至少要准备下面这些东西。

### 1. 当前仓库代码

最简单的方式就是把整个项目文件夹直接拷过去。

如果你是从 Git 拉代码，要特别注意：当前仓库的 `.gitignore` 默认不会把 `external/OpenGait/` 和大模型 checkpoint 一起带走，所以**仅靠 git clone 通常不够**。

### 2. `external/OpenGait/`

必须要有：

```text
external/OpenGait/opengait/
external/OpenGait/configs/
```

如果新电脑上没有这份目录：

- 可以先运行 `scripts/start_win11_demo.ps1`
- 脚本会在 `external/OpenGait` 缺失时尝试自动 `git clone`

但这一步只会补 OpenGait 代码，不会补你的私有 checkpoint。

### 3. 训练好的 checkpoint

完整 demo 默认会查找：

```text
external/OpenGait/output/CASIA-B/GaitSet/formal_conservative_real/checkpoints/formal_conservative_real-01000.pt
```

这一项**必须从旧电脑拷过去**，脚本不会自动下载。

建议把下面整个目录一起带走，这样最省事：

```text
external/OpenGait/output/CASIA-B/GaitSet/formal_conservative_real/
```

### 4. 如果你还想保留旧样本库

把这一份一起拷过去：

```text
data/private_gallery/
```

如果不拷也没关系，新电脑上会从空样本库开始。

### 5. 如果你后面还要做训练 / 评估

额外再带走：

```text
datasets/processed/CASIA-B-pkl/
```

这份数据**不是启动 GUI 的必需项**，但会影响后续训练和官方评估。

## 新电脑建议的目录形态

推荐把项目放在本地磁盘，比如：

```text
D:\gait\yhgait
```

进入项目后，关键目录最好是：

```text
yhgait/
  app/
  configs/
  docs/
  reports/
  scripts/
  src/
  external/
    OpenGait/
      opengait/
      output/
        CASIA-B/
          GaitSet/
            formal_conservative_real/
              checkpoints/
                formal_conservative_real-01000.pt
  data/
    private_gallery/
```

## 新电脑第一次启动：最简单的做法

### 方法 A：双击启动

在资源管理器里直接双击：

```text
scripts/start_win11_demo.bat
```

这是最适合小白的方式。

### 方法 B：PowerShell 启动

在项目根目录打开 PowerShell：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\start_win11_demo.ps1
```

## 启动脚本会做什么

默认脚本会：

1. 检查 `nvidia-smi`
2. 创建虚拟环境：`.venvs\win11-demo`
3. 安装 `torch / torchvision`
4. 安装 `requirements-gait-demo.txt`
5. 运行 `scripts/check_windows_demo_env.py`
6. 启动：

```powershell
python app\gradio_gait_demo.py
```

如果一切正常，浏览器会打开：

```text
http://127.0.0.1:7860
```

## 推荐的第一次执行命令

第一次我建议你在 PowerShell 里运行：

```powershell
.\scripts\start_win11_demo.ps1 -RecreateVenv
```

这样可以强制重建环境，避免旧环境残留。

## 常用参数

### 1. 指定端口

如果 `7860` 被占用：

```powershell
.\scripts\start_win11_demo.ps1 -ServerPort 7861
```

### 2. 只做环境部署，不立刻启动

```powershell
.\scripts\start_win11_demo.ps1 -SkipLaunch
```

### 3. 不自动 clone OpenGait

如果你不希望脚本联网拉 OpenGait：

```powershell
.\scripts\start_win11_demo.ps1 -SkipOpenGaitClone
```

### 4. 改 checkpoint iter

如果你以后想切别的 checkpoint：

```powershell
.\scripts\start_win11_demo.ps1 -CheckpointIter 500
```

前提是对应的 `.pt` 文件真的存在。

## 环境检查报告在哪里看

脚本会生成：

```text
reports/windows_env_check.json
```

也可以手动运行：

```powershell
.\.venvs\win11-demo\Scripts\python.exe .\scripts\check_windows_demo_env.py --strict
```

这个检查会告诉你：

- Python 版本是否合适
- PyTorch 是否识别到 CUDA
- Windows 上会使用 `gloo` 还是别的后端
- `external/OpenGait` 是否存在
- 默认 checkpoint 是否存在
- 哪些依赖缺失

## 推荐的手工启动命令

如果环境已经装好，以后不想每次都重新部署，可以直接：

```powershell
.\.venvs\win11-demo\Scripts\python.exe .\app\gradio_gait_demo.py --server-name 127.0.0.1 --server-port 7860
```

## 如果启动失败，优先看这几类问题

### 1. 提示缺少 checkpoint

最常见。  
你需要确认下面这个文件真的在新电脑上：

```text
external/OpenGait/output/CASIA-B/GaitSet/formal_conservative_real/checkpoints/formal_conservative_real-01000.pt
```

### 2. 提示 `torch.cuda.is_available() = False`

这通常意味着：

- NVIDIA 驱动没装好
- 装成了 CPU 版 PyTorch
- CUDA 轮子和当前机器不匹配

先看：

```powershell
nvidia-smi
```

再看环境检查报告里的 `torch` 段落。

### 3. `mediapipe` 安装失败

优先确认：

- 你用的是 `Python 3.10` 或 `3.11`
- 系统是 64 位 Win11

如果仍失败，通常要补装微软的 Visual C++ 运行库。

### 4. PowerShell 不允许执行脚本

直接在当前终端临时放开：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

然后再运行脚本。

### 5. 页面打不开

先看控制台里是否已经打印出：

```text
Running on local URL:  http://127.0.0.1:7860
```

如果已经打印，但浏览器没开：

- 手动复制这个地址到浏览器
- 或换一个端口重新启动

## 一个很重要的现实提醒

当前 Win11 方案的重点是“把 demo 跑起来”，不是“把 OpenGait 官方训练完整迁过去”。

也就是说：

- Win11 + RTX 3050 很适合本地演示和样本录入识别
- 真正大规模训练、正式 benchmark、长期训练，还是更推荐 `Linux + NVIDIA + CUDA/NCCL`

## 最后给小白的最短路径

如果你只想最省心地跑起来，就按这个顺序做：

1. 把整个项目目录复制到 Win11 电脑
2. 确认 `external/OpenGait/output/.../formal_conservative_real-01000.pt` 已经拷过去
3. 双击 `scripts/start_win11_demo.bat`
4. 看到浏览器打开 `http://127.0.0.1:7860`
5. 先去“样本录入”页面录入一段视频，再去“视频识别”页面测试
