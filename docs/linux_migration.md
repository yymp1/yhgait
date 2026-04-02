# Linux 迁移最简计划

这份计划只保留“能最快把当前成果迁到 Linux 服务器并开始继续跑”的最少步骤。

## 当前 Linux 第三轮现场结论（2026-04-01）

- 官方 `CASIA-B silhouette` 已直接下载到：
  - `downloads/GaitDatasetB-silh.zip`
- 外层 zip 与内层 `124` 个 subject 归档都已解开，真实原始数据目录已经落到：
  - `datasets/external/casia_b`
- 已安全清理 `47` 个真正为空的 `view` 目录，`scripts/check_casia_b_layout.py` 复检结果为：
  - `readiness = ready`
- 已使用补丁后的 `external/OpenGait/datasets/pretreatment.py` 真执行 pretreatment，日志在：
  - `datasets/processed/casia_b_pretreatment.log`
- 当前真实 pretreatment 产物已经到位：
  - `datasets/processed/CASIA-B-pkl`
  - `subject_count = 124`
  - `pkl_count = 13592`
- 当前 Linux GPU 环境已基于真实数据完成：
  - 标准 smoke：passed
  - GPU probe：passed（`2` iter）
  - 严格受限 short-run：passed（`5` iter）
- 当前最值得做的下一步已经不再是“补真实数据”，而是继续做更稳妥的真实数据小规模验证
- 但仍然**不建议**直接跳到长时间正式训练

第三轮结果请优先看：

- `reports/linux_third_round.md`
- `reports/casia_b_download.md`
- `reports/real_data_validation.md`
- `reports/linux_third_round/opengait_smoke_test.md`
- `reports/linux_third_round/real_gpu_probe.md`
- `reports/linux_third_round/real_gpu_short_run.md`

## 第二轮历史结论

- 当前这台 Linux 服务器已经补齐 `external/OpenGait`，并应用了 `pretreatment_skip_bad_frames.patch`
- 当前仓库内可见的 `datasets/processed/CASIA-B-pkl` 只有 `3` 个 `pkl`，属于第一轮合成最小数据，不是真实 pretreatment 产物
- 当前 Linux 文件系统扫描没有找到另一份真实 `CASIA-B-pkl`
- 当前服务器已经通过独立 GPU 环境验证：
  - `torch.cuda.is_available() = True`
  - `torch.distributed.is_nccl_available() = True`
  - 官方入口 smoke 可在最小合成数据上通过
- 因此，这台 Linux 机当前的主阻塞点不是 GPU/NCCL，而是“真实 `CASIA-B-pkl` 尚未同步到位”

第二轮详细结果见：

- `reports/linux_second_round.md`
- `reports/gpu_compatibility_report.md`
- `reports/real_data_smoke.md`

## 1. 服务器上先准备什么

- Ubuntu 或其他常见 Linux 发行版
- `git`
- `python3.10` 或 `python3.11`
- 足够磁盘空间
- 如果后续要进正式训练，优先准备 `NVIDIA GPU + CUDA + NCCL`

## 2. 从当前项目只带哪些东西

最少带这几类：

- 代码仓库本身
- `datasets/processed/CASIA-B-pkl`
- `configs/opengait_casiab_smoke.yaml`
- `patches/pretreatment_skip_bad_frames.patch`
- `docs/opengait_local_patch.md`
- `reports/opengait_smoke_test.md`
- `reports/opengait_training_readiness.md`
- `reports/opengait_training_env_freeze.txt`

如果你只想先验证训练链路，不需要带：

- `datasets/external/casia_b` 原始 silhouette
- `outputs/`
- `artifacts/`
- `data/`

## 3. Linux 上最小目录建议

```text
~/work/person_orientation_demo
~/data/CASIA-B-pkl
~/work/OpenGait
```

建议把 `CASIA-B-pkl` 放在独立数据盘，不要和代码仓库混在一起。

## 4. Linux 上最小拉起顺序

### 4.1 拉代码

如果 GitHub 已推上去：

```bash
git clone <你的 GitHub 仓库地址> ~/work/person_orientation_demo
cd ~/work/person_orientation_demo
```

### 4.2 拉 OpenGait 官方仓库

```bash
git clone https://github.com/ShiqiYu/OpenGait.git ~/work/OpenGait
```

### 4.3 应用本地 pretreatment 补丁

如果你还会重跑 pretreatment：

```bash
cd ~/work/OpenGait
git apply ~/work/person_orientation_demo/patches/pretreatment_skip_bad_frames.patch
```

### 4.4 准备 Python 环境

先用最小环境脚本把本项目侧依赖装起来。当前这台 Linux 服务器实际落地时使用的是 `Python 3.10` 独立环境：

```bash
cd ~/work/person_orientation_demo
python3.10 -m venv .venvs/opengait_pretreatment
source .venvs/opengait_pretreatment/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/setup_opengait_pretreatment_env.py --venv-dir .venvs/opengait_pretreatment --with-train-smoke
```

如果你已经有更规范的 Linux 训练环境，也可以直接参考：

- `reports/opengait_training_env_freeze.txt`

## 5. Linux 上先跑什么验证

先跑这三步，顺序不要乱。

### 5.1 检查 pretreatment 产物

```bash
cd ~/work/person_orientation_demo
python scripts/run_opengait_smoke_test.py \
  --python-bin .venvs/opengait_pretreatment/bin/python \
  --opengait-root ~/work/OpenGait \
  --config-path configs/opengait_casiab_smoke.yaml \
  --reports-dir reports
```

如果你把 `CASIA-B-pkl` 放在别处，先把配置里的 `dataset_root` 改成真实路径，或者拷一份 smoke 配置到服务器再改。

### 5.2 先看什么算成功

先看：

- `reports/opengait_smoke_test.md`

至少要看到：

- `import_check = passed`
- `artifact_check = passed`
- `data_smoke = passed`

如果当前仓库里只有极小合成数据，这三项也可能通过；这只能说明“路径、配置和入口逻辑可用”，不等价于“真实数据已接通”。

### 5.3 如果服务器有 GPU，再继续看训练入口

在 Linux + GPU + CUDA/NCCL 环境里，最关键的是：

- 官方入口不再报 `Distributed package doesn't have NCCL built in`

如果这一点消失，就说明 Linux 训练主路已经比当前这台 Mac 更接近正式训练。

当前第二轮现场里，这一点已经在独立 GPU 环境上消失；但由于真实 `CASIA-B-pkl` 仍未同步到 Linux，正式训练依然不应立即启动。

## 6. 迁移后不要先做什么

刚迁过去时，不要立刻：

- 开长时间全量训练
- 改大配置
- 换复杂模型
- 同时处理多套数据

先用最小配置验证：

1. 环境没问题
2. 数据没问题
3. 训练入口没问题

再决定是否放大。

## 7. 明确推荐的最短路径

如果你想要最简单、最稳的路线，就是：

1. 把代码仓库推上 GitHub
2. 把 `datasets/processed/CASIA-B-pkl` 单独传到 Linux 服务器
3. Linux 上 clone 当前仓库和 OpenGait
4. 应用 `pretreatment_skip_bad_frames.patch`
5. 创建 Python 3.10/3.11 环境
6. 先跑 `run_opengait_smoke_test.py`
7. 确认 Linux 上用的是“真实 `CASIA-B-pkl`”而不是合成最小数据
8. 成功后再开始更正式的训练配置

## 8. 当前这台 Mac 的角色

这台 Mac 现在最适合继续承担：

- 数据准备
- 配置梳理
- 极小兼容探针
- 文档与补丁整理

不适合承担：

- OpenGait 官方正式训练主机
