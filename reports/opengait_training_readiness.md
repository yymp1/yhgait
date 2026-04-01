# OpenGait 训练前最终检查与本机推进结论

## 一句话结论

- `CASIA-B-pkl` 已准备完成，可用于后续训练工程
- OpenGait 官方训练入口在当前这台 `macOS Apple Silicon` 机器上仍然卡在 `NCCL`
- 但本机已经通过一个低风险 `CPU/gloo` 兼容探针，真实完成了 `1` 步 smoke 和 `2` 步受限小跑
- 这意味着：本机适合继续做数据检查、环境核对、兼容探针和极小训练验证；正式训练仍更建议迁移到 `Linux + NVIDIA GPU + CUDA/NCCL`

## Pretreatment 产物状态

- 数据目录：`datasets/processed/CASIA-B-pkl`
- `subject_count = 124`
- `pkl_count = 13592`
- 抽样 `pkl` 可读，示例见 `reports/casiab_pkl_check.md`
- 已知原始坏帧：`6` 个，清单见 `reports/casia_b_bad_frames.tsv`

## 官方训练入口检查

- 训练入口：`external/OpenGait/opengait/main.py`
- 基线配置参考：`external/OpenGait/configs/gaitset/gaitset.yaml`
- 本地最小配置：`configs/opengait_casiab_smoke.yaml`

真实 smoke 结果见 `reports/opengait_smoke_test.md`：

- `import_check = passed`
- `artifact_check = passed`
- `data_smoke = passed`
- `entry_smoke = failed`

真实失败点：

```text
RuntimeError: Distributed package doesn't have NCCL built in
```

这说明当前 OpenGait 版本的官方训练入口仍然依赖：

- `torch.distributed.init_process_group('nccl', init_method='env://')`
- CUDA / NCCL 风格的分布式训练

## 本机 CPU/gloo 兼容探针结果

这部分不是官方支持路径，而是为了在当前机器上继续做低风险验证。

### Smoke

- 报告：`reports/cpu_probe_smoke.md`
- 结果：`passed`
- 迭代：`1`
- checkpoint：`1` 个
- 输出目录：`external/OpenGait/output/CASIA-B/GaitSet/cpu_probe_smoke`

### Short run

- 报告：`reports/cpu_probe_short_run.md`
- 结果：`passed`
- 迭代：`2`
- checkpoint：`2` 个
- 输出目录：`external/OpenGait/output/CASIA-B/GaitSet/cpu_probe_short_run_2step`

从这两次真实运行可以确认：

- 配置可解析
- 数据可读
- dataloader 可迭代
- 模型可实例化
- 损失可计算
- optimizer step 可执行
- 日志、summary、checkpoint 都能落盘

## 依赖分层

### 当前阶段必须安装

这些包已经被当前仓库实际用到，并在本机真实验证过：

- `numpy`
- `opencv-python-headless`
- `tqdm`
- `torch`
- `torchvision`
- `PyYAML`
- `tensorboard`
- `einops`
- `kornia`
- `matplotlib`
- `imageio`
- `scikit-learn`

完整环境快照见：

- `reports/opengait_training_env_freeze.txt`

### 可选依赖

- `xformers`

当前本地 clone 会打印 `xFormers not available`，但对于本次使用的 `GaitSet` smoke / short-run 并不构成阻塞。它更多与 `BigGait` / `DINO` 系列路径相关。

### 当前机器上不值得继续投入的重依赖

- CUDA toolkit
- NCCL
- 任何以 Linux + NVIDIA GPU 为前提的训练栈

原因不是“不会装”，而是这台机器本身不提供这条路径所需的运行基础。

## 本机还能继续做什么

- 继续做更小的 CPU 兼容探针
- 改更保守的本地 smoke 配置
- 做 checkpoint 可读性验证
- 做更多数据完整性检查
- 做 OpenGait 配置梳理和迁移交接准备

## 本机不建议继续做什么

- 不建议在这台机器上追求官方训练入口打通
- 不建议把 CPU 兼容探针误当成正式训练方案
- 不建议在没有 CUDA/NCCL 的前提下继续投入大量时间改 OpenGait 主干

## 推荐下一步

优先级最高的是把下面这些带去更合适的训练环境：

- `datasets/processed/CASIA-B-pkl`
- `configs/opengait_casiab_smoke.yaml`
- `reports/opengait_training_env_freeze.txt`
- `docs/opengait_local_patch.md`
- `patches/pretreatment_skip_bad_frames.patch`
- `reports/opengait_smoke_test.md`
- `reports/cpu_probe_smoke.md`
- `reports/cpu_probe_short_run.md`
