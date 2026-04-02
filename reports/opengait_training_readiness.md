# OpenGait 训练前最终检查与本机推进结论

> 说明：这份文档主要记录历史上的 `macOS Apple Silicon` 推进结论。
> 当前 Linux 服务器上的第三轮现场结果已经再次发生变化，尤其是“真实数据是否到位”这一点已经从 blocked 变为 ready。
> 请优先结合下面这些更新材料一起看：
>
> - `reports/linux_third_round.md`
> - `reports/linux_midrun_validation.md`
> - `reports/linux_longrun_validation.md`
> - `reports/night_run_assessment.md`
> - `reports/training_stability_assessment.md`
> - `reports/real_data_validation.md`
> - `reports/casia_b_download.md`
> - `reports/linux_third_round/opengait_smoke_test.md`
> - `reports/linux_third_round/real_gpu_probe.md`
> - `reports/linux_third_round/real_gpu_short_run.md`
> - `reports/linux_second_round.md`
> - `reports/gpu_compatibility_report.md`
> - `reports/real_data_smoke.md`
> - `reports/formal_conservative_training.md`
> - `reports/formal_conservative_assessment.md`
> - `reports/formal_conservative_eval.md`
> - `reports/formal_conservative_snapshot.md`
> - `reports/formal_conservative_final_summary.md`

## 当前 Linux 最新结论（2026-04-01）

- 真实 `CASIA-B silhouette` 已在 Linux 上直接下载、解压、检查并清理空目录
- 真实 `datasets/processed/CASIA-B-pkl` 已重新生成完成：
  - `subject_count = 124`
  - `pkl_count = 13592`
- 当前 Linux 的 GPU/NCCL 路线不只是在合成最小数据上可用，而是已经在**真实数据**上通过：
  - 标准 smoke
  - `2` iter probe
  - `5` iter 严格受限 short-run
- 当前 Linux 还进一步完成了：
  - `20` iter baseline small
  - checkpoint resume `20 -> 25`
- 当前 Linux 还进一步完成了受控中程验证：
  - `25 -> 50` iter resume
  - `50 -> 100` iter extension
  - checkpoint / log / TensorBoard 连续到 `100` iter
- 当前 Linux 还进一步完成了受控长跑验证：
  - `100 -> 300` iter resume
  - `300 -> 500` iter extension
  - checkpoint / log / TensorBoard 连续到 `500` iter
- 当前 Linux 还进一步完成了第一轮正式保守训练：
  - `500 -> 1000` iter
  - checkpoint / log / TensorBoard 连续到 `1000` iter
- 当前 Linux 已基于 `formal_conservative_real-01000.pt` 完成一次真实官方评估
- 当前最小官方评估结果：
  - `NM@R1 = 23.95%`
  - `BG@R1 = 17.17%`
  - `CL@R1 = 8.00%`
- 因此，当前 Linux 现场已经不再受“真实数据缺失”阻塞
- 当前更准确的判断是：
  - 第一轮正式保守训练已经完成并完成了评估收口
  - 当前更适合先看评估结果，再决定是否进入下一轮更长训练

## 一句话结论

- `CASIA-B-pkl` 已准备完成，当前 Linux GPU 路线已在真实数据上稳定跑完 `formal_conservative_real 1000 iter`
- `formal_conservative_real-01000.pt` 已完成一次真实官方评估并输出指标
- 当前已经具备“第一轮正式保守训练已完成并可交接”的收口状态
- 下文关于 `macOS Apple Silicon` 的大段说明仅作历史归档，不构成对当前 Linux readiness 的否定

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
- 正式保守训练配置：`configs/opengait_casiab_formal_conservative.yaml`
- 正式评估配置：`configs/opengait_casiab_formal_conservative_eval.yaml`

历史 smoke 结果见 `reports/opengait_smoke_test.md`：

- `import_check = passed`
- `artifact_check = passed`
- `data_smoke = passed`
- `entry_smoke = failed`

历史失败点：

```text
RuntimeError: Distributed package doesn't have NCCL built in
```

这说明 OpenGait 官方入口依赖：

- `torch.distributed.init_process_group('nccl', init_method='env://')`
- CUDA / NCCL 风格的分布式训练

但对当前 Linux 主机来说，这个阻塞已经在后续轮次解决，证据包括：

- 真实数据 GPU smoke / probe / short-run 已通过
- `formal_conservative_real` 正式保守训练已完成到 `1000` iter
- `formal_conservative_real-01000.pt` 官方评估已真实跑通

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

- 与当前 `GaitSet + CASIA-B + PyTorch cu118` 已验证组合无关的大版本栈切换
- 在没有明确收益判断前，盲目升级驱动、PyTorch 主版本或改训练栈

原因不是机器不支持 GPU，而是当前已经有一套可用组合，不值得在本轮收口阶段重新引入不必要变量。

## 本机还能继续做什么

- 做正式评估结果审阅
- 做更多 checkpoint 评估快照
- 基于 `01000.pt` 规划下一轮更长训练
- 做结果归档与交接材料整理

## 本机不建议继续做什么

- 不建议把本轮收口和下一轮长训练混写
- 不建议在没有先看评估结果的前提下直接继续扩训
- 不建议在当前收口阶段同时改模型、batch、workers、frames 等关键变量

## 推荐下一步

优先级最高的是先基于当前结果做判断，而不是马上继续训练：

1. 先审阅这轮正式训练与评估结果：
   - `reports/formal_conservative_training.md`
   - `reports/formal_conservative_assessment.md`
   - `reports/formal_conservative_eval.md`
   - `reports/formal_conservative_snapshot.md`
   - `reports/formal_conservative_final_summary.md`
2. 如果确认值得继续，再从 `formal_conservative_real-01000.pt` 规划下一轮更长训练。
