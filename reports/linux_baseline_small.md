# Linux Baseline Small

## 结论摘要

- 当前 Linux 服务器已经在真实 `CASIA-B-pkl` 上完成一次小规模 baseline 训练
- 本轮 baseline 真实通过，训练稳定，没有崩溃，生成了 checkpoint、日志和 TensorBoard summary
- 这次 baseline 的定位是“小规模、可回溯验证”，不是正式长训练

## 训练目标

- 验证在真实数据上，当前 `RTX 2070 SUPER + .venvs/opengait-gpu` 是否能够稳定连续训练到比 `5` iter short-run 更完整的阶段
- 验证保存节奏、日志节奏、TensorBoard summary 和 loss 信号是否正常
- 不做正式长训练，不追求最终精度，也不做大规模超参数搜索

## 运行配置

- 配置文件：`configs/opengait_casiab_baseline_small.yaml`
- Python 环境：`/home/bb/gait/yhgait/.venvs/opengait-gpu/bin/python`
- OpenGait 入口：`external/OpenGait/opengait/main.py`
- 数据目录：`/home/bb/gait/yhgait/datasets/processed/CASIA-B-pkl`
- 关键限制：
  - `total_iter = 20`
  - `save_iter = 5`
  - `log_iter = 1`
  - `num_workers = 1`
  - `batch_size = [2, 2]`
  - `frames_num_fixed = 16`
  - `frames_num_min = 16`
  - `frames_num_max = 16`
  - `enable_float16 = false`
  - `with_test = false`
- 学习率调度：
  - `milestones = [10, 15]`
  - 因此 `learning_rate` 在本轮小规模训练里会快速降到 `0.001`

## 执行过程

- 实际命令：

```bash
python3 scripts/run_opengait_gpu_probe.py \
  --python-bin .venvs/opengait-gpu/bin/python \
  --opengait-root external/OpenGait \
  --config-path configs/opengait_casiab_baseline_small.yaml \
  --reports-dir reports/linux_third_round \
  --mode baseline \
  --save-name baseline_small_real \
  --report-stem baseline_small_real
```

- 运行时长：`8.92` 秒
- 执行状态：`passed`

## 训练产物

- 输出目录：
  - `external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real`
- checkpoint：
  - `baseline_small_real-00005.pt`
  - `baseline_small_real-00010.pt`
  - `baseline_small_real-00015.pt`
  - `baseline_small_real-00020.pt`
- 日志：
  - `external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real/logs/2026-04-01-18-07-37.txt`
- TensorBoard：
  - `external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real/summary/events.out.tfevents.1775038057.bb-slam.86158.0`
- 机器生成报告：
  - `reports/linux_third_round/baseline_small_real.md`
  - `reports/linux_third_round/baseline_small_real.json`

## 指标与观察

- 日志中未出现崩溃、OOM、NaN 或 shape mismatch
- `triplet/loss` 的 TensorBoard 首尾值：
  - iter `1`: `0.194561`
  - iter `20`: `0.185069`
- `triplet/hard_loss` 在本轮中有波动，但训练过程保持连续
- `triplet/mean_dist` 从大约 `0.011664` 增长到大约 `0.038999`
- `learning_rate` 从 `0.1` 下降到 `0.001`
- 从日志看：
  - 前 `5` iter 有一段明显下降
  - 中后段存在波动，但仍是“可训练、可连续迭代”的状态
  - 这更像正常的小规模稳定性验证，而不是已经收敛的正式实验

## 风险与结论

- 当前 `RTX 2070 SUPER` 在这套小规模配置下是稳定的
- 当前已经验证通过的保守配置是：
  - `batch_size = [2, 2]`
  - `frames = 16`
  - `num_workers = 1`
  - `total_iter = 20`
- 当前还不建议直接上正式长训练，原因包括：
  - 还没有做更长时间的稳定性观察
  - 还没有验证更大 batch 或更多 worker 的资源边界
  - 当前学习率调度仍是面向小规模验证，不等价于正式训练 recipe
