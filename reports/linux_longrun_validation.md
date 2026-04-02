# Linux 真实数据受控长跑验证

## 结论摘要

- 本轮基于真实 `CASIA-B-pkl` 成功完成了两段更长但仍受控的夜跑验证：
  - `100 -> 300` iter：passed
  - `300 -> 500` iter：passed
- 当前 `baseline_small_real` 目录下 checkpoint 已连续覆盖到 `00500`
- 当前日志、TensorBoard 和 restore 连续性在 `500` iter 范围内保持正常
- 这仍然不是正式实验结果，也不是最终精度结论

## 复用的安全组合

- Python：`/home/bb/gait/yhgait/.venvs/opengait-gpu/bin/python`
- OpenGait：`external/OpenGait`
- 数据目录：`datasets/processed/CASIA-B-pkl`
- 配置：`configs/opengait_casiab_baseline_small.yaml`
- `save_name = baseline_small_real`
- `batch_size = [2, 2]`
- `frames = 16`
- `num_workers = 1`
- `save_iter = 5`
- `log_iter = 1`

## 实际命令

### `100 -> 300`

```bash
python3 scripts/run_opengait_gpu_probe.py \
  --python-bin .venvs/opengait-gpu/bin/python \
  --opengait-root external/OpenGait \
  --config-path configs/opengait_casiab_baseline_small.yaml \
  --reports-dir reports/linux_third_round \
  --mode baseline \
  --save-name baseline_small_real \
  --report-stem baseline_longrun_300 \
  --restore-hint 100 \
  --total-iter 300
```

### `300 -> 500`

```bash
python3 scripts/run_opengait_gpu_probe.py \
  --python-bin .venvs/opengait-gpu/bin/python \
  --opengait-root external/OpenGait \
  --config-path configs/opengait_casiab_baseline_small.yaml \
  --reports-dir reports/linux_third_round \
  --mode baseline \
  --save-name baseline_small_real \
  --report-stem baseline_longrun_500 \
  --restore-hint 300 \
  --total-iter 500
```

## `100 -> 300` 真实结果

- 自动报告：
  - `reports/linux_third_round/baseline_longrun_300.md`
  - `reports/linux_third_round/baseline_longrun_300.json`
- 结果：
  - `status = passed`
  - `restore_hint = 100`
  - `total_iter = 300`
  - `duration_seconds = 24.56`
  - `checkpoint_count = 60`
- 新增 checkpoint：
  - `baseline_small_real-00105.pt`
  - `...`
  - `baseline_small_real-00300.pt`
  - 共 `40` 个
- 主日志证据：
  - `external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real/logs/2026-04-01-20-11-39.txt`
  - 覆盖区间：`Iteration 00101 -> Iteration 00300`
- TensorBoard 主证据：
  - `events.out.tfevents.1775045499.bb-slam.151980.0`
  - 覆盖区间：`101 -> 300`
- 训练节奏：
  - 首 iter 约 `3.50s`
  - 后续稳态平均约 `0.0853s / iter`
- 指标窗口：
  - `triplet/loss`：`0.190061 -> 0.152316`
  - `learning_rate`：`0.001 -> 0.001`

### 关于 `101 -> 300` 的重复日志说明

- 当前共享工作区里保留了两组 `101 -> 300` 的成功日志与 event：
  - `2026-04-01-20-11-10.txt`
  - `2026-04-01-20-11-39.txt`
  - `events.out.tfevents.1775045470...`
  - `events.out.tfevents.1775045499...`
- 这说明 `101 -> 300` 这一恢复区间在共享现场里曾被执行过两次
- 两次都成功覆盖 `101 -> 300`，不是半途失败
- 本报告以最新一次成功记录 `20:11:39` 和自动报告 `baseline_longrun_300` 作为主证据

## `300 -> 500` 真实结果

- 自动报告：
  - `reports/linux_third_round/baseline_longrun_500.md`
  - `reports/linux_third_round/baseline_longrun_500.json`
- 结果：
  - `status = passed`
  - `restore_hint = 300`
  - `total_iter = 500`
  - `duration_seconds = 18.42`
  - `checkpoint_count = 100`
- 新增 checkpoint：
  - `baseline_small_real-00305.pt`
  - `...`
  - `baseline_small_real-00500.pt`
  - 共 `40` 个
- 日志证据：
  - `external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real/logs/2026-04-01-20-14-54.txt`
  - 覆盖区间：`Iteration 00301 -> Iteration 00500`
- TensorBoard 证据：
  - `events.out.tfevents.1775045694.bb-slam.154099.0`
  - 覆盖区间：`301 -> 500`
- 训练节奏：
  - 首 iter 约 `3.49s`
  - 后续稳态平均约 `0.0561s / iter`
- 指标窗口：
  - `triplet/loss`：`0.218166 -> 0.130385`
  - `learning_rate`：`0.001 -> 0.001`

## 连续性检查

- checkpoint 连续性：
  - `baseline_small_real-00005.pt`
  - `...`
  - `baseline_small_real-00500.pt`
  - 步长始终为 `5`，无缺口
- logs 连续性：
  - `1 -> 20`
  - `21 -> 25`
  - `26 -> 50`
  - `51 -> 100`
  - `101 -> 300`
  - `301 -> 500`
- TensorBoard 连续性：
  - `1 -> 20`
  - `21 -> 25`
  - `26 -> 50`
  - `51 -> 100`
  - `101 -> 300`
  - `301 -> 500`
- scheduler 连续性：
  - `learning_rate` 在 `101 -> 500` 始终为 `0.001`
  - 说明恢复训练后 scheduler 没被重置

## 资源与稳定性观察

资源采样日志：

- `reports/linux_third_round/longrun_gpu_100_to_300.log`
- `reports/linux_third_round/longrun_gpu_300_to_500.log`

说明：

- 为避免把训练结束后的空闲期混入结论，本报告按“采样起点 + 真实运行时长 + 3 秒缓冲”裁剪有效训练窗口

对齐后资源窗口统计：

- `100 -> 300`：
  - `mem_min = 668 MiB`
  - `mem_max = 2818 MiB`
  - `util_min = 13%`
  - `util_max = 37%`
  - `temp_min = 53C`
  - `temp_max = 57C`
- `300 -> 500`：
  - `mem_min = 677 MiB`
  - `mem_max = 2797 MiB`
  - `util_min = 9%`
  - `util_max = 40%`
  - `temp_min = 50C`
  - `temp_max = 57C`

稳定性判断：

- 没有观察到 OOM
- 没有观察到 CUDA error
- 没有观察到 NCCL fatal error
- 没有观察到 DataLoader 崩溃
- stderr 中仍只有退出阶段的 `ProcessGroupNCCL` warning

## 当前判断

- `300 iter`：稳定
- `500 iter`：稳定
- 当前这台 Linux 服务器已经完成一轮“更长但仍受控”的训练前夜跑
- 现在已经具备启动“第一轮正式保守训练”的条件
- 现在仍不适合把当前夜跑结果直接当成正式实验结论
