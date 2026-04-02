# Linux 真实数据受控中程验证

## 结论摘要

- 本轮基于真实 `CASIA-B-pkl` 成功完成了两段受控中程验证：
  - `25 -> 50` iter：passed
  - `50 -> 100` iter：passed
- 当前 `baseline_small_real` 目录下的 checkpoint、日志和 TensorBoard summary 已连续覆盖 `1 -> 100`
- 这说明当前 Linux 服务器已经具备“更长但仍受控”的训练验证基础
- 这仍然不是正式长时间训练，也不是最终精度实验

## 复用的稳定配方

- Python：`/home/bb/gait/yhgait/.venvs/opengait-gpu/bin/python`
- OpenGait：`external/OpenGait`
- 数据目录：`datasets/processed/CASIA-B-pkl`
- 配置基线：`configs/opengait_casiab_baseline_small.yaml`
- 输出目录：`external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real`
- 实际运行参数：
  - `batch_size = [2, 2]`
  - `frames = 16`
  - `save_iter = 5`
  - `log_iter = 1`
  - `num_workers = 1`
  - `save_name = baseline_small_real`

说明：

- 旧版自动报告把 `num_workers` 误记成了命令行默认值 `0`
- 但运行态 YAML 与 stderr 中的真实 `data_cfg.num_workers` 一直是 `1`
- 本轮已修正 `scripts/run_opengait_gpu_probe.py` 的报告字段

## 实际命令

### `25 -> 50`

```bash
python3 scripts/run_opengait_gpu_probe.py \
  --python-bin .venvs/opengait-gpu/bin/python \
  --opengait-root external/OpenGait \
  --config-path configs/opengait_casiab_baseline_small.yaml \
  --reports-dir reports/linux_third_round \
  --mode baseline \
  --save-name baseline_small_real \
  --report-stem baseline_midrun_50 \
  --restore-hint 25 \
  --total-iter 50
```

### `50 -> 100`

```bash
python3 scripts/run_opengait_gpu_probe.py \
  --python-bin .venvs/opengait-gpu/bin/python \
  --opengait-root external/OpenGait \
  --config-path configs/opengait_casiab_baseline_small.yaml \
  --reports-dir reports/linux_third_round \
  --mode baseline \
  --save-name baseline_small_real \
  --report-stem baseline_midrun_100 \
  --restore-hint 50 \
  --total-iter 100
```

## `25 -> 50` 真实结果

- 自动报告：
  - `reports/linux_third_round/baseline_midrun_50.md`
  - `reports/linux_third_round/baseline_midrun_50.json`
- 结果：
  - `status = passed`
  - `restore_hint = 25`
  - `total_iter = 50`
  - `duration_seconds = 8.7`
  - `checkpoint_count = 10`
- 日志连续性：
  - 恢复日志文件：`external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real/logs/2026-04-01-18-24-41.txt`
  - 日志区间：`Iteration 00026 -> Iteration 00050`
- 新增 checkpoint：
  - `baseline_small_real-00030.pt`
  - `baseline_small_real-00035.pt`
  - `baseline_small_real-00040.pt`
  - `baseline_small_real-00045.pt`
  - `baseline_small_real-00050.pt`
- 训练节奏：
  - 首 iter 耗时约 `3.46s`
  - 后续稳态 iter 平均耗时约 `0.0575s`
- 指标窗口：
  - `triplet/loss`：`0.187544 -> 0.191927`
  - 结论：存在正常短窗波动，但训练连续，没有卡死、NaN 或崩溃

## `50 -> 100` 真实结果

- 自动报告：
  - `reports/linux_third_round/baseline_midrun_100.md`
  - `reports/linux_third_round/baseline_midrun_100.json`
- 结果：
  - `status = passed`
  - `restore_hint = 50`
  - `total_iter = 100`
  - `duration_seconds = 9.86`
  - `checkpoint_count = 20`
- 日志连续性：
  - 恢复日志文件：`external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real/logs/2026-04-01-18-37-10.txt`
  - 日志区间：`Iteration 00051 -> Iteration 00100`
- 新增 checkpoint：
  - `baseline_small_real-00055.pt`
  - `baseline_small_real-00060.pt`
  - `baseline_small_real-00065.pt`
  - `baseline_small_real-00070.pt`
  - `baseline_small_real-00075.pt`
  - `baseline_small_real-00080.pt`
  - `baseline_small_real-00085.pt`
  - `baseline_small_real-00090.pt`
  - `baseline_small_real-00095.pt`
  - `baseline_small_real-00100.pt`
- 训练节奏：
  - 首 iter 耗时约 `3.41s`
  - 后续稳态 iter 平均耗时约 `0.0557s`
- 指标窗口：
  - `triplet/loss`：`0.189432 -> 0.175835`
  - 结论：仍是短中程高噪声窗口，但训练可持续、可恢复

## 连续性证据

- checkpoint 连续覆盖：
  - `baseline_small_real-00005.pt`
  - `...`
  - `baseline_small_real-00100.pt`
- logs 连续覆盖：
  - `2026-04-01-18-07-37.txt`：`1 -> 20`
  - `2026-04-01-18-10-41.txt`：`21 -> 25`
  - `2026-04-01-18-24-41.txt`：`26 -> 50`
  - `2026-04-01-18-37-10.txt`：`51 -> 100`
- TensorBoard event 连续覆盖：
  - `events.out.tfevents.1775038057...`：`1 -> 20`
  - `events.out.tfevents.1775038241...`：`21 -> 25`
  - `events.out.tfevents.1775039081...`：`26 -> 50`
  - `events.out.tfevents.1775039830...`：`51 -> 100`
- scheduler 连续性：
  - `learning_rate` 在 `21 -> 100` 始终为 `0.001`
  - 说明 resume 后 scheduler 没有被重置

## 资源与稳定性观察

资源采样日志：

- `reports/linux_third_round/midrun_gpu_25_to_50.log`
- `reports/linux_third_round/midrun_gpu_50_to_100.log`

说明：

- 原始 GPU 采样日志在训练结束后多保留了一段空闲期数据
- 为避免把空闲期噪声当成训练窗口，本报告按“日志起点 + 真实运行时长 + 3 秒缓冲”裁剪有效窗口

对齐后资源窗口统计：

- `25 -> 50`：
  - `mem_min = 651 MiB`
  - `mem_max = 2769 MiB`
  - `util_min = 7%`
  - `util_max = 42%`
  - `temp_min = 54C`
  - `temp_max = 57C`
- `50 -> 100`：
  - `mem_min = 655 MiB`
  - `mem_max = 2772 MiB`
  - `util_min = 4%`
  - `util_max = 38%`
  - `temp_min = 54C`
  - `temp_max = 57C`

稳定性判断：

- 没有观察到 OOM
- 没有观察到 CUDA error
- 没有观察到 NCCL error
- 没有观察到 DataLoader 阻塞或崩溃
- stderr 中持续出现 PyTorch 2.4+ 的 `ProcessGroupNCCL` 退出清理 warning，但本轮没有造成训练中断

## 当前判断

- `50 iter` 级别验证：稳定
- `100 iter` 级别验证：稳定
- 当前已经适合开始“更长但仍受控”的正式训练前夜跑
- 当前仍不适合直接把结果升级成正式长时间训练或最终实验结论

## 后续长跑

- `100 -> 300` 与 `300 -> 500` 的后续长跑结论已单独整理到：
  - `reports/linux_longrun_validation.md`
  - `reports/night_run_assessment.md`
