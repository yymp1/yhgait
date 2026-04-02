# Checkpoint Resume Validation

## 结论摘要

- 基于 `baseline_small_real-00020.pt` 的恢复训练已经真实通过
- OpenGait 成功加载 checkpoint，并从 `Iteration 00021` 连续训练到 `Iteration 00025`
- 恢复后没有立刻崩溃，并成功新增 `baseline_small_real-00025.pt`

## 验证目标

- 验证当前这套真实数据 baseline small 训练能否通过 `restore_hint` 正常恢复
- 验证 iteration 是否连续，而不是从 `00001` 重新开始
- 验证恢复后 optimizer / scheduler 状态不会立刻导致崩溃

## 初始训练产物

- 初始输出目录：
  - `external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real`
- 恢复前已有 checkpoint：
  - `baseline_small_real-00005.pt`
  - `baseline_small_real-00010.pt`
  - `baseline_small_real-00015.pt`
  - `baseline_small_real-00020.pt`
- 恢复前日志：
  - `external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real/logs/2026-04-01-18-07-37.txt`
- 恢复前 summary：
  - `external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real/summary/events.out.tfevents.1775038057.bb-slam.86158.0`

## 恢复配置

- 数据与环境沿用 baseline small：
  - 数据目录：`/home/bb/gait/yhgait/datasets/processed/CASIA-B-pkl`
  - Python：`/home/bb/gait/yhgait/.venvs/opengait-gpu/bin/python`
  - `save_name = baseline_small_real`
- 关键恢复参数：
  - `restore_hint = 20`
  - `total_iter = 25`
  - `save_iter = 5`
- 恢复方式：
  - OpenGait 会根据 `save_name + restore_hint` 自动定位：
    `output/CASIA-B/GaitSet/baseline_small_real/checkpoints/baseline_small_real-00020.pt`

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
  --report-stem baseline_small_real_resume \
  --restore-hint 20 \
  --total-iter 25
```

- 运行时长：`7.72` 秒
- 关键日志证据：
  - `Restore Parameters from output/CASIA-B/GaitSet/baseline_small_real/checkpoints/baseline_small_real-00020.pt !!!`
  - 随后日志直接出现：
    - `Iteration 00021`
    - `Iteration 00022`
    - `Iteration 00023`
    - `Iteration 00024`
    - `Iteration 00025`

## 恢复结果

- 恢复后 checkpoint 数从 `4` 个增长到 `5` 个
- 新增 checkpoint：
  - `baseline_small_real-00025.pt`
- 恢复后新增日志文件：
  - `external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real/logs/2026-04-01-18-10-41.txt`
- 恢复后新增 TensorBoard 文件：
  - `external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real/summary/events.out.tfevents.1775038241.bb-slam.86924.0`
- 恢复阶段 loss 继续变化，不是卡死状态：
  - iter `21`: `triplet_loss = 0.1874`
  - iter `25`: `triplet_loss = 0.1308`
- `learning_rate` 在恢复后延续为 `0.001`，符合 scheduler 已走过 milestone 后的状态

## 结论

- 当前这套真实数据小规模训练可以安全 resume
- `restore_hint` 走整数 iteration 的方式在当前仓库可用，且 iteration 连续性已得到真实验证
- 后续推荐：
  - 小规模验证继续沿用同一 `save_name`
  - 在放大训练前，先保留 checkpoint 间隔和 resume 验证流程
  - 如无必要，不要在下一步同时改 batch、frames、worker 和恢复策略
