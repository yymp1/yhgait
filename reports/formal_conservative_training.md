# 第一轮正式保守训练报告

## 结论

- 本轮已真实完成一轮“第一轮正式保守训练”
- 训练基于真实 `CASIA-B-pkl`
- 训练沿用已验证通过的安全组合
- 训练从已验证的 `baseline_small_real-00500.pt` 恢复到新的独立目录 `formal_conservative_real`
- 本轮实际完成区间：`500 -> 1000`
- 当前结论属于“工程稳定性 + 第一轮正式训练完成”，不属于最终实验精度结论

## 本轮使用的配置

- 配置文件：`configs/opengait_casiab_formal_conservative.yaml`
- 模型：`GaitSet`
- 数据：`datasets/processed/CASIA-B-pkl`
- 核心安全组合：
  - `batch = [2, 2]`
  - `frames = 16`
  - `num_workers = 1`
  - `save_iter = 5`
  - `log_iter = 1`
  - `enable_float16 = false`
  - `with_test = false`

## 输出隔离

- 旧目录：`external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real`
- 新目录：`external/OpenGait/output/CASIA-B/GaitSet/formal_conservative_real`

本轮没有覆盖 `baseline_small_real`。

为保留整数 `restore_hint = 500` 的恢复语义，本轮先复制了恢复点：

- 来源：`baseline_small_real-00500.pt`
- 目标：`formal_conservative_real-00500.pt`

## 实际命令

```bash
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=29571 \
WORLD_SIZE=1 \
RANK=0 \
LOCAL_RANK=0 \
/home/bb/gait/yhgait/.venvs/opengait-gpu/bin/python \
opengait/main.py \
  --cfgs /home/bb/gait/yhgait/configs/opengait_casiab_formal_conservative.yaml \
  --phase train \
  --log_to_file
```

## 实际结果

- 恢复点：`formal_conservative_real-00500.pt`
- 恢复日志证据：
  - `Restore Parameters from output/CASIA-B/GaitSet/formal_conservative_real/checkpoints/formal_conservative_real-00500.pt !!!`
- 完成迭代：
  - 起点：`Iteration 00501`
  - 终点：`Iteration 01000`
- 实际运行时长：
  - 约 `43` 秒

## 产物检查

- checkpoint 目录：
  - `external/OpenGait/output/CASIA-B/GaitSet/formal_conservative_real/checkpoints`
- checkpoint 总数：
  - `101`
- 覆盖范围：
  - `00500 -> 01000`
- 步长：
  - 固定为 `5`
- 连续性：
  - 无缺口

新增 checkpoint 示例：

- `formal_conservative_real-00505.pt`
- `formal_conservative_real-00510.pt`
- `...`
- `formal_conservative_real-01000.pt`

日志产物：

- `external/OpenGait/output/CASIA-B/GaitSet/formal_conservative_real/logs/2026-04-01-21-02-35.txt`
- `external/OpenGait/output/CASIA-B/GaitSet/formal_conservative_real/logs/2026-04-01-21-02-37.txt`

TensorBoard 产物：

- `external/OpenGait/output/CASIA-B/GaitSet/formal_conservative_real/summary/events.out.tfevents.1775048555.bb-slam.191438.0`
- `external/OpenGait/output/CASIA-B/GaitSet/formal_conservative_real/summary/events.out.tfevents.1775048557.bb-slam.191481.0`

说明：

- 本轮出现了 `2` 份日志和 `2` 份 event 文件
- 两者都覆盖同一训练区间，未影响 checkpoint 连续性
- 当前以较新的 `2026-04-01-21-02-37.txt` 作为主核查日志

## 指标与稳定性观察

主日志尾部可见：

- `Iteration 01000`
- `triplet_loss = 0.2785`
- `triplet_hard_loss = 0.4016`

训练中 loss 仍有噪声和尖峰，但本轮没有出现：

- OOM
- CUDA error
- NCCL fatal error
- DataLoader 崩溃

stderr / 退出侧仍可见：

- PyTorch 2.4+ 的 `ProcessGroupNCCL` 退出 warning

该 warning 在前几轮已经出现过，本轮没有导致训练失败。

## GPU 资源记录

资源采样文件：

- `reports/formal_conservative_run/resource_monitor.csv`

采样窗口统计：

- `mem_min = 2775 MiB`
- `mem_max = 4899 MiB`
- `util_min = 26%`
- `util_max = 62%`
- `temp_min = 54C`
- `temp_max = 58C`

观察结论：

- 显存占用稳定在当前卡可接受范围内
- 温度正常
- GPU 利用率有明显训练负载，但没有出现异常飙升

## 本轮判断

- 这轮 `1000 iter` 已足以作为“第一轮正式保守训练”收口
- 当前不需要自动继续无限扩大
- 如果后续继续，应作为下一轮独立实验决策，不应把本轮直接延长后混写成同一轮
