# formal_conservative_real 结果快照

## 快照结论

- 训练轮次：`500 -> 1000`
- 当前最终 checkpoint：`formal_conservative_real-01000.pt`
- 当前状态：训练已完整完成，官方评估已真实跑通

## 关键产物

- 最终 checkpoint：
  - `external/OpenGait/output/CASIA-B/GaitSet/formal_conservative_real/checkpoints/formal_conservative_real-01000.pt`
- 正式训练配置：
  - `configs/opengait_casiab_formal_conservative.yaml`
- 评估配置：
  - `configs/opengait_casiab_formal_conservative_eval.yaml`
- 训练主日志：
  - `external/OpenGait/output/CASIA-B/GaitSet/formal_conservative_real/logs/2026-04-01-21-02-37.txt`
- 评估日志：
  - `external/OpenGait/output/CASIA-B/GaitSet/formal_conservative_real/logs/2026-04-01-21-21-48.txt`
- TensorBoard：
  - `external/OpenGait/output/CASIA-B/GaitSet/formal_conservative_real/summary/events.out.tfevents.1775048555.bb-slam.191438.0`
  - `external/OpenGait/output/CASIA-B/GaitSet/formal_conservative_real/summary/events.out.tfevents.1775048557.bb-slam.191481.0`
- 资源监控：
  - `reports/formal_conservative_run/resource_monitor.csv`

## 训练产物概况

- checkpoint 数量：`101`
- checkpoint 范围：`00500 -> 01000`
- checkpoint 步长：`5`
- 连续性：无缺口
- 主训练日志迭代覆盖：`Iteration 00501 -> Iteration 01000`

## 评估结果快照

- checkpoint 加载：成功
- 数据读取：成功
- 官方评估入口：成功
- 最小评估结果：
  - `NM@R1 = 23.95%`
  - `BG@R1 = 17.17%`
  - `CL@R1 = 8.00%`

## 已知 warning / 非阻塞问题

- 训练阶段存在 `2` 份日志和 `2` 份 event 文件
  - 未影响 checkpoint 连续性
- 训练与评估退出时都可见 `ProcessGroupNCCL` 清理 warning
  - 当前没有导致训练或评估失败
- `xFormers not available`
  - 对本轮 `GaitSet` 路径不构成阻塞

## 一句话归档

- `formal_conservative_real` 已完成“真实数据正式保守训练 + 一次真实官方评估验证”，当前可以按第一轮正式可交接成果归档。
