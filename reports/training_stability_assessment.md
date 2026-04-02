# 训练稳定性评估与资源建议

## 结论

- 当前这台 `RTX 2070 SUPER` 机器在真实 `CASIA-B-pkl` 上已经稳定完成：
  - `20` iter baseline small
  - `20 -> 25` iter checkpoint resume
  - `25 -> 50` iter 中程恢复
  - `50 -> 100` iter 中程扩展
  - `100 -> 300` iter 受控长跑恢复
  - `300 -> 500` iter 受控长跑扩展
- checkpoint、日志、TensorBoard 与 resume 连续性都已得到真实验证
- 当前明确结论是：
  - 已具备启动“第一轮正式保守训练”的条件
  - 仍不适合把当前夜跑结果包装成正式实验结果

## 当前已验证安全组合

- 模型：`GaitSet`
- 数据：真实 `datasets/processed/CASIA-B-pkl`
- 配置：`configs/opengait_casiab_baseline_small.yaml`
- `batch_size = [2, 2]`
- `frames = 16`
- `num_workers = 1`
- `save_iter = 5`
- `log_iter = 1`
- `enable_float16 = false`
- `with_test = false`
- `save_name = baseline_small_real`

这组参数的意义是：

- 先保证真实数据训练稳定、可恢复、可诊断
- 不把机器一下子推到更激进的显存或数据加载边界

## 资源观察

- 有效训练窗口内显存峰值约 `2.7 GiB`
- 有效训练窗口内 GPU 利用率峰值约 `37% ~ 40%`
- 温度维持在 `54C ~ 57C`
- `100 -> 300` 首个恢复 iter 约 `3.50s`，后续稳态约 `0.085s / iter`
- `300 -> 500` 首个恢复 iter 约 `3.49s`，后续稳态约 `0.056s / iter`

这说明：

- 当前保守组合对这张卡是安全的
- 显存没有逼近极限
- 数据加载没有成为当前瓶颈

## 仍然存在的风险点

- 虽然已经稳定验证到 `500` iter，但仍缺少更长时间的耐久观察
- 当前 loss 仍然是高噪声训练窗口，不代表模型已经可靠收敛
- 还没有纳入正式评估流程，本轮没有讨论精度
- 还没有测试更大 `batch_size`、更高 `num_workers` 或更激进混精策略
- stderr 中仍有 PyTorch 2.4+ 的 `ProcessGroupNCCL` 退出清理 warning，虽然本轮没有导致失败，但更长运行仍应继续观察
- `101 -> 300` 在共享工作区里保留了两组成功日志/event，说明这个区间曾被执行过两次；当前结论以最新一次成功记录和 `300 -> 500` 独立续跑为主证据

## 推荐动作

- 如果继续推进，优先把第一轮正式保守训练的评估节奏、记录方式和中止条件定义清楚
- 如果转入第一轮正式保守训练，建议核心训练组合仍保持不变：
  - `batch_size = [2, 2]`
  - `frames = 16`
  - `num_workers = 1`
  - `save_iter = 5`
  - `save_name = baseline_small_real`
- 正式训练阶段如果需要做实验隔离，可在备份当前产物后另开 `save_name`，但这不是本轮夜跑成立的前置条件
- 夜跑目标和正式训练目标要分开记录，不要混写成一条实验结论

## 不建议现在就做的事

- 不建议直接切到正式长时间训练
- 不建议同时改 `batch / workers / frames / lr 策略 / save_name`
- 不建议把本轮 `500 iter` 夜跑结果包装成正式基线实验

## 一句话判断

- 现在已经适合启动第一轮正式保守训练
- 现在仍不适合把训练结果当成正式实验结论直接对外使用
