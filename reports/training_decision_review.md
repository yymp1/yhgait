# formal_conservative_real 结果分析与下一步决策

## 这次到底看了什么

- 训练与评估报告：
  - `reports/formal_conservative_eval.md`
  - `reports/formal_conservative_snapshot.md`
  - `reports/formal_conservative_final_summary.md`
  - `reports/formal_conservative_assessment.md`
  - `reports/index.md`
  - `reports/opengait_training_readiness.md`
- 训练与评估配置：
  - `configs/opengait_casiab_formal_conservative.yaml`
  - `configs/opengait_casiab_formal_conservative_eval.yaml`
- 训练输出：
  - `external/OpenGait/output/CASIA-B/GaitSet/formal_conservative_real/checkpoints`
  - `external/OpenGait/output/CASIA-B/GaitSet/formal_conservative_real/logs`
  - `external/OpenGait/output/CASIA-B/GaitSet/formal_conservative_real/summary`
- 关键 checkpoint：
  - `external/OpenGait/output/CASIA-B/GaitSet/formal_conservative_real/checkpoints/formal_conservative_real-01000.pt`
- 资源监控：
  - `reports/formal_conservative_run/resource_monitor.csv`
  - `reports/formal_conservative/gpu_usage.csv`
  - `reports/formal_conservative/gpu_usage_resume.csv`

## 训练曲线怎么读

### 先看工程层面的事实

- `00500 -> 01000` 的 checkpoint 共 `101` 个，步长固定 `5`，没有缺口
- `01000.pt` 文件存在，大小约 `20M`
- 主训练日志覆盖 `Iteration 00501 -> Iteration 01000`
- TensorBoard event 文件可读取，两个 event 的 scalar 内容一致
- 官方评估日志确认 `01000.pt` 能被正常 restore

### TensorBoard 里真正记录到了什么

当前 event 文件只包含这 5 个 scalar：

- `triplet/loss`
- `triplet/hard_loss`
- `triplet/loss_num`
- `triplet/mean_dist`
- `learning_rate`

关键读数：

- `triplet/loss`
  - 前 `50` iter 均值：`0.2038`
  - 后 `100` iter 均值：`0.1915`
  - 后 `50` iter 均值：`0.1974`
- `triplet/hard_loss`
  - 前 `50` iter 均值：`0.4150`
  - 后 `100` iter 均值：`0.3761`
  - 后 `50` iter 均值：`0.3805`
- `triplet/mean_dist`
  - 前 `50` iter 均值：`0.2190`
  - 后 `100` iter 均值：`0.2594`
  - 后 `50` iter 均值：`0.2657`
- `learning_rate`
  - `501 -> 1000` 全程固定为 `0.001`

### 只看 500 到 1000 的分段均值

- `501-600`: `triplet_loss = 0.1888`
- `601-700`: `triplet_loss = 0.1789`
- `701-800`: `triplet_loss = 0.1859`
- `801-900`: `triplet_loss = 0.1864`
- `901-1000`: `triplet_loss = 0.1915`

### 曲线判断

- 这段训练不是“越跑越差”，因为和最前段相比，后 `100` iter 的 `triplet_loss` 和 `triplet/hard_loss` 仍略低
- 但它也不是“还在明显持续收敛”，因为 `700` iter 之后没有稳定下行，`901-1000` 反而略高于 `801-900`
- 结合学习率全程固定 `0.001`，当前更像高噪声平台区，而不是还在明显吃到训练长度红利

一句话说，就是：

- 训练还算健康
- 但从 `1000 -> 1500` 继续拉长，收益看起来不会很大，至少目前没有强证据支持“马上续训一定划算”

## 评估结果怎么读

官方评估入口已经真实跑通，结果来自：

- `external/OpenGait/output/CASIA-B/GaitSet/formal_conservative_real/logs/2026-04-01-21-21-48.txt`
- `reports/formal_conservative_eval/eval_stderr.log`

评估结果是：

- `NM@R1 = 23.95%`
- `BG@R1 = 17.17%`
- `CL@R1 = 8.00%`

这组结果说明了两件事：

1. 工程闭环是真的通了
   - 真实数据可读
   - checkpoint 可 restore
   - 官方 test 入口可执行
   - 指标可正常产出
2. 实验上它更像“第一轮正式 baseline”
   - 不是空跑
   - 不是失败结果
   - 但也还不是一个足以证明“应该立刻重投入深挖”的强结果

更准确地说：

- 这组结果已经足够当后续所有实验的对照组
- 但它当前的价值主要是“建立了可信 baseline”，不是“已经逼近最终结论”

## 工程层面是否成功

结论：成功。

依据：

- checkpoint 连续
- restore 可用
- eval 跑通
- 没有 OOM、CUDA fatal error、NCCL fatal error、DataLoader 崩溃
- GPU 资源稳定：
  - 显存大致在 `4.89 GB` 左右运行
  - 利用率大致 `54% ~ 63%`
  - 温度大致 `54C ~ 58C`

当前 warning 的性质：

- `xFormers not available`
  - 对本轮 `GaitSet` 不构成阻塞
- `ProcessGroupNCCL` 退出 warning
  - 当前更像清理不完整的退出噪声，不是训练失败信号

工程判断可以直接下结论：

- 这轮训练在工程层面是成功的
- 当前没有必须先修的阻塞级工程风险

## 实验层面是否成功

结论：成功，但成功的类型是“baseline 成功”，不是“结果强成功”。

更具体一点：

- 成功在于：
  - 拿到了真实数据上的正式训练产物
  - 拿到了可复现的官方评估数值
  - 已经有了一个后续可比较的正式对照组
- 还不算强成功在于：
  - 只评估了 `01000.pt` 这一点
  - 曲线后半段没有明显继续改善
  - 当前数值更像第一轮起点，而不是已经证明值得立刻拉长训练的结果

## 当前最合理的主建议

主建议：`暂停继续训练，先做更多评估/分析再决定`。

也就是四个选项里的第 `4` 条。

理由很直接：

1. 这轮的工程目标已经完成，不需要用“继续训练”来证明链路稳定。
2. 当前 `1000 iter` 之后的曲线没有给出很强的继续收益信号。
3. 目前只有 `01000.pt` 的正式评估，没有 `00500`、`00750`、`00900` 这些对照点。
4. 在这种信息量下，先补评估比直接再烧 `500 iter` 更划算。

### 这一步最该补什么分析

优先顺序建议是：

1. 用同一套评估配置补评估 `00500.pt`
2. 如果想再多一个点，补评估 `00750.pt` 或 `00900.pt`
3. 把 `500 / 750(or 900) / 1000` 三个点做成一张对比表

如果 `1000` 明显优于 `500`，那时再讨论 `1000 -> 1500` 会更有依据。
如果 `500` 和 `1000` 差别很小，那继续拉长训练的优先级就会下降。

## 备选建议

### 备选 1

`暂停继续训练，先整理 baseline 并做对比实验`。

适用情况：

- 你现在更关心“下一轮实验怎么设计”而不是“这 500 iter 值不值得再跑”

原因：

- `formal_conservative_real-01000.pt` 已经足够当正式 baseline
- 先把它作为对照组固定下来，会比一边续训一边变更实验目标更清楚

### 备选 2

`暂停继续训练，优先改单一变量：frames`。

如果一定要做一个单变量实验，我最推荐先改 `frames`，而不是先改 `batch` 或 `num_workers`。

建议方式：

- 保持 `batch=[2,2]`
- 保持 `num_workers=1`
- 保持 `save_iter=5`
- 只把 `frames` 从 `16` 小步改到 `20`

原因：

- 当前 `16` 帧本身就是偏保守的工程安全值
- `frames` 更直接影响序列信息量，实验价值通常高于改 `workers`
- 先改 `batch` 容易把优化噪声和显存行为一起改掉，不利于解释结果

## 最后一句明确话

- `建议先不要继续训练，先做更多评估/分析再决定。`
