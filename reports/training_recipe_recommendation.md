# Training Recipe Recommendation

## 结论摘要

- 当前更适合继续使用“保守、小规模、可恢复”的训练配方，而不是直接进入长时间正式训练
- 在这台 `RTX 2070 SUPER` 机器上，当前已真实验证通过的推荐起点是：
  - `batch_size = [2, 2]`
  - `frames = 16`
  - `num_workers = 1`
  - `total_iter = 20 / 25 / 50 / 100` 已验证
  - `save_iter = 5`
  - `log_iter = 1`
- 当前最值得做的是基于同一配方进入“更长但仍受控”的夜跑，而不是直接上正式实验

## 当前状态

- 真实数据已经 ready：
  - `subject_count = 124`
  - `pkl_count = 13592`
- GPU 路线已经基于真实数据验证通过：
  - smoke：passed
  - probe：passed
  - `5` iter short-run：passed
  - `20` iter baseline small：passed
  - checkpoint resume `20 -> 25`：passed
  - 中程 resume `25 -> 50`：passed
  - 中程 extension `50 -> 100`：passed
- 当前不建议直接长训的原因：
  - 还没有做真正更长时间的耐久观察
  - 还没有验证更高 worker / 更大 batch 的资源边界
  - 当前 loss 仍处于短中程高噪声窗口，不代表最终收敛或最终精度

## 推荐配方

- 当前推荐的小规模验证配方：
  - 配置基线：`configs/opengait_casiab_baseline_small.yaml`
  - `batch_size = [2, 2]`
  - `frames_num_fixed = 16`
  - `frames_num_min = 16`
  - `frames_num_max = 16`
  - `num_workers = 1`
  - `enable_float16 = false`
  - `with_test = false`
  - `save_iter = 5`
  - `log_iter = 1`
  - `save_name = baseline_small_real`
- 当前推荐保留 resume 验证与同目录连续训练：
  - `restore_hint = 20 / 25 / 50`
  - 将同一 `save_name` 延续到 `100` iter
- 当前不建议立即尝试：
  - 更大的 batch
  - 更高的 worker 数
  - 直接拉到长时间正式训练级别的 iter
  - 同时引入多处配置变化

## 适用边界

- 这个配方适合：
  - 验证真实数据训练链路是否稳定
  - 验证 checkpoint / 日志 / summary 是否正常
  - 验证恢复训练是否连续
  - 做下一轮更长但仍受控的训练前准备
- 这个配方不适合：
  - 追求最终精度
  - 对外宣称正式基线结果
  - 代替正式训练 recipe

## 风险点

- 当前机器是 `RTX 2070 SUPER`，显存和吞吐量都不是为长时间高强度实验准备的
- 当前尚未验证更大 `batch_size` 或更高 `num_workers` 的资源边界
- 当前 baseline small 的学习率调度是为了小规模验证收口，不是正式长训策略
- 当前 stderr 中仍有 PyTorch 2.4+ 的 `ProcessGroupNCCL` 退出清理 warning，需要在更长训练中继续观察
- 若出现以下信号，就不应继续放大训练：
  - OOM
  - checkpoint 写入异常
  - 恢复训练后 iteration 不连续
  - loss 出现 NaN 或长期异常发散

## 下一步

1. 保留当前 `baseline_small_real` 目录和 `00100.pt`
2. 在不改动 batch / frames / workers 主参数的前提下，优先做一轮更长但仍受控的夜跑验证
3. 在继续放大前，优先复看：
   - `reports/linux_baseline_small.md`
   - `reports/checkpoint_resume_validation.md`
   - `reports/linux_midrun_validation.md`
   - `reports/training_stability_assessment.md`
4. 只有当更长一轮的受控夜跑仍稳定时，才值得讨论是否进入正式长训练
