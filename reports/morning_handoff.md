# 明早先看什么，下一步做什么

## 先看什么

1. `reports/opengait_training_readiness.md`
2. `reports/opengait_smoke_test.md`
3. `reports/cpu_probe_short_run.md`
4. `docs/opengait_local_patch.md`

## 当前结论

- `CASIA-B-pkl` 已经准备好
- OpenGait 官方训练入口在本机仍然卡在 `NCCL`
- 但本机的 `CPU/gloo` 兼容探针已经真实跑通：
  - `1` 步 smoke
  - `2` 步受限小跑
  - checkpoint / log / summary 都已生成

## 如果你明天还在这台 Mac 上继续

优先做这些低风险工作：

1. 复看 `reports/cpu_probe_short_run.md` 的日志和 checkpoint 路径
2. 再决定是否把 `short-run` 从 `2` 步扩到 `3` 到 `5` 步
3. 不要尝试在这台机器上硬啃 OpenGait 官方 `nccl` 训练入口
4. 继续做配置梳理、数据检查和迁移准备

## 如果你明天能切到 Linux/GPU

优先带走这些：

1. `datasets/processed/CASIA-B-pkl`
2. `configs/opengait_casiab_smoke.yaml`
3. `reports/opengait_training_env_freeze.txt`
4. `docs/opengait_local_patch.md`
5. `patches/pretreatment_skip_bad_frames.patch`
6. `reports/opengait_smoke_test.md`
7. `reports/cpu_probe_short_run.md`

## 一句话建议

这台 Mac 现在已经把“数据准备 + 训练前检查 + 极小训练验证”做到位了；下一步最划算的方向不是继续硬改官方训练入口，而是迁移到 `Linux + NVIDIA GPU + CUDA/NCCL` 环境进入正式训练。
