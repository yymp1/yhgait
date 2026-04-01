# OpenGait CPU Probe 报告

- 状态：`passed`
- OpenGait：`/Volumes/Data/person_orientation_demo/external/OpenGait`
- 配置：`/Volumes/Data/person_orientation_demo/configs/opengait_casiab_smoke.yaml`
- 训练步数：`5`
- save_name：`cpu_probe_short_run`
- 数据根目录：`/Volumes/Data/person_orientation_demo/datasets/processed/CASIA-B-pkl`
- partition：`/Volumes/Data/person_orientation_demo/external/OpenGait/datasets/CASIA-B/CASIA-B.json`
- 输出目录：`/Volumes/Data/person_orientation_demo/external/OpenGait/output/CASIA-B/GaitSet/cpu_probe_short_run`
- 总耗时：`2.362s`
- 清理 sidecar 数：`6`

## 步级结果
- step=1 iteration=1 loss=0.191288 optimizer_step_ok=True
- step=2 iteration=2 loss=0.194605 optimizer_step_ok=True
- step=3 iteration=3 loss=0.192401 optimizer_step_ok=True
- step=4 iteration=4 loss=0.192605 optimizer_step_ok=True
- step=5 iteration=5 loss=0.194123 optimizer_step_ok=True

## Checkpoint
- `output/CASIA-B/GaitSet/cpu_probe_short_run/checkpoints/._cpu_probe_short_run-00005.pt`
- `output/CASIA-B/GaitSet/cpu_probe_short_run/checkpoints/cpu_probe_short_run-00005.pt`

## Logs
- `/Volumes/Data/person_orientation_demo/external/OpenGait/output/CASIA-B/GaitSet/cpu_probe_short_run/logs/2026-04-01-03-14-35.txt`

## Summary Files
- `/Volumes/Data/person_orientation_demo/external/OpenGait/output/CASIA-B/GaitSet/cpu_probe_short_run/summary/events.out.tfevents.1774984475.yympdeMac-mini.local.95090.0`

## 说明
- 这个脚本只用于 macOS/CPU 上的极小训练步探针，不替代 OpenGait 官方训练入口。
- 官方入口 external/OpenGait/opengait/main.py 仍然需要 NCCL/CUDA，本脚本通过运行时 monkeypatch 绕开该约束，只验证数据、模型、loss 和优化步是否能在 CPU 上跑通。
