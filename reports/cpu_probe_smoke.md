# OpenGait CPU 兼容探针报告

- 模式：`smoke`
- 状态：`passed`
- 迭代数：`1`
- save_name：`cpu_probe_smoke`
- 输出目录：`/Volumes/Data/person_orientation_demo/external/OpenGait/output/CASIA-B/GaitSet/cpu_probe_smoke`
- 实际迭代：`1`
- checkpoint 数：`1`
- 运行耗时：`4.25` 秒
- 清理的 `._*` 数量：`6`

## 说明
- 这是本机 CPU/gloo 兼容探针，不是官方支持的 CUDA/NCCL 正式训练入口。
- 探针通过 monkeypatch 绕过了 OpenGait 当前版本里硬编码的 cuda/ddp 包装逻辑。

## 产物
- checkpoints：`['/Volumes/Data/person_orientation_demo/external/OpenGait/output/CASIA-B/GaitSet/cpu_probe_smoke/checkpoints/cpu_probe_smoke-00001.pt']`
- logs：`['/Volumes/Data/person_orientation_demo/external/OpenGait/output/CASIA-B/GaitSet/cpu_probe_smoke/logs/2026-04-01-03-15-23.txt']`
- summary：`['/Volumes/Data/person_orientation_demo/external/OpenGait/output/CASIA-B/GaitSet/cpu_probe_smoke/summary/events.out.tfevents.1774984523.yympdeMac-mini.local.95147.0']`

## 最新日志尾部
```text
[2026-04-01 03:15:23] [INFO]: {'dataset_name': 'CASIA-B', 'dataset_root': '/Volumes/Data/person_orientation_demo/datasets/processed/CASIA-B-pkl', 'num_workers': 0, 'dataset_partition': '/Volumes/Data/person_orientation_demo/external/OpenGait/datasets/CASIA-B/CASIA-B.json', 'remove_no_gallery': False, 'cache': False, 'test_dataset_name': 'CASIA-B'}
[2026-04-01 03:15:23] [INFO]: -------- Train Pid List --------
[2026-04-01 03:15:23] [INFO]: [001, 002, ..., 074]
[2026-04-01 03:15:25] [INFO]: {'lr': 0.1, 'momentum': 0.9, 'solver': 'SGD', 'weight_decay': 0.0005}
[2026-04-01 03:15:25] [INFO]: {'gamma': 0.1, 'milestones': [1], 'scheduler': 'MultiStepLR'}
[2026-04-01 03:15:25] [INFO]: Iteration 00001, Cost 3.50s, triplet_loss=0.1950, triplet_hard_loss=0.2000, triplet_loss_num=16.0000, triplet_mean_dist=0.0055
```
