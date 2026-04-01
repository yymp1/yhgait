# OpenGait CPU 兼容探针报告

- 模式：`short-run`
- 状态：`passed`
- 迭代数：`2`
- save_name：`cpu_probe_short_run_2step`
- 输出目录：`/Volumes/Data/person_orientation_demo/external/OpenGait/output/CASIA-B/GaitSet/cpu_probe_short_run_2step`
- 实际迭代：`2`
- checkpoint 数：`2`
- 运行耗时：`11.73` 秒
- 清理的 `._*` 数量：`7`

## 说明
- 这是本机 CPU/gloo 兼容探针，不是官方支持的 CUDA/NCCL 正式训练入口。
- 探针通过 monkeypatch 绕过了 OpenGait 当前版本里硬编码的 cuda/ddp 包装逻辑。

## 产物
- checkpoints：`['/Volumes/Data/person_orientation_demo/external/OpenGait/output/CASIA-B/GaitSet/cpu_probe_short_run_2step/checkpoints/cpu_probe_short_run_2step-00001.pt', '/Volumes/Data/person_orientation_demo/external/OpenGait/output/CASIA-B/GaitSet/cpu_probe_short_run_2step/checkpoints/cpu_probe_short_run_2step-00002.pt']`
- logs：`['/Volumes/Data/person_orientation_demo/external/OpenGait/output/CASIA-B/GaitSet/cpu_probe_short_run_2step/logs/2026-04-01-03-20-43.txt']`
- summary：`['/Volumes/Data/person_orientation_demo/external/OpenGait/output/CASIA-B/GaitSet/cpu_probe_short_run_2step/summary/events.out.tfevents.1774984843.yympdeMac-mini.local.95614.0']`

## 最新日志尾部
```text
[2026-04-01 03:20:43] [INFO]: {'dataset_name': 'CASIA-B', 'dataset_root': '/Volumes/Data/person_orientation_demo/datasets/processed/CASIA-B-pkl', 'num_workers': 0, 'dataset_partition': '/Volumes/Data/person_orientation_demo/external/OpenGait/datasets/CASIA-B/CASIA-B.json', 'remove_no_gallery': False, 'cache': False, 'test_dataset_name': 'CASIA-B'}
[2026-04-01 03:20:43] [INFO]: -------- Train Pid List --------
[2026-04-01 03:20:43] [INFO]: [001, 002, ..., 074]
[2026-04-01 03:20:52] [INFO]: {'lr': 0.1, 'momentum': 0.9, 'solver': 'SGD', 'weight_decay': 0.0005}
[2026-04-01 03:20:52] [INFO]: {'gamma': 0.1, 'milestones': [1], 'scheduler': 'MultiStepLR'}
[2026-04-01 03:20:52] [INFO]: Iteration 00001, Cost 10.78s, triplet_loss=0.1943, triplet_hard_loss=0.2080, triplet_loss_num=16.0000, triplet_mean_dist=0.0105
[2026-04-01 03:20:52] [INFO]: Iteration 00002, Cost 0.17s, triplet_loss=0.1945, triplet_hard_loss=0.2066, triplet_loss_num=16.0000, triplet_mean_dist=0.0089
```
