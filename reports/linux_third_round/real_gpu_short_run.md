# OpenGait GPU Probe 报告

- 状态：`passed`
- 模式：`short-run`
- Python：`/home/bb/gait/yhgait/.venvs/opengait-gpu/bin/python3.10`
- OpenGait：`/home/bb/gait/yhgait/external/OpenGait`
- 基础配置：`/home/bb/gait/yhgait/configs/opengait_casiab_smoke.yaml`
- 运行配置：`/home/bb/gait/yhgait/reports/linux_third_round/real_gpu_short_run.yaml`
- dataset_root：`/home/bb/gait/yhgait/datasets/processed/CASIA-B-pkl`
- dataset_partition：`/home/bb/gait/yhgait/external/OpenGait/datasets/CASIA-B/CASIA-B.json`
- save_name：`real_gpu_short_run_5iter`
- total_iter：`5`
- num_workers：`0`
- master_port：`29541`
- returncode：`0`
- 输出目录：`/home/bb/gait/yhgait/external/OpenGait/output/CASIA-B/GaitSet/real_gpu_short_run_5iter`
- checkpoint 数：`5`

## 实际命令

```bash
/home/bb/gait/yhgait/.venvs/opengait-gpu/bin/python3.10 opengait/main.py --cfgs /home/bb/gait/yhgait/reports/linux_third_round/real_gpu_short_run.yaml --phase train --log_to_file
```

## 产物

- checkpoint: `/home/bb/gait/yhgait/external/OpenGait/output/CASIA-B/GaitSet/real_gpu_short_run_5iter/checkpoints/real_gpu_short_run_5iter-00001.pt`
- checkpoint: `/home/bb/gait/yhgait/external/OpenGait/output/CASIA-B/GaitSet/real_gpu_short_run_5iter/checkpoints/real_gpu_short_run_5iter-00002.pt`
- checkpoint: `/home/bb/gait/yhgait/external/OpenGait/output/CASIA-B/GaitSet/real_gpu_short_run_5iter/checkpoints/real_gpu_short_run_5iter-00003.pt`
- checkpoint: `/home/bb/gait/yhgait/external/OpenGait/output/CASIA-B/GaitSet/real_gpu_short_run_5iter/checkpoints/real_gpu_short_run_5iter-00004.pt`
- checkpoint: `/home/bb/gait/yhgait/external/OpenGait/output/CASIA-B/GaitSet/real_gpu_short_run_5iter/checkpoints/real_gpu_short_run_5iter-00005.pt`
- log: `/home/bb/gait/yhgait/external/OpenGait/output/CASIA-B/GaitSet/real_gpu_short_run_5iter/logs/2026-04-01-17-22-18.txt`
- summary: `/home/bb/gait/yhgait/external/OpenGait/output/CASIA-B/GaitSet/real_gpu_short_run_5iter/summary/events.out.tfevents.1775035338.bb-slam.75577.0`

## stdout tail

```text
(empty)
```

## stderr tail

```text
xFormers not available
xFormers not available
[2026-04-01 17:22:18] [INFO]: {'find_unused_parameters': False, 'enable_float16': False, 'with_test': False, 'fix_BN': False, 'log_iter': 1, 'restore_ckpt_strict': True, 'optimizer_reset': False, 'scheduler_reset': False, 'restore_hint': 0, 'save_iter': 1, 'save_name': 'real_gpu_short_run_5iter', 'sync_BN': False, 'total_iter': 5, 'sampler': {'batch_shuffle': False, 'batch_size': [2, 2], 'frames_num_fixed': 8, 'frames_num_max': 8, 'frames_num_min': 8, 'sample_type': 'fixed_unordered', 'type': 'TripletSampler'}, 'transform': [{'type': 'BaseSilCuttingTransform'}]}
[2026-04-01 17:22:18] [INFO]: {'model': 'GaitSet', 'in_channels': [1, 32, 64, 128], 'SeparateFCs': {'in_channels': 128, 'out_channels': 256, 'parts_num': 62}, 'bin_num': [16, 8, 4, 2, 1]}
[2026-04-01 17:22:18] [INFO]: {'dataset_name': 'CASIA-B', 'dataset_root': '/home/bb/gait/yhgait/datasets/processed/CASIA-B-pkl', 'num_workers': 0, 'dataset_partition': '/home/bb/gait/yhgait/external/OpenGait/datasets/CASIA-B/CASIA-B.json', 'remove_no_gallery': False, 'cache': False, 'test_dataset_name': 'CASIA-B'}
[2026-04-01 17:22:18] [INFO]: -------- Train Pid List --------
[2026-04-01 17:22:18] [INFO]: [001, 002, ..., 074]
[2026-04-01 17:22:19] [INFO]: {'lr': 0.1, 'momentum': 0.9, 'solver': 'SGD', 'weight_decay': 0.0005}
[2026-04-01 17:22:19] [INFO]: {'gamma': 0.1, 'milestones': [1], 'scheduler': 'MultiStepLR'}
[2026-04-01 17:22:19] [INFO]: Parameters Count: 2.59459M
[2026-04-01 17:22:19] [INFO]: Model Initialization Finished!
[2026-04-01 17:22:20] [INFO]: Iteration 00001, Cost 3.21s, triplet_loss=0.1949, triplet_hard_loss=0.2099, triplet_loss_num=16.0000, triplet_mean_dist=0.0108
[2026-04-01 17:22:20] [INFO]: Iteration 00002, Cost 0.06s, triplet_loss=0.1937, triplet_hard_loss=0.2067, triplet_loss_num=16.0000, triplet_mean_dist=0.0106
[2026-04-01 17:22:20] [INFO]: Iteration 00003, Cost 0.05s, triplet_loss=0.1909, triplet_hard_loss=0.2023, triplet_loss_num=16.0000, triplet_mean_dist=0.0114
[2026-04-01 17:22:20] [INFO]: Iteration 00004, Cost 0.05s, triplet_loss=0.1912, triplet_hard_loss=0.1981, triplet_loss_num=16.0000, triplet_mean_dist=0.0085
[2026-04-01 17:22:20] [INFO]: Iteration 00005, Cost 0.05s, triplet_loss=0.1852, triplet_hard_loss=0.1919, triplet_loss_num=16.0000, triplet_mean_dist=0.0109
[rank0]:[W401 17:22:21.803200616 ProcessGroupNCCL.cpp:1250] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())

```
