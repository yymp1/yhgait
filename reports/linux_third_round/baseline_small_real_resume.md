# OpenGait GPU Probe 报告

- 状态：`passed`
- 模式：`baseline`
- Python：`/home/bb/gait/yhgait/.venvs/opengait-gpu/bin/python3.10`
- OpenGait：`/home/bb/gait/yhgait/external/OpenGait`
- 基础配置：`/home/bb/gait/yhgait/configs/opengait_casiab_baseline_small.yaml`
- 运行配置：`/home/bb/gait/yhgait/reports/linux_third_round/baseline_small_real_resume.yaml`
- dataset_root：`/home/bb/gait/yhgait/datasets/processed/CASIA-B-pkl`
- dataset_partition：`/home/bb/gait/yhgait/external/OpenGait/datasets/CASIA-B/CASIA-B.json`
- save_name：`baseline_small_real`
- total_iter：`25`
- num_workers：`1`
- restore_hint：`20`
- master_port：`29541`
- returncode：`0`
- 耗时：`7.72` 秒
- 输出目录：`/home/bb/gait/yhgait/external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real`
- checkpoint 数：`5`

## 实际命令

```bash
/home/bb/gait/yhgait/.venvs/opengait-gpu/bin/python3.10 opengait/main.py --cfgs /home/bb/gait/yhgait/reports/linux_third_round/baseline_small_real_resume.yaml --phase train --log_to_file
```

## 产物

- checkpoint: `/home/bb/gait/yhgait/external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real/checkpoints/baseline_small_real-00005.pt`
- checkpoint: `/home/bb/gait/yhgait/external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real/checkpoints/baseline_small_real-00010.pt`
- checkpoint: `/home/bb/gait/yhgait/external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real/checkpoints/baseline_small_real-00015.pt`
- checkpoint: `/home/bb/gait/yhgait/external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real/checkpoints/baseline_small_real-00020.pt`
- checkpoint: `/home/bb/gait/yhgait/external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real/checkpoints/baseline_small_real-00025.pt`
- log: `/home/bb/gait/yhgait/external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real/logs/2026-04-01-18-07-37.txt`
- log: `/home/bb/gait/yhgait/external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real/logs/2026-04-01-18-10-41.txt`
- summary: `/home/bb/gait/yhgait/external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real/summary/events.out.tfevents.1775038057.bb-slam.86158.0`
- summary: `/home/bb/gait/yhgait/external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real/summary/events.out.tfevents.1775038241.bb-slam.86924.0`

## 最新迭代日志

- `[2026-04-01 18:10:44] [INFO]: Iteration 00021, Cost 3.64s, triplet_loss=0.1874, triplet_hard_loss=0.2573, triplet_loss_num=15.8387, triplet_mean_dist=0.0408`
- `[2026-04-01 18:10:44] [INFO]: Iteration 00022, Cost 0.06s, triplet_loss=0.1801, triplet_hard_loss=0.2418, triplet_loss_num=16.0000, triplet_mean_dist=0.0387`
- `[2026-04-01 18:10:44] [INFO]: Iteration 00023, Cost 0.06s, triplet_loss=0.1616, triplet_hard_loss=0.2055, triplet_loss_num=16.0000, triplet_mean_dist=0.0395`
- `[2026-04-01 18:10:44] [INFO]: Iteration 00024, Cost 0.06s, triplet_loss=0.1469, triplet_hard_loss=0.1769, triplet_loss_num=16.0000, triplet_mean_dist=0.0376`
- `[2026-04-01 18:10:44] [INFO]: Iteration 00025, Cost 0.06s, triplet_loss=0.1308, triplet_hard_loss=0.1493, triplet_loss_num=14.5806, triplet_mean_dist=0.0455`

## stdout tail

```text
(empty)
```

## stderr tail

```text
xFormers not available
xFormers not available
[2026-04-01 18:10:41] [INFO]: {'find_unused_parameters': False, 'enable_float16': False, 'with_test': False, 'fix_BN': False, 'log_iter': 1, 'restore_ckpt_strict': True, 'optimizer_reset': False, 'scheduler_reset': False, 'restore_hint': 20, 'save_iter': 5, 'save_name': 'baseline_small_real', 'sync_BN': False, 'total_iter': 25, 'sampler': {'batch_shuffle': False, 'batch_size': [2, 2], 'frames_num_fixed': 16, 'frames_num_max': 16, 'frames_num_min': 16, 'sample_type': 'fixed_unordered', 'type': 'TripletSampler'}, 'transform': [{'type': 'BaseSilCuttingTransform'}]}
[2026-04-01 18:10:41] [INFO]: {'model': 'GaitSet', 'in_channels': [1, 32, 64, 128], 'SeparateFCs': {'in_channels': 128, 'out_channels': 256, 'parts_num': 62}, 'bin_num': [16, 8, 4, 2, 1]}
[2026-04-01 18:10:41] [INFO]: {'dataset_name': 'CASIA-B', 'dataset_root': '/home/bb/gait/yhgait/datasets/processed/CASIA-B-pkl', 'num_workers': 1, 'dataset_partition': '/home/bb/gait/yhgait/external/OpenGait/datasets/CASIA-B/CASIA-B.json', 'remove_no_gallery': False, 'cache': False, 'test_dataset_name': 'CASIA-B'}
[2026-04-01 18:10:41] [INFO]: -------- Train Pid List --------
[2026-04-01 18:10:41] [INFO]: [001, 002, ..., 074]
[2026-04-01 18:10:43] [INFO]: {'lr': 0.1, 'momentum': 0.9, 'solver': 'SGD', 'weight_decay': 0.0005}
[2026-04-01 18:10:43] [INFO]: {'gamma': 0.1, 'milestones': [10, 15], 'scheduler': 'MultiStepLR'}
[2026-04-01 18:10:43] [INFO]: Restore Parameters from output/CASIA-B/GaitSet/baseline_small_real/checkpoints/baseline_small_real-00020.pt !!!
[2026-04-01 18:10:43] [INFO]: Parameters Count: 2.59459M
[2026-04-01 18:10:43] [INFO]: Model Initialization Finished!
[2026-04-01 18:10:44] [INFO]: Iteration 00021, Cost 3.64s, triplet_loss=0.1874, triplet_hard_loss=0.2573, triplet_loss_num=15.8387, triplet_mean_dist=0.0408
[2026-04-01 18:10:44] [INFO]: Iteration 00022, Cost 0.06s, triplet_loss=0.1801, triplet_hard_loss=0.2418, triplet_loss_num=16.0000, triplet_mean_dist=0.0387
[2026-04-01 18:10:44] [INFO]: Iteration 00023, Cost 0.06s, triplet_loss=0.1616, triplet_hard_loss=0.2055, triplet_loss_num=16.0000, triplet_mean_dist=0.0395
[2026-04-01 18:10:44] [INFO]: Iteration 00024, Cost 0.06s, triplet_loss=0.1469, triplet_hard_loss=0.1769, triplet_loss_num=16.0000, triplet_mean_dist=0.0376
[2026-04-01 18:10:44] [INFO]: Iteration 00025, Cost 0.06s, triplet_loss=0.1308, triplet_hard_loss=0.1493, triplet_loss_num=14.5806, triplet_mean_dist=0.0455
[rank0]:[W401 18:10:45.782877847 ProcessGroupNCCL.cpp:1250] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())

```
