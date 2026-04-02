# OpenGait GPU Probe 报告

- 状态：`passed`
- 模式：`baseline`
- Python：`/home/bb/gait/yhgait/.venvs/opengait-gpu/bin/python3.10`
- OpenGait：`/home/bb/gait/yhgait/external/OpenGait`
- 基础配置：`/home/bb/gait/yhgait/configs/opengait_casiab_baseline_small.yaml`
- 运行配置：`/home/bb/gait/yhgait/reports/linux_third_round/baseline_midrun_50.yaml`
- dataset_root：`/home/bb/gait/yhgait/datasets/processed/CASIA-B-pkl`
- dataset_partition：`/home/bb/gait/yhgait/external/OpenGait/datasets/CASIA-B/CASIA-B.json`
- save_name：`baseline_small_real`
- total_iter：`50`
- num_workers：`1`
- restore_hint：`25`
- master_port：`29541`
- returncode：`0`
- 耗时：`8.7` 秒
- 输出目录：`/home/bb/gait/yhgait/external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real`
- checkpoint 数：`10`

## 实际命令

```bash
/home/bb/gait/yhgait/.venvs/opengait-gpu/bin/python3.10 opengait/main.py --cfgs /home/bb/gait/yhgait/reports/linux_third_round/baseline_midrun_50.yaml --phase train --log_to_file
```

## 产物

- checkpoint: `/home/bb/gait/yhgait/external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real/checkpoints/baseline_small_real-00005.pt`
- checkpoint: `/home/bb/gait/yhgait/external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real/checkpoints/baseline_small_real-00010.pt`
- checkpoint: `/home/bb/gait/yhgait/external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real/checkpoints/baseline_small_real-00015.pt`
- checkpoint: `/home/bb/gait/yhgait/external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real/checkpoints/baseline_small_real-00020.pt`
- checkpoint: `/home/bb/gait/yhgait/external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real/checkpoints/baseline_small_real-00025.pt`
- checkpoint: `/home/bb/gait/yhgait/external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real/checkpoints/baseline_small_real-00030.pt`
- checkpoint: `/home/bb/gait/yhgait/external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real/checkpoints/baseline_small_real-00035.pt`
- checkpoint: `/home/bb/gait/yhgait/external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real/checkpoints/baseline_small_real-00040.pt`
- checkpoint: `/home/bb/gait/yhgait/external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real/checkpoints/baseline_small_real-00045.pt`
- checkpoint: `/home/bb/gait/yhgait/external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real/checkpoints/baseline_small_real-00050.pt`
- log: `/home/bb/gait/yhgait/external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real/logs/2026-04-01-18-07-37.txt`
- log: `/home/bb/gait/yhgait/external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real/logs/2026-04-01-18-10-41.txt`
- log: `/home/bb/gait/yhgait/external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real/logs/2026-04-01-18-24-41.txt`
- summary: `/home/bb/gait/yhgait/external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real/summary/events.out.tfevents.1775038057.bb-slam.86158.0`
- summary: `/home/bb/gait/yhgait/external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real/summary/events.out.tfevents.1775038241.bb-slam.86924.0`
- summary: `/home/bb/gait/yhgait/external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real/summary/events.out.tfevents.1775039081.bb-slam.90288.0`

## 最新迭代日志

- `[2026-04-01 18:24:44] [INFO]: Iteration 00031, Cost 0.08s, triplet_loss=0.1807, triplet_hard_loss=0.2333, triplet_loss_num=16.0000, triplet_mean_dist=0.0329`
- `[2026-04-01 18:24:44] [INFO]: Iteration 00032, Cost 0.06s, triplet_loss=0.1721, triplet_hard_loss=0.2460, triplet_loss_num=15.8387, triplet_mean_dist=0.0399`
- `[2026-04-01 18:24:44] [INFO]: Iteration 00033, Cost 0.06s, triplet_loss=0.1785, triplet_hard_loss=0.2314, triplet_loss_num=16.0000, triplet_mean_dist=0.0340`
- `[2026-04-01 18:24:44] [INFO]: Iteration 00034, Cost 0.05s, triplet_loss=0.1871, triplet_hard_loss=0.2210, triplet_loss_num=16.0000, triplet_mean_dist=0.0225`
- `[2026-04-01 18:24:44] [INFO]: Iteration 00035, Cost 0.05s, triplet_loss=0.1800, triplet_hard_loss=0.2359, triplet_loss_num=16.0000, triplet_mean_dist=0.0298`
- `[2026-04-01 18:24:44] [INFO]: Iteration 00036, Cost 0.08s, triplet_loss=0.1620, triplet_hard_loss=0.2330, triplet_loss_num=15.3226, triplet_mean_dist=0.0452`
- `[2026-04-01 18:24:44] [INFO]: Iteration 00037, Cost 0.05s, triplet_loss=0.1720, triplet_hard_loss=0.2335, triplet_loss_num=15.5161, triplet_mean_dist=0.0379`
- `[2026-04-01 18:24:44] [INFO]: Iteration 00038, Cost 0.05s, triplet_loss=0.1796, triplet_hard_loss=0.2315, triplet_loss_num=16.0000, triplet_mean_dist=0.0333`
- `[2026-04-01 18:24:44] [INFO]: Iteration 00039, Cost 0.05s, triplet_loss=0.1855, triplet_hard_loss=0.2483, triplet_loss_num=15.6774, triplet_mean_dist=0.0335`
- `[2026-04-01 18:24:44] [INFO]: Iteration 00040, Cost 0.05s, triplet_loss=0.1784, triplet_hard_loss=0.2303, triplet_loss_num=15.8710, triplet_mean_dist=0.0322`
- `[2026-04-01 18:24:44] [INFO]: Iteration 00041, Cost 0.08s, triplet_loss=0.1758, triplet_hard_loss=0.2641, triplet_loss_num=14.9355, triplet_mean_dist=0.0516`
- `[2026-04-01 18:24:44] [INFO]: Iteration 00042, Cost 0.05s, triplet_loss=0.1701, triplet_hard_loss=0.1948, triplet_loss_num=16.0000, triplet_mean_dist=0.0260`
- `[2026-04-01 18:24:44] [INFO]: Iteration 00043, Cost 0.05s, triplet_loss=0.1849, triplet_hard_loss=0.2610, triplet_loss_num=15.8710, triplet_mean_dist=0.0372`
- `[2026-04-01 18:24:45] [INFO]: Iteration 00044, Cost 0.05s, triplet_loss=0.1736, triplet_hard_loss=0.2492, triplet_loss_num=15.1774, triplet_mean_dist=0.0442`
- `[2026-04-01 18:24:45] [INFO]: Iteration 00045, Cost 0.05s, triplet_loss=0.1852, triplet_hard_loss=0.2657, triplet_loss_num=15.8387, triplet_mean_dist=0.0447`
- `[2026-04-01 18:24:45] [INFO]: Iteration 00046, Cost 0.08s, triplet_loss=0.1653, triplet_hard_loss=0.2232, triplet_loss_num=15.8387, triplet_mean_dist=0.0372`
- `[2026-04-01 18:24:45] [INFO]: Iteration 00047, Cost 0.05s, triplet_loss=0.1804, triplet_hard_loss=0.2049, triplet_loss_num=16.0000, triplet_mean_dist=0.0204`
- `[2026-04-01 18:24:45] [INFO]: Iteration 00048, Cost 0.05s, triplet_loss=0.1750, triplet_hard_loss=0.2459, triplet_loss_num=16.0000, triplet_mean_dist=0.0410`
- `[2026-04-01 18:24:45] [INFO]: Iteration 00049, Cost 0.05s, triplet_loss=0.1777, triplet_hard_loss=0.2233, triplet_loss_num=15.9677, triplet_mean_dist=0.0301`
- `[2026-04-01 18:24:45] [INFO]: Iteration 00050, Cost 0.06s, triplet_loss=0.1919, triplet_hard_loss=0.2570, triplet_loss_num=15.6129, triplet_mean_dist=0.0437`

## stdout tail

```text
(empty)
```

## stderr tail

```text
xFormers not available
xFormers not available
[2026-04-01 18:24:41] [INFO]: {'find_unused_parameters': False, 'enable_float16': False, 'with_test': False, 'fix_BN': False, 'log_iter': 1, 'restore_ckpt_strict': True, 'optimizer_reset': False, 'scheduler_reset': False, 'restore_hint': 25, 'save_iter': 5, 'save_name': 'baseline_small_real', 'sync_BN': False, 'total_iter': 50, 'sampler': {'batch_shuffle': False, 'batch_size': [2, 2], 'frames_num_fixed': 16, 'frames_num_max': 16, 'frames_num_min': 16, 'sample_type': 'fixed_unordered', 'type': 'TripletSampler'}, 'transform': [{'type': 'BaseSilCuttingTransform'}]}
[2026-04-01 18:24:41] [INFO]: {'model': 'GaitSet', 'in_channels': [1, 32, 64, 128], 'SeparateFCs': {'in_channels': 128, 'out_channels': 256, 'parts_num': 62}, 'bin_num': [16, 8, 4, 2, 1]}
[2026-04-01 18:24:41] [INFO]: {'dataset_name': 'CASIA-B', 'dataset_root': '/home/bb/gait/yhgait/datasets/processed/CASIA-B-pkl', 'num_workers': 1, 'dataset_partition': '/home/bb/gait/yhgait/external/OpenGait/datasets/CASIA-B/CASIA-B.json', 'remove_no_gallery': False, 'cache': False, 'test_dataset_name': 'CASIA-B'}
[2026-04-01 18:24:41] [INFO]: -------- Train Pid List --------
[2026-04-01 18:24:41] [INFO]: [001, 002, ..., 074]
[2026-04-01 18:24:42] [INFO]: {'lr': 0.1, 'momentum': 0.9, 'solver': 'SGD', 'weight_decay': 0.0005}
[2026-04-01 18:24:42] [INFO]: {'gamma': 0.1, 'milestones': [10, 15], 'scheduler': 'MultiStepLR'}
[2026-04-01 18:24:42] [INFO]: Restore Parameters from output/CASIA-B/GaitSet/baseline_small_real/checkpoints/baseline_small_real-00025.pt !!!
[2026-04-01 18:24:42] [INFO]: Parameters Count: 2.59459M
[2026-04-01 18:24:42] [INFO]: Model Initialization Finished!
[2026-04-01 18:24:43] [INFO]: Iteration 00026, Cost 3.46s, triplet_loss=0.1875, triplet_hard_loss=0.2589, triplet_loss_num=15.8065, triplet_mean_dist=0.0417
[2026-04-01 18:24:44] [INFO]: Iteration 00027, Cost 0.06s, triplet_loss=0.1797, triplet_hard_loss=0.2430, triplet_loss_num=16.0000, triplet_mean_dist=0.0397
[2026-04-01 18:24:44] [INFO]: Iteration 00028, Cost 0.05s, triplet_loss=0.1606, triplet_hard_loss=0.2057, triplet_loss_num=16.0000, triplet_mean_dist=0.0406
[2026-04-01 18:24:44] [INFO]: Iteration 00029, Cost 0.06s, triplet_loss=0.1454, triplet_hard_loss=0.1764, triplet_loss_num=16.0000, triplet_mean_dist=0.0387
[2026-04-01 18:24:44] [INFO]: Iteration 00030, Cost 0.06s, triplet_loss=0.1294, triplet_hard_loss=0.1479, triplet_loss_num=14.3710, triplet_mean_dist=0.0467
[2026-04-01 18:24:44] [INFO]: Iteration 00031, Cost 0.08s, triplet_loss=0.1807, triplet_hard_loss=0.2333, triplet_loss_num=16.0000, triplet_mean_dist=0.0329
[2026-04-01 18:24:44] [INFO]: Iteration 00032, Cost 0.06s, triplet_loss=0.1721, triplet_hard_loss=0.2460, triplet_loss_num=15.8387, triplet_mean_dist=0.0399
[2026-04-01 18:24:44] [INFO]: Iteration 00033, Cost 0.06s, triplet_loss=0.1785, triplet_hard_loss=0.2314, triplet_loss_num=16.0000, triplet_mean_dist=0.0340
[2026-04-01 18:24:44] [INFO]: Iteration 00034, Cost 0.05s, triplet_loss=0.1871, triplet_hard_loss=0.2210, triplet_loss_num=16.0000, triplet_mean_dist=0.0225
[2026-04-01 18:24:44] [INFO]: Iteration 00035, Cost 0.05s, triplet_loss=0.1800, triplet_hard_loss=0.2359, triplet_loss_num=16.0000, triplet_mean_dist=0.0298
[2026-04-01 18:24:44] [INFO]: Iteration 00036, Cost 0.08s, triplet_loss=0.1620, triplet_hard_loss=0.2330, triplet_loss_num=15.3226, triplet_mean_dist=0.0452
[2026-04-01 18:24:44] [INFO]: Iteration 00037, Cost 0.05s, triplet_loss=0.1720, triplet_hard_loss=0.2335, triplet_loss_num=15.5161, triplet_mean_dist=0.0379
[2026-04-01 18:24:44] [INFO]: Iteration 00038, Cost 0.05s, triplet_loss=0.1796, triplet_hard_loss=0.2315, triplet_loss_num=16.0000, triplet_mean_dist=0.0333
[2026-04-01 18:24:44] [INFO]: Iteration 00039, Cost 0.05s, triplet_loss=0.1855, triplet_hard_loss=0.2483, triplet_loss_num=15.6774, triplet_mean_dist=0.0335
[2026-04-01 18:24:44] [INFO]: Iteration 00040, Cost 0.05s, triplet_loss=0.1784, triplet_hard_loss=0.2303, triplet_loss_num=15.8710, triplet_mean_dist=0.0322
[2026-04-01 18:24:44] [INFO]: Iteration 00041, Cost 0.08s, triplet_loss=0.1758, triplet_hard_loss=0.2641, triplet_loss_num=14.9355, triplet_mean_dist=0.0516
[2026-04-01 18:24:44] [INFO]: Iteration 00042, Cost 0.05s, triplet_loss=0.1701, triplet_hard_loss=0.1948, triplet_loss_num=16.0000, triplet_mean_dist=0.0260
[2026-04-01 18:24:44] [INFO]: Iteration 00043, Cost 0.05s, triplet_loss=0.1849, triplet_hard_loss=0.2610, triplet_loss_num=15.8710, triplet_mean_dist=0.0372
[2026-04-01 18:24:45] [INFO]: Iteration 00044, Cost 0.05s, triplet_loss=0.1736, triplet_hard_loss=0.2492, triplet_loss_num=15.1774, triplet_mean_dist=0.0442
[2026-04-01 18:24:45] [INFO]: Iteration 00045, Cost 0.05s, triplet_loss=0.1852, triplet_hard_loss=0.2657, triplet_loss_num=15.8387, triplet_mean_dist=0.0447
[2026-04-01 18:24:45] [INFO]: Iteration 00046, Cost 0.08s, triplet_loss=0.1653, triplet_hard_loss=0.2232, triplet_loss_num=15.8387, triplet_mean_dist=0.0372
[2026-04-01 18:24:45] [INFO]: Iteration 00047, Cost 0.05s, triplet_loss=0.1804, triplet_hard_loss=0.2049, triplet_loss_num=16.0000, triplet_mean_dist=0.0204
[2026-04-01 18:24:45] [INFO]: Iteration 00048, Cost 0.05s, triplet_loss=0.1750, triplet_hard_loss=0.2459, triplet_loss_num=16.0000, triplet_mean_dist=0.0410
[2026-04-01 18:24:45] [INFO]: Iteration 00049, Cost 0.05s, triplet_loss=0.1777, triplet_hard_loss=0.2233, triplet_loss_num=15.9677, triplet_mean_dist=0.0301
[2026-04-01 18:24:45] [INFO]: Iteration 00050, Cost 0.06s, triplet_loss=0.1919, triplet_hard_loss=0.2570, triplet_loss_num=15.6129, triplet_mean_dist=0.0437
[rank0]:[W401 18:24:45.372862785 ProcessGroupNCCL.cpp:1250] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())

```
