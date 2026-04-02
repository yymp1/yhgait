# OpenGait Smoke Test 报告

- Python: `/home/bb/gait/yhgait/.venvs/opengait-gpu-cu118/bin/python`
- OpenGait: `/home/bb/gait/yhgait/external/OpenGait`
- 配置: `/home/bb/gait/yhgait/configs/opengait_casiab_smoke.yaml`
- 训练入口端口: `29531`

## 依赖检查
- 状态：`passed`
- `torch`: `ok` 2.5.1+cu118
- `yaml`: `ok` 6.0.3
- `tensorboard`: `ok` 2.20.0
- `torchvision`: `ok` 0.20.1+cu118
- `einops`: `ok` 0.8.2
- `kornia`: `ok` 0.8.2
- `matplotlib`: `ok` 3.10.8
- `imageio`: `ok` 2.37.3
- `sklearn`: `ok` 1.7.2
- `cv2`: `ok` 4.13.0
- `numpy`: `ok` 2.2.6
- `tqdm`: `ok` 4.67.3

## Pretreatment 产物检查
- 状态：`passed`
- 数据根目录：`/home/bb/gait/yhgait/datasets/processed/CASIA-B-pkl`
- subject 数：`3`
- type 数：`1`
- view 数：`1`
- pkl 数：`3`
- 抽样：
  - `/home/bb/gait/yhgait/datasets/processed/CASIA-B-pkl/001/nm-01/000/000.pkl` shape=[16, 64, 64] dtype=uint8
  - `/home/bb/gait/yhgait/datasets/processed/CASIA-B-pkl/002/nm-01/000/000.pkl` shape=[18, 64, 64] dtype=uint8
  - `/home/bb/gait/yhgait/datasets/processed/CASIA-B-pkl/075/nm-01/000/000.pkl` shape=[20, 64, 64] dtype=uint8

## 数据集 Smoke
- 状态：`passed`
- train 序列数：`2`
- test 序列数：`1`
- train 身份数：`2`
- 样例序列：`['001', 'nm-01', '000', ['../../datasets/processed/CASIA-B-pkl/001/nm-01/000/000.pkl']]`
- batch 形状：`[[[8, 64, 64], [8, 64, 64]]]`

## 官方训练入口 Smoke
- 状态：`passed`
- 返回码：`0`

```bash
/home/bb/gait/yhgait/.venvs/opengait-gpu-cu118/bin/python opengait/main.py --cfgs /home/bb/gait/yhgait/configs/opengait_casiab_smoke.yaml --phase train --log_to_file
```

```text
xFormers not available
xFormers not available
[2026-04-01 14:35:40] [INFO]: {'find_unused_parameters': False, 'enable_float16': False, 'with_test': False, 'fix_BN': False, 'log_iter': 1, 'restore_ckpt_strict': True, 'optimizer_reset': False, 'scheduler_reset': False, 'restore_hint': 0, 'save_iter': 1, 'save_name': 'smoke_gaitset', 'sync_BN': False, 'total_iter': 1, 'sampler': {'batch_shuffle': False, 'batch_size': [2, 2], 'frames_num_fixed': 8, 'frames_num_max': 8, 'frames_num_min': 8, 'sample_type': 'fixed_unordered', 'type': 'TripletSampler'}, 'transform': [{'type': 'BaseSilCuttingTransform'}]}
[2026-04-01 14:35:40] [INFO]: {'model': 'GaitSet', 'in_channels': [1, 32, 64, 128], 'SeparateFCs': {'in_channels': 128, 'out_channels': 256, 'parts_num': 62}, 'bin_num': [16, 8, 4, 2, 1]}
[2026-04-01 14:35:40] [INFO]: {'dataset_name': 'CASIA-B', 'dataset_root': '../../datasets/processed/CASIA-B-pkl', 'num_workers': 0, 'dataset_partition': './datasets/CASIA-B/CASIA-B.json', 'remove_no_gallery': False, 'cache': False, 'test_dataset_name': 'CASIA-B'}
[2026-04-01 14:35:40] [INFO]: -------- Train Pid List --------
[2026-04-01 14:35:40] [INFO]: ['001', '002']
[2026-04-01 14:35:41] [INFO]: {'lr': 0.1, 'momentum': 0.9, 'solver': 'SGD', 'weight_decay': 0.0005}
[2026-04-01 14:35:41] [INFO]: {'gamma': 0.1, 'milestones': [1], 'scheduler': 'MultiStepLR'}
[2026-04-01 14:35:41] [INFO]: Parameters Count: 2.59459M
[2026-04-01 14:35:41] [INFO]: Model Initialization Finished!
[2026-04-01 14:35:42] [INFO]: Iteration 00001, Cost 3.72s, triplet_loss=0.1975, triplet_hard_loss=0.2034, triplet_loss_num=16.0000, triplet_mean_dist=0.0042
[rank0]:[W401 14:35:44.142290276 ProcessGroupNCCL.cpp:1250] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())
```
