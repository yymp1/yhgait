# OpenGait GPU Probe 报告

- 状态：`passed`
- 模式：`baseline`
- Python：`/home/bb/gait/yhgait/.venvs/opengait-gpu/bin/python`
- OpenGait：`/home/bb/gait/yhgait/external/OpenGait`
- 基础配置：`/home/bb/gait/yhgait/configs/opengait_casiab_formal_conservative.yaml`
- 运行配置：`/home/bb/gait/yhgait/reports/formal_conservative/formal_conservative_real.yaml`
- dataset_root：`/home/bb/gait/yhgait/datasets/processed/CASIA-B-pkl`
- dataset_partition：`/home/bb/gait/yhgait/external/OpenGait/datasets/CASIA-B/CASIA-B.json`
- save_name：`formal_conservative_real`
- total_iter：`1000`
- num_workers：`1`
- restore_hint：`/home/bb/gait/yhgait/external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real/checkpoints/baseline_small_real-00500.pt`
- master_port：`29541`
- returncode：`0`
- 耗时：`63.07` 秒
- 输出目录：`/home/bb/gait/yhgait/external/OpenGait/output/CASIA-B/GaitSet/formal_conservative_real`
- checkpoint 数：`1`

## 实际命令

```bash
/home/bb/gait/yhgait/.venvs/opengait-gpu/bin/python opengait/main.py --cfgs /home/bb/gait/yhgait/reports/formal_conservative/formal_conservative_real.yaml --phase train --log_to_file
```

## 产物

- checkpoint: `/home/bb/gait/yhgait/external/OpenGait/output/CASIA-B/GaitSet/formal_conservative_real/checkpoints/formal_conservative_real-00500.pt`
- log: none
- summary: none

## 最新迭代日志

- none

## stdout tail

```text
(empty)
```

## stderr tail

```text
let_hard_loss=0.3771, triplet_loss_num=6.8548, triplet_mean_dist=0.2744
[2026-04-01 21:01:03] [INFO]: Iteration 00928, Cost 0.05s, triplet_loss=0.1638, triplet_hard_loss=0.3414, triplet_loss_num=8.9677, triplet_mean_dist=0.2152
[2026-04-01 21:01:03] [INFO]: Iteration 00929, Cost 0.05s, triplet_loss=0.1810, triplet_hard_loss=0.3007, triplet_loss_num=6.7419, triplet_mean_dist=0.2895
[2026-04-01 21:01:03] [INFO]: Iteration 00930, Cost 0.05s, triplet_loss=0.1628, triplet_hard_loss=0.2938, triplet_loss_num=7.5161, triplet_mean_dist=0.2807
[2026-04-01 21:01:03] [INFO]: Iteration 00931, Cost 0.07s, triplet_loss=0.1585, triplet_hard_loss=0.2657, triplet_loss_num=6.8387, triplet_mean_dist=0.2983
[2026-04-01 21:01:03] [INFO]: Iteration 00932, Cost 0.05s, triplet_loss=0.1688, triplet_hard_loss=0.3210, triplet_loss_num=6.2419, triplet_mean_dist=0.3573
[2026-04-01 21:01:03] [INFO]: Iteration 00933, Cost 0.05s, triplet_loss=0.2211, triplet_hard_loss=0.4594, triplet_loss_num=6.9839, triplet_mean_dist=0.2835
[2026-04-01 21:01:03] [INFO]: Iteration 00934, Cost 0.05s, triplet_loss=0.1753, triplet_hard_loss=0.3266, triplet_loss_num=6.3226, triplet_mean_dist=0.2755
[2026-04-01 21:01:04] [INFO]: Iteration 00935, Cost 0.05s, triplet_loss=0.1082, triplet_hard_loss=0.1886, triplet_loss_num=4.6452, triplet_mean_dist=0.3299
[2026-04-01 21:01:04] [INFO]: Iteration 00936, Cost 0.07s, triplet_loss=0.1477, triplet_hard_loss=0.2604, triplet_loss_num=4.8871, triplet_mean_dist=0.2630
[2026-04-01 21:01:04] [INFO]: Iteration 00937, Cost 0.05s, triplet_loss=0.1348, triplet_hard_loss=0.2721, triplet_loss_num=6.9677, triplet_mean_dist=0.2340
[2026-04-01 21:01:04] [INFO]: Iteration 00938, Cost 0.05s, triplet_loss=0.2137, triplet_hard_loss=0.4852, triplet_loss_num=8.8065, triplet_mean_dist=0.2181
[2026-04-01 21:01:04] [INFO]: Iteration 00939, Cost 0.05s, triplet_loss=0.1917, triplet_hard_loss=0.3621, triplet_loss_num=9.6290, triplet_mean_dist=0.2078
[2026-04-01 21:01:04] [INFO]: Iteration 00940, Cost 0.05s, triplet_loss=0.2413, triplet_hard_loss=0.4120, triplet_loss_num=7.3548, triplet_mean_dist=0.3093
[2026-04-01 21:01:04] [INFO]: Iteration 00941, Cost 0.07s, triplet_loss=0.2800, triplet_hard_loss=0.5237, triplet_loss_num=9.1129, triplet_mean_dist=0.3203
[2026-04-01 21:01:04] [INFO]: Iteration 00942, Cost 0.05s, triplet_loss=0.2239, triplet_hard_loss=0.3686, triplet_loss_num=8.0645, triplet_mean_dist=0.2749
[2026-04-01 21:01:04] [INFO]: Iteration 00943, Cost 0.05s, triplet_loss=0.2463, triplet_hard_loss=0.4663, triplet_loss_num=6.2903, triplet_mean_dist=0.3751
[2026-04-01 21:01:04] [INFO]: Iteration 00944, Cost 0.05s, triplet_loss=0.2201, triplet_hard_loss=0.4213, triplet_loss_num=8.3548, triplet_mean_dist=0.2872
[2026-04-01 21:01:04] [INFO]: Iteration 00945, Cost 0.05s, triplet_loss=0.1287, triplet_hard_loss=0.2160, triplet_loss_num=7.9355, triplet_mean_dist=0.2295
[2026-04-01 21:01:04] [INFO]: Iteration 00946, Cost 0.07s, triplet_loss=0.2408, triplet_hard_loss=0.4390, triplet_loss_num=8.8710, triplet_mean_dist=0.2604
[2026-04-01 21:01:04] [INFO]: Iteration 00947, Cost 0.05s, triplet_loss=0.3120, triplet_hard_loss=0.5278, triplet_loss_num=7.7419, triplet_mean_dist=0.3336
[2026-04-01 21:01:04] [INFO]: Iteration 00948, Cost 0.05s, triplet_loss=0.1597, triplet_hard_loss=0.2647, triplet_loss_num=8.1774, triplet_mean_dist=0.2107
[2026-04-01 21:01:04] [INFO]: Iteration 00949, Cost 0.05s, triplet_loss=0.1640, triplet_hard_loss=0.2753, triplet_loss_num=6.9194, triplet_mean_dist=0.2798
[2026-04-01 21:01:04] [INFO]: Iteration 00950, Cost 0.05s, triplet_loss=0.1592, triplet_hard_loss=0.2861, triplet_loss_num=8.0484, triplet_mean_dist=0.2317
[2026-04-01 21:01:04] [INFO]: Iteration 00951, Cost 0.07s, triplet_loss=0.2631, triplet_hard_loss=0.4545, triplet_loss_num=8.6613, triplet_mean_dist=0.2796
[2026-04-01 21:01:04] [INFO]: Iteration 00952, Cost 0.05s, triplet_loss=0.1725, triplet_hard_loss=0.3729, triplet_loss_num=8.1290, triplet_mean_dist=0.2374
[2026-04-01 21:01:05] [INFO]: Iteration 00953, Cost 0.05s, triplet_loss=0.2798, triplet_hard_loss=0.4874, triplet_loss_num=7.9839, triplet_mean_dist=0.2999
[2026-04-01 21:01:05] [INFO]: Iteration 00954, Cost 0.05s, triplet_loss=0.1731, triplet_hard_loss=0.3283, triplet_loss_num=7.8065, triplet_mean_dist=0.2327
[2026-04-01 21:01:05] [INFO]: Iteration 00955, Cost 0.05s, triplet_loss=0.2241, triplet_hard_loss=0.3665, triplet_loss_num=7.5000, triplet_mean_dist=0.2707
[2026-04-01 21:01:05] [INFO]: Iteration 00956, Cost 0.08s, triplet_loss=0.2112, triplet_hard_loss=0.3810, triplet_loss_num=6.2419, triplet_mean_dist=0.2904
[2026-04-01 21:01:05] [INFO]: Iteration 00957, Cost 0.05s, triplet_loss=0.1973, triplet_hard_loss=0.3509, triplet_loss_num=10.8065, triplet_mean_dist=0.2073
[2026-04-01 21:01:05] [INFO]: Iteration 00958, Cost 0.05s, triplet_loss=0.2511, triplet_hard_loss=0.4640, triplet_loss_num=7.3710, triplet_mean_dist=0.3110
[2026-04-01 21:01:05] [INFO]: Iteration 00959, Cost 0.05s, triplet_loss=0.1483, triplet_hard_loss=0.3033, triplet_loss_num=7.5323, triplet_mean_dist=0.2548
[2026-04-01 21:01:05] [INFO]: Iteration 00960, Cost 0.05s, triplet_loss=0.2330, triplet_hard_loss=0.4197, triplet_loss_num=7.6613, triplet_mean_dist=0.3004
[2026-04-01 21:01:05] [INFO]: Iteration 00961, Cost 0.07s, triplet_loss=0.2140, triplet_hard_loss=0.4223, triplet_loss_num=7.9516, triplet_mean_dist=0.2832
[2026-04-01 21:01:05] [INFO]: Iteration 00962, Cost 0.05s, triplet_loss=0.1298, triplet_hard_loss=0.2145, triplet_loss_num=6.9516, triplet_mean_dist=0.2304
[2026-04-01 21:01:05] [INFO]: Iteration 00963, Cost 0.05s, triplet_loss=0.1910, triplet_hard_loss=0.3624, triplet_loss_num=7.6129, triplet_mean_dist=0.2639
[2026-04-01 21:01:05] [INFO]: Iteration 00964, Cost 0.05s, triplet_loss=0.2075, triplet_hard_loss=0.4049, triplet_loss_num=7.3871, triplet_mean_dist=0.2598
[2026-04-01 21:01:05] [INFO]: Iteration 00965, Cost 0.05s, triplet_loss=0.1579, triplet_hard_loss=0.2741, triplet_loss_num=8.3710, triplet_mean_dist=0.2258
[2026-04-01 21:01:05] [INFO]: Iteration 00966, Cost 0.07s, triplet_loss=0.2202, triplet_hard_loss=0.3864, triplet_loss_num=6.8226, triplet_mean_dist=0.3276
[2026-04-01 21:01:05] [INFO]: Iteration 00967, Cost 0.05s, triplet_loss=0.0092, triplet_hard_loss=0.0147, triplet_loss_num=1.0323, triplet_mean_dist=0.2658
[2026-04-01 21:01:05] [INFO]: Iteration 00968, Cost 0.05s, triplet_loss=0.2921, triplet_hard_loss=0.4894, triplet_loss_num=7.6129, triplet_mean_dist=0.3531
[2026-04-01 21:01:05] [INFO]: Iteration 00969, Cost 0.05s, triplet_loss=0.1646, triplet_hard_loss=0.3178, triplet_loss_num=10.7903, triplet_mean_dist=0.1821
[2026-04-01 21:01:05] [INFO]: Iteration 00970, Cost 0.05s, triplet_loss=0.2432, triplet_hard_loss=0.3993, triplet_loss_num=8.7258, triplet_mean_dist=0.2553
[2026-04-01 21:01:06] [INFO]: Iteration 00971, Cost 0.07s, triplet_loss=0.1459, triplet_hard_loss=0.2476, triplet_loss_num=7.6613, triplet_mean_dist=0.2253
[2026-04-01 21:01:06] [INFO]: Iteration 00972, Cost 0.05s, triplet_loss=0.1715, triplet_hard_loss=0.2994, triplet_loss_num=7.6290, triplet_mean_dist=0.2266
[2026-04-01 21:01:06] [INFO]: Iteration 00973, Cost 0.05s, triplet_loss=0.1645, triplet_hard_loss=0.3383, triplet_loss_num=6.3548, triplet_mean_dist=0.2916
[2026-04-01 21:01:06] [INFO]: Iteration 00974, Cost 0.05s, triplet_loss=0.1116, triplet_hard_loss=0.2040, triplet_loss_num=4.7258, triplet_mean_dist=0.3647
[2026-04-01 21:01:06] [INFO]: Iteration 00975, Cost 0.05s, triplet_loss=0.3654, triplet_hard_loss=0.6866, triplet_loss_num=5.8548, triplet_mean_dist=0.5253
[2026-04-01 21:01:06] [INFO]: Iteration 00976, Cost 0.08s, triplet_loss=0.1545, triplet_hard_loss=0.2818, triplet_loss_num=8.3226, triplet_mean_dist=0.2224
[2026-04-01 21:01:06] [INFO]: Iteration 00977, Cost 0.05s, triplet_loss=0.2511, triplet_hard_loss=0.4260, triplet_loss_num=9.1774, triplet_mean_dist=0.2641
[2026-04-01 21:01:06] [INFO]: Iteration 00978, Cost 0.05s, triplet_loss=0.1718, triplet_hard_loss=0.3584, triplet_loss_num=8.4516, triplet_mean_dist=0.2397
[2026-04-01 21:01:06] [INFO]: Iteration 00979, Cost 0.05s, triplet_loss=0.1193, triplet_hard_loss=0.2510, triplet_loss_num=8.5161, triplet_mean_dist=0.2016
[2026-04-01 21:01:06] [INFO]: Iteration 00980, Cost 0.05s, triplet_loss=0.2657, triplet_hard_loss=0.4326, triplet_loss_num=7.8387, triplet_mean_dist=0.3016
[2026-04-01 21:01:06] [INFO]: Iteration 00981, Cost 0.07s, triplet_loss=0.2019, triplet_hard_loss=0.3806, triplet_loss_num=10.2258, triplet_mean_dist=0.1995
[2026-04-01 21:01:06] [INFO]: Iteration 00982, Cost 0.05s, triplet_loss=0.2641, triplet_hard_loss=0.4644, triplet_loss_num=7.3871, triplet_mean_dist=0.2991
[2026-04-01 21:01:06] [INFO]: Iteration 00983, Cost 0.05s, triplet_loss=0.1840, triplet_hard_loss=0.3274, triplet_loss_num=5.5161, triplet_mean_dist=0.4148
[2026-04-01 21:01:06] [INFO]: Iteration 00984, Cost 0.05s, triplet_loss=0.1574, triplet_hard_loss=0.2890, triplet_loss_num=7.1452, triplet_mean_dist=0.3061
[2026-04-01 21:01:06] [INFO]: Iteration 00985, Cost 0.05s, triplet_loss=0.1969, triplet_hard_loss=0.3381, triplet_loss_num=8.6129, triplet_mean_dist=0.2416
[2026-04-01 21:01:06] [INFO]: Iteration 00986, Cost 0.07s, triplet_loss=0.1569, triplet_hard_loss=0.2710, triplet_loss_num=7.5806, triplet_mean_dist=0.2692
[2026-04-01 21:01:06] [INFO]: Iteration 00987, Cost 0.05s, triplet_loss=0.1739, triplet_hard_loss=0.3453, triplet_loss_num=7.2742, triplet_mean_dist=0.2668
[2026-04-01 21:01:06] [INFO]: Iteration 00988, Cost 0.05s, triplet_loss=0.1937, triplet_hard_loss=0.3304, triplet_loss_num=8.5806, triplet_mean_dist=0.2327
[2026-04-01 21:01:06] [INFO]: Iteration 00989, Cost 0.05s, triplet_loss=0.1412, triplet_hard_loss=0.2400, triplet_loss_num=5.7903, triplet_mean_dist=0.2363
[2026-04-01 21:01:07] [INFO]: Iteration 00990, Cost 0.05s, triplet_loss=0.2345, triplet_hard_loss=0.3903, triplet_loss_num=8.4032, triplet_mean_dist=0.2639
[2026-04-01 21:01:07] [INFO]: Iteration 00991, Cost 0.07s, triplet_loss=0.2386, triplet_hard_loss=0.3893, triplet_loss_num=8.9355, triplet_mean_dist=0.2635
[2026-04-01 21:01:07] [INFO]: Iteration 00992, Cost 0.05s, triplet_loss=0.1120, triplet_hard_loss=0.1974, triplet_loss_num=11.0161, triplet_mean_dist=0.1403
[2026-04-01 21:01:07] [INFO]: Iteration 00993, Cost 0.05s, triplet_loss=0.2540, triplet_hard_loss=0.4683, triplet_loss_num=8.4677, triplet_mean_dist=0.2451
[2026-04-01 21:01:07] [INFO]: Iteration 00994, Cost 0.05s, triplet_loss=0.0975, triplet_hard_loss=0.1747, triplet_loss_num=8.0000, triplet_mean_dist=0.2001
[2026-04-01 21:01:07] [INFO]: Iteration 00995, Cost 0.05s, triplet_loss=0.1526, triplet_hard_loss=0.2749, triplet_loss_num=5.2742, triplet_mean_dist=0.2589
[2026-04-01 21:01:07] [INFO]: Iteration 00996, Cost 0.07s, triplet_loss=0.2560, triplet_hard_loss=0.4361, triplet_loss_num=7.1290, triplet_mean_dist=0.2556
[2026-04-01 21:01:07] [INFO]: Iteration 00997, Cost 0.05s, triplet_loss=0.0921, triplet_hard_loss=0.1619, triplet_loss_num=5.7581, triplet_mean_dist=0.2078
[2026-04-01 21:01:07] [INFO]: Iteration 00998, Cost 0.05s, triplet_loss=0.1859, triplet_hard_loss=0.2952, triplet_loss_num=5.5484, triplet_mean_dist=0.4132
[2026-04-01 21:01:07] [INFO]: Iteration 00999, Cost 0.05s, triplet_loss=0.1269, triplet_hard_loss=0.2482, triplet_loss_num=7.2097, triplet_mean_dist=0.2540
[2026-04-01 21:01:07] [INFO]: Iteration 01000, Cost 0.05s, triplet_loss=0.1690, triplet_hard_loss=0.2863, triplet_loss_num=5.6613, triplet_mean_dist=0.3425
[rank0]:[W401 21:01:07.614989022 ProcessGroupNCCL.cpp:1250] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())

```
