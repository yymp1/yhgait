# 批量采集汇总

- 生成时间：2026-03-31 20:11:44
- 输入目录：`samples`
- 扫描到视频数：10
- 成功：2
- 跳过：8
- 失败：0
- 导出轨迹数：93
- 可能质量较差轨迹数：158

## 视频汇总

| 视频 | 状态 | 导出轨迹 | 可疑轨迹 | 结果视频 | artifacts | data | 备注 |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| IMG_9879.MOV | skipped | 2 | 0 | `outputs/IMG_9879_result.mp4` | `artifacts/IMG_9879` | `data/IMG_9879` | 已跳过已有结果 |
| IMG_9880.MOV | success | 3 | 2 | `outputs/IMG_9880_result.mp4` | `artifacts/IMG_9880` | `data/IMG_9880` |  |
| IMG_9881.MOV | skipped | 36 | 88 | `outputs/IMG_9881_result.mp4` | `artifacts/IMG_9881` | `data/IMG_9881` | 已跳过已有结果 |
| IMG_9882.MOV | skipped | 10 | 14 | `outputs/IMG_9882_result.mp4` | `artifacts/IMG_9882` | `data/IMG_9882` | 已跳过已有结果 |
| IMG_9883.MOV | skipped | 9 | 8 | `outputs/IMG_9883_result.mp4` | `artifacts/IMG_9883` | `data/IMG_9883` | 已跳过已有结果 |
| IMG_9884.MOV | skipped | 4 | 12 | `outputs/IMG_9884_result.mp4` | `artifacts/IMG_9884` | `data/IMG_9884` | 已跳过已有结果 |
| IMG_9885.MOV | skipped | 21 | 25 | `outputs/IMG_9885_result.mp4` | `artifacts/IMG_9885` | `data/IMG_9885` | 已跳过已有结果 |
| IMG_9886.MOV | skipped | 5 | 5 | `outputs/IMG_9886_result.mp4` | `artifacts/IMG_9886` | `data/IMG_9886` | 已跳过已有结果 |
| IMG_9887.MOV | skipped | 3 | 1 | `outputs/IMG_9887_result.mp4` | `artifacts/IMG_9887` | `data/IMG_9887` | 已跳过已有结果 |
| test.mp4 | success | 0 | 3 | `outputs/test_result.mp4` | `artifacts/test` | `data/test` |  |

## 可疑轨迹

| 视频 | track_id | 帧数 | kept | avg_bbox_area | avg_confidence | 说明 |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| IMG_9880.MOV | 0004 | 2 | 0 | 8048.62 | 0.3805 | too_short |
| IMG_9880.MOV | 0005 | 2 | 0 | 5818.10 | 0.3807 | too_short |
| IMG_9881.MOV | 0005 | 1 | 0 | 7318.47 | 0.3988 | too_short |
| IMG_9881.MOV | 0006 | 7 | 0 | 9407.72 | 0.4079 | too_short |
| IMG_9881.MOV | 0007 | 2 | 0 | 7827.93 | 0.3840 | too_short |
| IMG_9881.MOV | 0008 | 2 | 0 | 5093.50 | 0.4277 | too_short |
| IMG_9881.MOV | 0009 | 6 | 0 | 10201.52 | 0.5040 | too_short |
| IMG_9881.MOV | 0011 | 1 | 0 | 6165.50 | 0.3638 | too_short |
| IMG_9881.MOV | 0012 | 1 | 0 | 6766.64 | 0.3703 | too_short |
| IMG_9881.MOV | 0013 | 2 | 0 | 7549.08 | 0.4290 | too_short |
| IMG_9881.MOV | 0014 | 3 | 0 | 6978.56 | 0.4127 | too_short |
| IMG_9881.MOV | 0016 | 2 | 0 | 10895.29 | 0.3970 | too_short |
| IMG_9881.MOV | 0017 | 12 | 0 | 22907.16 | 0.6222 | too_short |
| IMG_9881.MOV | 0018 | 3 | 0 | 11105.16 | 0.3610 | too_short |
| IMG_9881.MOV | 0019 | 25 | 1 | 858725.60 | 0.8061 | borderline_short |
| IMG_9881.MOV | 0020 | 3 | 0 | 268740.13 | 0.5691 | too_short |
| IMG_9881.MOV | 0021 | 3 | 0 | 12000.48 | 0.4356 | too_short |
| IMG_9881.MOV | 0024 | 3 | 0 | 53584.10 | 0.5923 | too_short |
| IMG_9881.MOV | 0025 | 1 | 0 | 2530.76 | 0.3988 | too_short,small_bbox |
| IMG_9881.MOV | 0026 | 3 | 0 | 24369.29 | 0.4101 | too_short |
| IMG_9881.MOV | 0027 | 2 | 0 | 3086.86 | 0.3723 | too_short,small_bbox |
| IMG_9881.MOV | 0028 | 1 | 0 | 6725.31 | 0.4033 | too_short |
| IMG_9881.MOV | 0029 | 1 | 0 | 2725.07 | 0.3707 | too_short,small_bbox |
| IMG_9881.MOV | 0030 | 3 | 0 | 24930.35 | 0.3725 | too_short |
| IMG_9881.MOV | 0031 | 24 | 1 | 10076.23 | 0.5271 | borderline_short |
| IMG_9881.MOV | 0032 | 4 | 0 | 4583.33 | 0.4716 | too_short |
| IMG_9881.MOV | 0033 | 2 | 0 | 23972.30 | 0.3711 | too_short |
| IMG_9881.MOV | 0034 | 29 | 1 | 9852.83 | 0.5270 | borderline_short |
| IMG_9881.MOV | 0035 | 15 | 0 | 8393.15 | 0.4921 | too_short |
| IMG_9881.MOV | 0038 | 1 | 0 | 11350.04 | 0.3608 | too_short |
| IMG_9881.MOV | 0039 | 26 | 1 | 14025.17 | 0.7232 | borderline_short |
| IMG_9881.MOV | 0040 | 1 | 0 | 3592.03 | 0.4082 | too_short |
| IMG_9881.MOV | 0041 | 2 | 0 | 3659.45 | 0.4726 | too_short |
| IMG_9881.MOV | 0042 | 21 | 1 | 12748.30 | 0.7185 | borderline_short |
| IMG_9881.MOV | 0043 | 8 | 0 | 9783.71 | 0.5808 | too_short |
| IMG_9881.MOV | 0044 | 4 | 0 | 3387.11 | 0.4453 | too_short |
| IMG_9881.MOV | 0045 | 15 | 0 | 8072.69 | 0.5256 | too_short |
| IMG_9881.MOV | 0046 | 9 | 0 | 12163.66 | 0.6704 | too_short |
| IMG_9881.MOV | 0047 | 1 | 0 | 7179.33 | 0.5847 | too_short |
| IMG_9881.MOV | 0048 | 2 | 0 | 3484.20 | 0.4926 | too_short |
| IMG_9881.MOV | 0049 | 1 | 0 | 7771.02 | 0.4280 | too_short |
| IMG_9881.MOV | 0050 | 6 | 0 | 7319.92 | 0.5579 | too_short |
| IMG_9881.MOV | 0051 | 5 | 0 | 7855.82 | 0.4813 | too_short |
| IMG_9881.MOV | 0052 | 4 | 0 | 7730.59 | 0.4901 | too_short |
| IMG_9881.MOV | 0053 | 3 | 0 | 3804.87 | 0.3994 | too_short |
| IMG_9881.MOV | 0054 | 6 | 0 | 9776.32 | 0.5007 | too_short |
| IMG_9881.MOV | 0055 | 1 | 0 | 3959.69 | 0.4387 | too_short |
| IMG_9881.MOV | 0056 | 11 | 0 | 9421.37 | 0.5869 | too_short |
| IMG_9881.MOV | 0057 | 3 | 0 | 9231.40 | 0.5999 | too_short |
| IMG_9881.MOV | 0058 | 5 | 0 | 11087.50 | 0.6629 | too_short |
| IMG_9881.MOV | 0059 | 3 | 0 | 9162.79 | 0.5770 | too_short |
| IMG_9881.MOV | 0060 | 7 | 0 | 12519.45 | 0.5380 | too_short |
| IMG_9881.MOV | 0061 | 4 | 0 | 10439.30 | 0.6386 | too_short |
| IMG_9881.MOV | 0062 | 4 | 0 | 10887.40 | 0.6463 | too_short |
| IMG_9881.MOV | 0063 | 4 | 0 | 10624.23 | 0.6612 | too_short |
| IMG_9881.MOV | 0064 | 4 | 0 | 11103.72 | 0.6197 | too_short |
| IMG_9881.MOV | 0065 | 4 | 0 | 11259.49 | 0.6723 | too_short |
| IMG_9881.MOV | 0066 | 4 | 0 | 10987.65 | 0.6064 | too_short |
| IMG_9881.MOV | 0067 | 2 | 0 | 11748.14 | 0.6058 | too_short |
| IMG_9881.MOV | 0068 | 4 | 0 | 11347.00 | 0.5806 | too_short |
| IMG_9881.MOV | 0069 | 4 | 0 | 11719.44 | 0.6110 | too_short |
| IMG_9881.MOV | 0070 | 4 | 0 | 11878.62 | 0.6033 | too_short |
| IMG_9881.MOV | 0071 | 4 | 0 | 12070.83 | 0.6313 | too_short |
| IMG_9881.MOV | 0077 | 2 | 0 | 15328.23 | 0.3790 | too_short |
| IMG_9881.MOV | 0080 | 2 | 0 | 15171.88 | 0.3874 | too_short |
| IMG_9881.MOV | 0081 | 16 | 0 | 61392.44 | 0.6661 | too_short |
| IMG_9881.MOV | 0084 | 1 | 0 | 43522.12 | 0.3672 | too_short |
| IMG_9881.MOV | 0085 | 1 | 0 | 8688.38 | 0.3633 | too_short |
| IMG_9881.MOV | 0086 | 6 | 0 | 9074.62 | 0.4720 | too_short |
| IMG_9881.MOV | 0087 | 2 | 0 | 6320.08 | 0.4814 | too_short |
| IMG_9881.MOV | 0088 | 2 | 0 | 8676.94 | 0.4668 | too_short |
| IMG_9881.MOV | 0089 | 3 | 0 | 33202.41 | 0.5195 | too_short |
| IMG_9881.MOV | 0090 | 9 | 0 | 16060.88 | 0.4073 | too_short |
| IMG_9881.MOV | 0091 | 12 | 0 | 59651.11 | 0.5332 | too_short |
| IMG_9881.MOV | 0093 | 1 | 0 | 44991.67 | 0.3982 | too_short |
| IMG_9881.MOV | 0095 | 1 | 0 | 45675.82 | 0.3784 | too_short |
| IMG_9881.MOV | 0096 | 3 | 0 | 63699.92 | 0.4483 | too_short |
| IMG_9881.MOV | 0097 | 9 | 0 | 21349.90 | 0.5041 | too_short |
| IMG_9881.MOV | 0099 | 4 | 0 | 12005.61 | 0.5174 | too_short |
| IMG_9881.MOV | 0100 | 1 | 0 | 9935.61 | 0.5785 | too_short |
| IMG_9881.MOV | 0102 | 22 | 1 | 14580.36 | 0.7710 | borderline_short |
| IMG_9881.MOV | 0103 | 1 | 0 | 4768.83 | 0.3682 | too_short |
| IMG_9881.MOV | 0105 | 20 | 1 | 6522.53 | 0.4486 | borderline_short |
| IMG_9881.MOV | 0107 | 25 | 1 | 7356.65 | 0.5471 | borderline_short |
| IMG_9881.MOV | 0109 | 1 | 0 | 5834.67 | 0.3578 | too_short |
| IMG_9881.MOV | 0111 | 7 | 0 | 7876.74 | 0.4200 | too_short |
| IMG_9881.MOV | 0112 | 13 | 0 | 6001.13 | 0.4512 | too_short |
| IMG_9881.MOV | 0113 | 4 | 0 | 4700.97 | 0.3832 | too_short |
| IMG_9881.MOV | 0114 | 11 | 0 | 10040.30 | 0.4307 | too_short |
| IMG_9881.MOV | 0115 | 21 | 1 | 8850.19 | 0.5823 | borderline_short |
| IMG_9882.MOV | 0003 | 1 | 0 | 13368.52 | 0.6262 | too_short |
| IMG_9882.MOV | 0005 | 15 | 0 | 24738.42 | 0.5099 | too_short |
| IMG_9882.MOV | 0007 | 1 | 0 | 6536.21 | 0.3825 | too_short |
| IMG_9882.MOV | 0009 | 7 | 0 | 15644.21 | 0.4333 | too_short |
| IMG_9882.MOV | 0010 | 7 | 0 | 6883.49 | 0.4180 | too_short |
| IMG_9882.MOV | 0011 | 14 | 0 | 5149.89 | 0.4847 | too_short |
| IMG_9882.MOV | 0012 | 13 | 0 | 6087.25 | 0.4158 | too_short |
| IMG_9882.MOV | 0013 | 13 | 0 | 6110.06 | 0.4610 | too_short |
| IMG_9882.MOV | 0014 | 21 | 1 | 6005.19 | 0.4572 | borderline_short |
| IMG_9882.MOV | 0016 | 9 | 0 | 6163.77 | 0.4584 | too_short |
| IMG_9882.MOV | 0018 | 10 | 0 | 6248.83 | 0.4452 | too_short |
| IMG_9882.MOV | 0019 | 10 | 0 | 6209.20 | 0.4204 | too_short |
| IMG_9882.MOV | 0022 | 1 | 0 | 3149.55 | 0.3502 | too_short,small_bbox |
| IMG_9882.MOV | 0023 | 10 | 0 | 5260.42 | 0.4069 | too_short |
| IMG_9883.MOV | 0002 | 8 | 0 | 78736.85 | 0.6516 | too_short |
| IMG_9883.MOV | 0005 | 1 | 0 | 6096.32 | 0.5459 | too_short |
| IMG_9883.MOV | 0007 | 23 | 1 | 7532.83 | 0.5423 | borderline_short |
| IMG_9883.MOV | 0010 | 6 | 0 | 6594.12 | 0.4487 | too_short |
| IMG_9883.MOV | 0014 | 16 | 0 | 7149.64 | 0.4403 | too_short |
| IMG_9883.MOV | 0015 | 14 | 0 | 6847.73 | 0.4295 | too_short |
| IMG_9883.MOV | 0016 | 1 | 0 | 6505.02 | 0.3746 | too_short |
| IMG_9883.MOV | 0017 | 1 | 0 | 7171.55 | 0.4174 | too_short |
| IMG_9884.MOV | 0001 | 10 | 0 | 73841.08 | 0.6527 | too_short |
| IMG_9884.MOV | 0003 | 8 | 0 | 7829.32 | 0.5653 | too_short |
| IMG_9884.MOV | 0005 | 9 | 0 | 5582.75 | 0.4513 | too_short |
| IMG_9884.MOV | 0007 | 7 | 0 | 5135.39 | 0.4400 | too_short |
| IMG_9884.MOV | 0008 | 6 | 0 | 57358.84 | 0.5950 | too_short |
| IMG_9884.MOV | 0009 | 4 | 0 | 9912.05 | 0.4152 | too_short |
| IMG_9884.MOV | 0010 | 25 | 1 | 11042.58 | 0.6801 | borderline_short |
| IMG_9884.MOV | 0011 | 10 | 0 | 43528.78 | 0.4953 | too_short |
| IMG_9884.MOV | 0012 | 8 | 0 | 5145.86 | 0.4394 | too_short |
| IMG_9884.MOV | 0013 | 4 | 0 | 7011.93 | 0.4274 | too_short |
| IMG_9884.MOV | 0014 | 5 | 0 | 9585.47 | 0.3976 | too_short |
| IMG_9884.MOV | 0015 | 1 | 0 | 19490.39 | 0.3630 | too_short |
| IMG_9885.MOV | 0007 | 27 | 1 | 8480.31 | 0.4470 | borderline_short |
| IMG_9885.MOV | 0011 | 1 | 0 | 4528.64 | 0.3972 | too_short |
| IMG_9885.MOV | 0012 | 1 | 0 | 11315.38 | 0.3800 | too_short |
| IMG_9885.MOV | 0013 | 10 | 0 | 8723.85 | 0.4849 | too_short |
| IMG_9885.MOV | 0014 | 14 | 0 | 8275.42 | 0.4214 | too_short |
| IMG_9885.MOV | 0018 | 1 | 0 | 11553.97 | 0.4769 | too_short |
| IMG_9885.MOV | 0019 | 1 | 0 | 10836.29 | 0.4152 | too_short |
| IMG_9885.MOV | 0021 | 1 | 0 | 23502.63 | 0.3683 | too_short |
| IMG_9885.MOV | 0022 | 11 | 0 | 44096.57 | 0.4979 | too_short |
| IMG_9885.MOV | 0024 | 1 | 0 | 17046.07 | 0.3532 | too_short |
| IMG_9885.MOV | 0026 | 6 | 0 | 12712.11 | 0.4686 | too_short |
| IMG_9885.MOV | 0028 | 1 | 0 | 11680.87 | 0.3552 | too_short |
| IMG_9885.MOV | 0029 | 4 | 0 | 35491.61 | 0.4350 | too_short |
| IMG_9885.MOV | 0030 | 5 | 0 | 9866.40 | 0.4247 | too_short |
| IMG_9885.MOV | 0031 | 1 | 0 | 49144.44 | 0.3748 | too_short |
| IMG_9885.MOV | 0032 | 2 | 0 | 57643.23 | 0.5370 | too_short |
| IMG_9885.MOV | 0034 | 18 | 0 | 30332.85 | 0.4530 | too_short |
| IMG_9885.MOV | 0038 | 1 | 0 | 10880.90 | 0.3534 | too_short |
| IMG_9885.MOV | 0039 | 1 | 0 | 12982.06 | 0.3507 | too_short |
| IMG_9885.MOV | 0040 | 8 | 0 | 15409.38 | 0.4035 | too_short |
| IMG_9885.MOV | 0041 | 5 | 0 | 8197.02 | 0.3727 | too_short |
| IMG_9885.MOV | 0042 | 2 | 0 | 5403.69 | 0.3978 | too_short |
| IMG_9885.MOV | 0043 | 1 | 0 | 11756.03 | 0.3743 | too_short |
| IMG_9885.MOV | 0044 | 23 | 1 | 49375.41 | 0.4596 | borderline_short |
| IMG_9885.MOV | 0045 | 1 | 0 | 24797.79 | 0.3956 | too_short |
| IMG_9886.MOV | 0004 | 2 | 0 | 7835.41 | 0.3800 | too_short |
| IMG_9886.MOV | 0005 | 1 | 0 | 5066.23 | 0.3803 | too_short |
| IMG_9886.MOV | 0006 | 1 | 0 | 3959.34 | 0.4028 | too_short |
| IMG_9886.MOV | 0009 | 19 | 0 | 5558.40 | 0.4082 | too_short |
| IMG_9886.MOV | 0010 | 4 | 0 | 11001.96 | 0.3837 | too_short |
| IMG_9887.MOV | 0005 | 17 | 0 | 6663.03 | 0.4668 | too_short |
| test.mp4 | 0001 | 15 | 0 | 99393.34 | 0.8644 | too_short |
| test.mp4 | 0002 | 15 | 0 | 55696.42 | 0.8364 | too_short |
| test.mp4 | 0003 | 15 | 0 | 68027.78 | 0.8536 | too_short |

## 失败视频

没有失败视频。
