# 公开 gait 数据集预留目录

这个目录只用于给后续接 OpenGait 预留固定放置位置，本项目当前不会自动下载任何公开数据集。

建议约定：

- `datasets/external/casia_b/`：放 CASIA-B 原始数据
- `datasets/external/oumvlp/`：放 OUMVLP 原始数据
- `datasets/external/*_pkl/`：放后续按 OpenGait 官方流程生成的预处理结果

推荐顺序：

1. 先用 `CASIA-B` 跑通 OpenGait 的数据准备、训练和评测
2. 再接 `OUMVLP` 做更大规模验证
3. 最后把本项目导出的自拍视频序列作为小规模验证或 demo 数据

注意：

- 当前项目负责视频侧提取、tracking、轨迹导出和目录整理
- 公开数据集负责标准训练与标准评测
- 少量自拍视频不应当替代公开数据集训练
