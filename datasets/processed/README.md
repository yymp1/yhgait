# 预处理数据输出目录

这个目录用于存放后续由 OpenGait `datasets/pretreatment.py` 生成的中间产物。

建议：

- `datasets/processed/CASIA-B-pkl/`：CASIA-B pretreatment 输出
- `datasets/processed/*`：后续也可以放 OUMVLP 或其他公开 gait 数据集的 pretreatment 结果

当前阶段只做命令生成、目录检查和对接准备，不在这里直接启动正式训练。
