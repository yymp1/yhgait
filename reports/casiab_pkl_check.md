# CASIA-B pretreatment 产物检查

## 结论

- `datasets/processed/CASIA-B-pkl` 已生成
- `subject_count = 124`
- `pkl_count = 13592`
- 随机抽样的 `pkl` 可正常读取，形状符合 `n x 64 x 64`

## 抽样结果

- `004/nm-04/036/036.pkl` -> `shape=(111, 64, 64)`, `dtype=uint8`
- `017/nm-03/144/144.pkl` -> `shape=(112, 64, 64)`, `dtype=uint8`
- `042/bg-02/036/036.pkl` -> `shape=(82, 64, 64)`, `dtype=uint8`

## 说明

- 当前检查针对的是 pretreatment 输出，不再依赖原始 png 是否完美
- 原始 silhouette 目录里仍然已知存在 `6` 个 `0B png` 坏帧
- 这些坏帧已通过本地 `pretreatment.py` 健壮性补丁做 `warning + skip`
