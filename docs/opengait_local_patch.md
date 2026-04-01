# OpenGait 本地补丁说明

## 修改文件

- `external/OpenGait/datasets/pretreatment.py`

## 补丁内容

在 `cv2.imread()` 之后增加不可读帧判断：

```python
img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
if img is None:
    logging.warning(f'{img_file} is unreadable.')
    continue
```

## 为什么需要

- 当前下载得到的 `CASIA-B silhouette` 原始目录里，真实存在 `6` 个 `0B png`
- 对这类坏帧，`cv2.imread()` 会返回 `None`
- OpenGait 原始 `pretreatment.py` 后续会访问 `img.sum()`，因此会在坏帧处直接中断整批预处理

## 影响

- 正常可读帧的处理逻辑不变
- 坏帧只会被记 warning 并跳过
- 当前 `datasets/processed/CASIA-B-pkl` 就是在这个补丁存在的前提下成功生成的

## 重新 clone 后怎么保留

优先保留并重新应用：

- `patches/pretreatment_skip_bad_frames.patch`

也可以参考：

- `reports/opengait_local_patch_summary.md`
- `reports/opengait_pretreatment_local.patch`
