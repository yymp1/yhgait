# 真实视频 Demo 常见失败点

## 1. 检测 / 跟踪失败

表现：

- 没有 track
- track 很碎
- 自动选中的 track 明显不是目标人

原因：

- 人物太小
- 画面里多人交错
- 遮挡严重
- 视频质量差

## 2. silhouette 提取失败

表现：

- `quality_note = segmentation_failed`
- `silhouette_strip` 断裂、噪声很重、人体形状不完整

原因：

- 前景和背景分离差
- 光照差
- crop 过紧或过松
- 姿态变化太激烈

## 3. 连续有效帧不足

表现：

- `quality_note = valid_run_too_short<16`
- 有 track，但进不了 retrieval

原因：

- track 虽然存在，但可用 silhouette 不连续
- 人物离开画面太快
- 分割稳定性不够

## 4. 检索结果看起来“不像”

表现：

- Top-K 很分散
- distance 不够集中
- Top-1 看起来不合理

原因：

- 真实视频和 `CASIA-B` 有明显域差异
- 视角不匹配
- silhouette 质量一般
- 当前 checkpoint 仍然只是第一轮正式 baseline

## 5. hit/miss 无法解释

表现：

- 页面显示 `n/a`

原因：

- 没有填写 `Expected Subject`
- 这段视频本来就不属于 `CASIA-B` 的已知 subject

## 排查顺序建议

先看这四个东西，不要一上来怀疑模型：

1. `annotated_result.mp4`
2. `track clip`
3. `silhouette_strip.png`
4. `session_summary.json`
