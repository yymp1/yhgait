# 现在上传我的视频后，系统会怎么识别

## 主流程

当你在 GUI 的“上传识别”页提交一段视频后，系统会按下面顺序处理：

1. 读取视频
2. 做单人检测与 tracking
3. 导出每条 track 的 clip 和逐帧 crop
4. 从每条 track 中提取 silhouette 序列
5. 对通过质量门槛的 track 提 gait embedding
6. 用该 embedding 和私有步态库中的每个 identity prototype 做距离比较
7. 输出 Top-K 候选

## 识别结果怎么看

页面会展示：

- Top-1 最像谁
- Top-K 候选列表
- distance
- score
- 是否建议人工复核

解释如下：

- `distance` 越小，表示越相似
- `score` 只是展示分数，不是概率
- 如果 Top-1 和 Top-2 太接近，或者库太小、样本太少，系统会提示人工复核

## 多人视频怎么处理

如果视频里有多个人，系统会先拆成多条 track。

当前默认会自动优先选择：

1. 连续有效帧更多的 track
2. 轨迹更长的 track
3. bbox 更大的 track
4. 置信度更高的 track

同时你也可以在 GUI 里手动切换 track，再看另一条轨迹的识别结果。

## 当前的最小判定策略

当前不是直接“强行认定一个人”，而是：

1. 给出 Top-1
2. 给出 Top-K 候选
3. 对低置信度情况主动提示人工复核

这更适合当前 demo 阶段。
