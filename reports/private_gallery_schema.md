# 私有步态库数据结构说明

## 根目录

当前默认私有库目录：

```text
data/private_gallery/
```

核心文件结构如下：

```text
data/private_gallery/
  index.json
  identities/
    <identity_dir>/
      prototype.npy
      aggregate_embedding.npy   # 兼容旧格式时可能存在
      samples/
        <sample_id>/
          embedding.npy
          silhouettes.pkl
          silhouette_strip.png
          track_clip.mp4
          sample_meta.json
```

## index.json 的最小语义

当前主实现使用这两个主表：

1. `identities`
2. `samples`

其中：

- `identities`：按 identity 聚合后的索引
- `samples`：每一段录入视频样本的明细

每个 identity 至少包含：

- `identity`
- `dir_name`
- `prototype_method`
- `prototype_path`
- `representative_sample_id`
- `sample_count`
- `sample_ids`

每个 sample 至少包含：

- `sample_id`
- `identity`
- `input_video`
- `track_id`
- `frame_count`
- `valid_frames`
- `used_frames`
- `embedding_path`
- `silhouette_pkl_path`
- `silhouette_strip_path`
- `track_clip_path`
- `note`

## 多样本怎么聚合

当前采用最小可行方案：

1. 每个 identity 保留多条原始样本
2. 每条样本都保存自己的 `embedding.npy`
3. identity 的代表特征使用“所有样本 embedding 的均值”

也就是：

- 样本级信息保留
- 身份级匹配使用 mean prototype

这样做的好处是：

- 实现简单
- 便于后续继续追加样本
- 便于重建 prototype

## 识别时怎么比

识别时的主流程是：

1. 对新视频选定 track 提取 embedding
2. 和私有库里每个 identity 的 prototype 做距离比较
3. 按 distance 从小到大排序
4. 输出 Top-K

当前 distance 使用的是和现有 gait retrieval 逻辑一致的 embedding 距离度量；distance 越小越相似。

GUI 中显示的 `score` 是一个展示分数：

```text
score = 1 / (1 + distance)
```

它不是概率，也不是官方 benchmark 分数。

## 什么时候提示人工复核

当前最小规则是：

- 私有库 identity 太少
- Top-1 与 Top-2 距离差太小
- Top-1 身份样本数太少
- Top-1 distance 偏大

只要命中这些条件之一，GUI 会提示“建议人工复核”。

## 兼容性说明

当前 loader 已兼容较早阶段留下的旧索引格式：

- 旧格式 `identities` 可能是 `dict`
- 旧格式可能使用 `aggregate_embedding_path`
- 新旧 prototype 命名都能读取
