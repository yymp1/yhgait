# 真实视频步态检索 Demo 启动说明

## 这次交付了什么

- 现有 Gradio 页面已经新增 `真实视频 Demo` tab
- 它会把一段真实视频接到当前链路：
  - 单人检测 / 跟踪
  - silhouette 提取
  - 连续有效序列筛选
  - 用 `formal_conservative_real-01000.pt` 提 embedding
  - 在真实 `CASIA-B-pkl` gallery 上做 Top-K 检索

## 依赖前提

- Python 环境：`/home/bb/gait/yhgait/.venvs/opengait-gpu`
- checkpoint：`external/OpenGait/output/CASIA-B/GaitSet/formal_conservative_real/checkpoints/formal_conservative_real-01000.pt`
- gallery cache：`reports/gait_demo_cache_formal_conservative_real_iter01000.npz`
- 配置：`configs/opengait_casiab_formal_conservative_eval.yaml`

## 启动 GUI

```bash
/home/bb/gait/yhgait/.venvs/opengait-gpu/bin/python \
app/gradio_gait_demo.py \
  --server-name 127.0.0.1 \
  --server-port 7860
```

启动后访问：

- `http://127.0.0.1:7860`

如果你是通过 VS Code 远程连到 Linux：

- 转发 `7860` 端口后再在本地浏览器打开

## 页面里怎么用

在 `真实视频 Demo` tab：

1. 上传一段真实视频，或者填写服务器上的本地视频绝对路径
2. 选择 `Gallery 视角过滤`
   - 默认 `all`
   - 真实视频通常视角未知，所以默认不要强行限定
3. 可选填写 `Expected Subject`
   - 只有你知道这段视频确实对应 `CASIA-B` 某个 subject 时才填
   - 不知道就留空，页面里的 `hit/miss` 会显示 `n/a`
4. 点击 `处理视频并检索`
5. 页面会返回：
   - 标注预览视频
   - track 汇总表
   - 自动选中的 track
   - 当前选中 track clip
   - 当前选中 track 的 silhouette 序列摘要
   - Top-K 检索结果与 distance
6. 如果视频里有多个可用 track，可以在下拉框里切换，再点 `加载当前 track 结果`

## 输出产物会落到哪里

每次运行都会生成一个新的 run 目录：

- `reports/real_video_demo_runs/<video_stem>_<timestamp>/`

其中最关键的产物有：

- `annotated_result.mp4`
- `session_summary.json`
- `probe_tracks/<track_id>/silhouettes.pkl`
- `probe_tracks/<track_id>/silhouette_strip.png`
- `probe_tracks/<track_id>/embedding.npy`

## 命令行最小验证

如果你不想先开 GUI，也可以直接跑 CLI：

```bash
/home/bb/gait/yhgait/.venvs/opengait-gpu/bin/python \
scripts/run_real_video_gait_demo.py \
  --video-path /abs/path/to/video.mp4 \
  --gallery-cache-path /home/bb/gait/yhgait/reports/gait_demo_cache_formal_conservative_real_iter01000.npz \
  --cfg-path /home/bb/gait/yhgait/configs/opengait_casiab_formal_conservative_eval.yaml \
  --checkpoint-iter 1000 \
  --run-root /home/bb/gait/yhgait/reports/real_video_demo_runs \
  --topk 5 \
  --device cuda:0
```

## 相关说明

- 当前页面展示的是“真实视频 demo 效果”
- 它不是官方 benchmark 页面
- 它也不是完整业务系统
- 如果你想先看当前边界，请接着看 `reports/real_video_demo_scope.md`
