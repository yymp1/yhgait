# 真实视频 Demo 验证记录

## 验证目标

确认当前页面不是静态壳，而是已经真实跑通：

- 真实视频输入
- 单人检测 / 跟踪
- silhouette 提取
- 当前 checkpoint embedding
- Top-K 检索
- GUI 页面返回

## 使用的核心配置

- checkpoint：`formal_conservative_real-01000.pt`
- gallery cache：`reports/gait_demo_cache_formal_conservative_real_iter01000.npz`
- 配置：`configs/opengait_casiab_formal_conservative_eval.yaml`
- GUI：`app/gradio_gait_demo.py`
- 真实视频处理核心：`src/video_gait_core.py`

## 真实执行过的命令

重新启动 GUI：

```bash
/home/bb/gait/yhgait/.venvs/opengait-gpu/bin/python \
app/gradio_gait_demo.py \
  --server-name 127.0.0.1 \
  --server-port 7860
```

探活：

```bash
curl -I http://127.0.0.1:7860
```

结果：

- `HTTP/1.1 200 OK`

函数级真实视频验证：

```bash
/home/bb/gait/yhgait/.venvs/opengait-gpu/bin/python - <<'PY'
from pathlib import Path
import sys
ROOT = Path('/home/bb/gait/yhgait')
sys.path.insert(0, str(ROOT / 'app'))
import gradio_gait_demo as demo
from video_gait_core import shutdown_distributed

video_path = Path('/home/bb/.../eb19955971e2cad2909ca8f6aad25d19.mp4')
primary_cache = demo.load_cache(ROOT / 'reports/gait_demo_cache_formal_conservative_real_iter01000.npz')
try:
    outputs = demo.run_real_video(
        uploaded_video=None,
        local_video_path=str(video_path),
        gallery_view='all',
        expected_subject='',
        topk=5,
        detection_device='cuda:0',
        gallery_cache=primary_cache,
        cfg_path=ROOT / 'configs/opengait_casiab_formal_conservative_eval.yaml',
        checkpoint_iter=1000,
        run_root=ROOT / 'reports/real_video_demo_runs',
    )
finally:
    shutdown_distributed()
PY
```

## 当前采用的验证 run

- run 目录：`reports/real_video_demo_runs/eb19955971e2cad2909ca8f6aad25d19_20260401_234415`
- 摘要文件：`reports/real_video_demo_runs/eb19955971e2cad2909ca8f6aad25d19_20260401_234415/session_summary.json`
- 标注视频：`reports/real_video_demo_runs/eb19955971e2cad2909ca8f6aad25d19_20260401_234415/annotated_result.mp4`

## 真实结果

- track 数量：`2`
- 自动选择 track：`0001`
- `0001`
  - `status = ok`
  - `used_frames = 50`
  - `top1_subject = 096`
  - `top1_distance = 0.5862`
- `0002`
  - `status = ok`
  - `used_frames = 25`
  - `top1_subject = 101`
  - `top1_distance = 0.5762`

GUI 事件返回也已验证：

- `annotated_video` 文件存在
- 当前选中 `track clip` 文件存在
- 主 track 的 `silhouette_strip.png` 文件存在
- Top-5 gallery 结果已返回
- 切换到第二个 track 后，也能正常返回 track clip 与 Top-5

## 当前可直接引用的产物

- `reports/real_video_demo_runs/eb19955971e2cad2909ca8f6aad25d19_20260401_234415/session_summary.json`
- `reports/real_video_demo_runs/eb19955971e2cad2909ca8f6aad25d19_20260401_234415/annotated_result.mp4`
- `reports/real_video_demo_runs/eb19955971e2cad2909ca8f6aad25d19_20260401_234415/artifacts/clips/track_0001.mp4`
- `reports/real_video_demo_runs/eb19955971e2cad2909ca8f6aad25d19_20260401_234415/probe_tracks/0001/silhouette_strip.png`
- `reports/real_video_demo_runs/eb19955971e2cad2909ca8f6aad25d19_20260401_234415/artifacts/clips/track_0002.mp4`
- `reports/real_video_demo_runs/eb19955971e2cad2909ca8f6aad25d19_20260401_234415/probe_tracks/0002/silhouette_strip.png`

## 非阻塞提示

- 首次处理真实视频时，YOLO 权重下载会更慢；当前机器已经下载完成
- `mediapipe / tflite` 会打印一些 warning，这轮没有阻塞结果
- 如果用“一次性短脚本”直接调 OpenGait，再立刻退出，可能看到 NCCL 退出提示；GUI 常驻服务本身不依赖这个提示来判断成功与否
