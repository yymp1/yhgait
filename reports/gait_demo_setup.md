# 步态检索闭环 GUI 落地说明

## 先看哪份文档

- 如果你现在要看“真实视频 demo”，优先看：
  - `reports/real_video_demo_setup.md`
  - `reports/real_video_demo_validation.md`
  - `reports/real_video_demo_scope.md`

## 这次做的是什么

- 基于真实 `CASIA-B-pkl`
- 基于真实 `formal_conservative_real-00500.pt` 与 `formal_conservative_real-01000.pt`
- 复用 OpenGait 官方推理链路抽取 embedding
- 对固定测试集 probe 做单视角 gallery 检索分析
- 用 Gradio 提供一个“统计 + 案例 + checkpoint 对比”的闭环页面

## 当前主要文件

- 构建 embedding 缓存：
  - `scripts/build_gait_gallery.py`
- 构建固定测试集检索分析：
  - `scripts/build_gait_retrieval_analysis.py`
- CLI 检索验证：
  - `scripts/run_gait_retrieval_demo.py`
- GUI 入口：
  - `app/gradio_gait_demo.py`
- 共享检索逻辑：
  - `src/gait_demo_core.py`
- GUI 依赖：
  - `requirements-gait-demo.txt`
- 已构建缓存与分析：
  - `reports/gait_demo_cache_formal_conservative_real_iter00500.npz`
  - `reports/gait_demo_cache_formal_conservative_real_iter01000.npz`
  - `reports/gait_demo_analysis_iter00500.json`
  - `reports/gait_demo_analysis_iter01000.json`

## 当前页面能展示什么

1. 加载真实 checkpoint 的 embedding 结果
2. 读取真实 CASIA-B test 集 probe / gallery
3. 展示固定测试集的 Top-1 / Top-5 命中率
4. 展示 `NM / BG / CL` 三组分开统计
5. 展示 `00500` 与 `01000` 的同一 probe Top-K 对照
6. 展示 `01000` 的成功案例和失败案例
7. 展示 probe 与 gallery 的 silhouette 摘要图
8. 展示每个结果的 `subject / type / view / distance / hit`

## 当前页面不能展示什么

1. 它不是现实业务视频检索系统
2. 它不处理原始 RGB 视频
3. 它不做多人检测和跟踪
4. 它不代表最终业务效果
5. 它展示的是“数据集检索效果”
6. 它的固定测试集命中率不等于官方 `NM/BG/CL Rank-1` 评测值

## 页面里的 Top-K 和“80%”怎么理解

- 当前 score 是 GaitSet embedding 的平均欧式距离
- 距离越小，表示越相似
- probe 固定使用 `nm-05 / nm-06 / bg-01 / bg-02 / cl-01 / cl-02`
- gallery 固定使用同视角 `nm-01 ~ nm-04`
- 页面里更应该说“固定测试集 Top-1 / Top-5 命中率”，不要笼统叫“模型准确率”
- 当前这套 GUI 口径和官方 `NM/BG/CL Rank-1` 不是同一个协议

## 启动方式

当前版本会默认读取 `00500` 与 `01000` 的 cache / analysis，并在缺失时自动补建。

最直接的启动命令：

```bash
/home/bb/gait/yhgait/.venvs/opengait-gpu/bin/python \
app/gradio_gait_demo.py \
  --server-name 127.0.0.1 \
  --server-port 7860
```

如果你想手动预先构建 cache / analysis，可以执行：

```bash
/home/bb/gait/yhgait/.venvs/opengait-gpu/bin/python \
scripts/build_gait_gallery.py \
  --cfg-path /home/bb/gait/yhgait/configs/opengait_casiab_formal_conservative_eval.yaml \
  --checkpoint-iter 500 \
  --cache-path /home/bb/gait/yhgait/reports/gait_demo_cache_formal_conservative_real_iter00500.npz \
  --master-port 29722
```

```bash
/home/bb/gait/yhgait/.venvs/opengait-gpu/bin/python \
scripts/build_gait_gallery.py \
  --cfg-path /home/bb/gait/yhgait/configs/opengait_casiab_formal_conservative_eval.yaml \
  --checkpoint-iter 1000 \
  --cache-path /home/bb/gait/yhgait/reports/gait_demo_cache_formal_conservative_real_iter01000.npz \
  --master-port 29712
```

```bash
/home/bb/gait/yhgait/.venvs/opengait-gpu/bin/python \
scripts/build_gait_retrieval_analysis.py \
  --cache-path /home/bb/gait/yhgait/reports/gait_demo_cache_formal_conservative_real_iter00500.npz \
  --analysis-path /home/bb/gait/yhgait/reports/gait_demo_analysis_iter00500.json \
  --topk 5
```

```bash
/home/bb/gait/yhgait/.venvs/opengait-gpu/bin/python \
scripts/build_gait_retrieval_analysis.py \
  --cache-path /home/bb/gait/yhgait/reports/gait_demo_cache_formal_conservative_real_iter01000.npz \
  --analysis-path /home/bb/gait/yhgait/reports/gait_demo_analysis_iter01000.json \
  --topk 5
```

如果你在 VS Code 远程连接 Linux 服务器：

- 直接转发 `7860` 端口即可在本地浏览器里看界面

## 如果后面要继续做成业务 demo

最值得做的不是继续拉长训练，而是：

1. 先把 silhouette 生成链路接到真实视频
2. 把单人轨迹切成稳定 gait 序列
3. 用同一个 embedding + retrieval 逻辑接真实视频序列
4. 再单独验证跨场景、跨设备和时序质量
