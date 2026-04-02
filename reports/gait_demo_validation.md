# 步态检索 GUI Demo 验证记录

## 本轮真实验证了什么

1. 真实加载 `formal_conservative_real-01000.pt`
2. 真实读取 `CASIA-B-pkl`
3. 真实抽取 test 集 `5485` 条序列 embedding
4. 真实执行一次 Top-K 检索
5. 真实启动 GUI，并确认页面返回 `HTTP 200`

## 关键产物

- checkpoint：
  - `external/OpenGait/output/CASIA-B/GaitSet/formal_conservative_real/checkpoints/formal_conservative_real-01000.pt`
- embedding 缓存：
  - `reports/gait_demo_cache_formal_conservative_real_iter01000.npz`
- GUI 入口：
  - `app/gradio_gait_demo.py`
- CLI 检索入口：
  - `scripts/run_gait_retrieval_demo.py`

## 真实缓存构建结果

实际命令：

```bash
/home/bb/gait/yhgait/.venvs/opengait-gpu/bin/python \
scripts/build_gait_gallery.py \
  --cfg-path /home/bb/gait/yhgait/configs/opengait_casiab_formal_conservative_eval.yaml \
  --checkpoint-iter 1000 \
  --cache-path /home/bb/gait/yhgait/reports/gait_demo_cache_formal_conservative_real_iter01000.npz \
  --master-port 29712
```

真实结果：

- `sequence_count = 5485`
- `embedding_shape = (5485, 256, 62)`
- cache 已生成

## 真实 Top-K 检索验证

实际命令：

```bash
python3 scripts/run_gait_retrieval_demo.py \
  --cache-path /home/bb/gait/yhgait/reports/gait_demo_cache_formal_conservative_real_iter01000.npz \
  --topk 5 \
  --seed 13
```

本次随机到的 probe：

- `091 / bg-01 / 108`

Top-5：

| rank | subject | type | view | distance | hit |
| --- | --- | --- | --- | ---: | --- |
| 1 | `091` | `nm-01` | `108` | `0.1036` | `hit` |
| 2 | `091` | `nm-03` | `108` | `0.1085` | `hit` |
| 3 | `091` | `nm-02` | `108` | `0.1167` | `hit` |
| 4 | `091` | `nm-04` | `108` | `0.1210` | `hit` |
| 5 | `098` | `nm-03` | `108` | `0.1441` | `miss` |

这说明：

- 检索逻辑是真实接到模型和数据的
- 至少在这次随机样本上，可以直观看到同人结果被排到了前面

## GUI 启动验证

实际命令：

```bash
/home/bb/gait/yhgait/.venvs/opengait-gpu/bin/python \
app/gradio_gait_demo.py \
  --cache-path /home/bb/gait/yhgait/reports/gait_demo_cache_formal_conservative_real_iter01000.npz \
  --server-name 127.0.0.1 \
  --server-port 7860
```

探活结果：

- `http://127.0.0.1:7860` 返回 `HTTP 200`
- GUI 进程存在

## 当前可以怎么理解这个 demo

- 它已经足够让人“看见步态识别效果”
- 但看到的是：
  - `数据集上的检索效果`
  - 不是 `真实业务视频系统效果`
