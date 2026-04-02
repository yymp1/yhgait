# formal_conservative_real checkpoint 对照评估

## 结论

- 本轮已使用同一套官方评估配置，真实评估：
  - `formal_conservative_real-00500.pt`
  - `formal_conservative_real-00900.pt`
  - `formal_conservative_real-01000.pt`
- `500 -> 1000` 的总提升是真实存在的，不是白跑
- 但 `900 -> 1000` 的边际提升已经明显变小
- 因此当前更合理的收口判断是：
  - `先不要继续训练`
  - 当前提升不足以支持直接续到 `1500 iter`

## 使用的评估配置

- 配置文件：
  - `configs/opengait_casiab_formal_conservative_eval.yaml`
- 说明：
  - 未更换评估配置
  - 仅通过官方 CLI `--iter` 切换 checkpoint

## 实际命令

执行目录：

- `external/OpenGait`

实际命令模板：

```bash
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=<port> \
WORLD_SIZE=1 \
RANK=0 \
LOCAL_RANK=0 \
/home/bb/gait/yhgait/.venvs/opengait-gpu/bin/python \
opengait/main.py \
  --cfgs /home/bb/gait/yhgait/configs/opengait_casiab_formal_conservative_eval.yaml \
  --phase test \
  --log_to_file \
  --iter <iter>
```

本轮实际使用：

- `--iter 500`
- `--iter 900`
- `--iter 1000`

## 对照表

| checkpoint | NM@R1 | BG@R1 | CL@R1 | return code | 评估日志 |
| --- | ---: | ---: | ---: | ---: | --- |
| `formal_conservative_real-00500.pt` | `16.06%` | `12.12%` | `6.05%` | `0` | `external/OpenGait/output/CASIA-B/GaitSet/formal_conservative_real/logs/2026-04-01-22-02-51.txt` |
| `formal_conservative_real-00900.pt` | `22.97%` | `16.58%` | `7.69%` | `0` | `external/OpenGait/output/CASIA-B/GaitSet/formal_conservative_real/logs/2026-04-01-22-03-52.txt` |
| `formal_conservative_real-01000.pt` | `23.95%` | `17.17%` | `8.00%` | `0` | `external/OpenGait/output/CASIA-B/GaitSet/formal_conservative_real/logs/2026-04-01-22-04-54.txt` |

原始汇总 CSV：

- `reports/checkpoint_eval_compare/results.csv`

## 关键对比

### 500 -> 900

- `NM@R1`: `+6.91`
- `BG@R1`: `+4.46`
- `CL@R1`: `+1.64`

这说明：

- 训练从 `500` 拉到 `900` 是有明显收益的
- `01000` 不是靠偶然抖动赢出来的，前面这段确实在变好

### 900 -> 1000

- `NM@R1`: `+0.98`
- `BG@R1`: `+0.59`
- `CL@R1`: `+0.31`

这说明：

- 最后 `100 iter` 仍有小幅提升
- 但这个提升已经很小，和 `500 -> 900` 这段相比明显变缓

## 与此前曲线分析合并后的判断

结合：

- `reports/training_decision_review.md` 中的训练曲线判断
- 本轮的 checkpoint 对照评估结果

当前最稳妥的结论是：

1. `1000 iter` 相比 `500 iter` 是更好的 checkpoint
2. 但模型在 `900 -> 1000` 已经表现出平台化迹象
3. 因此现在缺少足够证据支持继续直接拉到 `1500 iter`

## 最终主结论

- `建议先不要继续训练，当前提升不足以支持续训。`
