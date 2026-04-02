# formal_conservative_real 正式评估报告

## 结论

- 本轮已基于真实 `CASIA-B-pkl` 和 `formal_conservative_real-01000.pt` 完成一次真实官方评估
- OpenGait 官方评估入口已打通
- 本次结果属于“正式保守训练后的评估与结果收口”，不属于最终实验结论

## 评估入口检查

- 官方入口：`external/OpenGait/opengait/main.py`
- 评估模式：`--phase test`
- 当前仓库要求：
  - `torch.distributed.init_process_group('nccl', init_method='env://')`
  - `WORLD_SIZE == evaluator_cfg.sampler.batch_size`
- checkpoint 加载方式：
  - `evaluator_cfg.restore_hint = 1000`
  - `evaluator_cfg.save_name = formal_conservative_real`
  - 最终会加载 `output/CASIA-B/GaitSet/formal_conservative_real/checkpoints/formal_conservative_real-01000.pt`

本轮使用的最小评估配置：

- `configs/opengait_casiab_formal_conservative_eval.yaml`

关键字段：

- `eval_func = evaluate_indoor_dataset`
- `transform = BaseSilCuttingTransform`
- `sampler.batch_size = 1`
- `metric = euc`

## 实际命令

```bash
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=29581 \
WORLD_SIZE=1 \
RANK=0 \
LOCAL_RANK=0 \
/home/bb/gait/yhgait/.venvs/opengait-gpu/bin/python \
opengait/main.py \
  --cfgs /home/bb/gait/yhgait/configs/opengait_casiab_formal_conservative_eval.yaml \
  --phase test \
  --log_to_file
```

执行目录：

- `external/OpenGait`

## 实际结果

- 结果：`success`
- return code：`0`
- 真实数据：`datasets/processed/CASIA-B-pkl`
- checkpoint：
  - `external/OpenGait/output/CASIA-B/GaitSet/formal_conservative_real/checkpoints/formal_conservative_real-01000.pt`
- 评估日志：
  - `external/OpenGait/output/CASIA-B/GaitSet/formal_conservative_real/logs/2026-04-01-21-21-48.txt`

日志中可见：

- `Restore Parameters from output/CASIA-B/GaitSet/formal_conservative_real/checkpoints/formal_conservative_real-01000.pt !!!`
- `===Rank-1 (Exclude identical-view cases)===`

本次最小评估结果：

- `NM@R1 = 23.95%`
- `BG@R1 = 17.17%`
- `CL@R1 = 8.00%`

分视角数组已写入评估日志。

## 已知 warning

- `xFormers not available`
  - 对本次 `GaitSet` 评估不构成阻塞
- `ProcessGroupNCCL` 退出 warning
  - 本次评估正常结束，未导致失败

## 本轮判断

- 这次不只是“入口能启动”，而是已经真实完成了基于 `01000.pt` 的一次官方评估
- 因此当前 `formal_conservative_real` 已具备“训练完成 + 评估已验证”的可交接状态
