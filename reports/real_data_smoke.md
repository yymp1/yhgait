# 真实数据 Smoke 结果

## 结论

- 本轮已经完成基于真实 `CASIA-B-pkl` 的 smoke / probe / strict short-run
- 当前结论不再是 blocked，而是：
  - `real_data_status = ready`
  - `smoke = passed`
  - `probe = passed`
  - `strict_short_run = passed`

## 已确认的事实

当前 Linux 机器上当前使用的真实 `CASIA-B-pkl` 路径是：

- `/home/bb/gait/yhgait/datasets/processed/CASIA-B-pkl`

它当前是本轮 Linux 服务器上重新下载 silhouette、重新执行 pretreatment 后得到的真实产物，数量是：

- `subject_count = 124`
- `pkl_count = 13592`

抽样 `pkl` 示例见：

- `reports/real_data_validation.md`

这次结果已经与第二轮当时那份“只有 3 个 pkl 的合成最小数据”明确区分开。第二轮的旧状态是：

- `subject_count = 3`
- `pkl_count = 3`

## 真实数据来源与 pretreatment

- 官方下载：
  - `downloads/GaitDatasetB-silh.zip`
- 原始目录：
  - `datasets/external/casia_b`
- pretreatment 日志：
  - `datasets/processed/casia_b_pretreatment.log`

本轮还做了这些真实前置动作：

- 解压外层 zip 与内层 `124` 个 subject 归档
- 安全清理 `47` 个真正为空的 `view` 目录
- 用补丁后的 `external/OpenGait/datasets/pretreatment.py` 真执行 pretreatment

## 基于真实数据的执行结果

### 标准 smoke

- 环境：`/home/bb/gait/yhgait/.venvs/opengait-gpu/bin/python`
- 报告：`reports/linux_third_round/opengait_smoke_test.md`
- 结果：
  - `import_check = passed`
  - `artifact_check = passed`
  - `data_smoke = passed`
  - `entry_smoke = passed`

### GPU probe

- 环境：`/home/bb/gait/yhgait/.venvs/opengait-gpu/bin/python`
- 报告：`reports/linux_third_round/real_gpu_probe.md`
- 结果：
  - `status = passed`
  - `total_iter = 2`
  - `checkpoint_count = 2`

### 严格受限 short-run

- 环境：`/home/bb/gait/yhgait/.venvs/opengait-gpu/bin/python`
- 报告：`reports/linux_third_round/real_gpu_short_run.md`
- 结果：
  - `status = passed`
  - `total_iter = 5`
  - `checkpoint_count = 5`

## 当前判断

- 当前 Linux 现场已经不再受“真实数据尚未同步到位”阻塞
- 当前通过的是**真实数据链路**，不是合成最小数据链路
- 但这仍然只是短时验证，不应把它直接等价为长时间正式训练已经准备完毕

## 下一步

1. 继续保留当前 `opengait-gpu` 环境，不要切回 CPU-only 路线
2. 若要再前进一步，优先做：
   - 更保守的真实数据小规模验证
   - 更明确的 checkpoint / 日志复核
3. 仍然不要直接启动长时间正式训练
