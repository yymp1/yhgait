# Linux 第三轮真实数据落地结果

## 总结

- 当前 Linux 服务器已经直接下载官方 `CASIA-B silhouette`
- 当前 Linux 服务器已经完成真实原始数据解压、layout 检查与空目录清理
- 当前 Linux 服务器已经基于补丁后的 OpenGait pretreatment 真执行，生成真实 `CASIA-B-pkl`
- 当前真实 pretreatment 产物已经明确 ready：
  - `subject_count = 124`
  - `pkl_count = 13592`
- 当前 GPU 环境已经在**真实数据**上通过：
  - 标准 smoke
  - `2` iter probe
  - `5` iter strict short-run
  - `20` iter baseline small
  - checkpoint resume `20 -> 25`
  - 受控中程恢复 `25 -> 50`
  - 受控中程扩展 `50 -> 100`
  - 受控长跑恢复 `100 -> 300`
  - 受控长跑扩展 `300 -> 500`
- 本轮没有启动长时间正式训练

## 下载与原始数据落地

- 官方下载地址：
  - `http://www.cbsr.ia.ac.cn/GaitDatasetB-silh.zip`
- 下载结果：
  - `downloads/GaitDatasetB-silh.zip`
  - `bytes_after = 659103200`
  - `remote_content_length = 659103200`
  - `status = downloaded`
- 外层 zip 解压后发现其中包含 `124` 个按 subject 切分的 `.tar.gz`
- 内层归档已进一步解包到：
  - `datasets/external/casia_b`
- 中转归档目录已移到：
  - `downloads/GaitDatasetB-silh_subject_archives`

对应材料：

- `reports/casia_b_download.md`

## 原始数据检查与清理

- `scripts/check_casia_b_layout.py` 复检结果：
  - `subject_count = 124`
  - `type_count = 10`
  - `view_count = 11`
  - `png_count = 1118373`
  - `readiness = ready`
- `scripts/clean_casia_b_empty_views.py` 结果：
  - `empty_view_count = 47`
  - `suspicious_view_count = 0`
  - `deleted_count = 47`

这说明：

- 当前原始数据结构已经满足 `subject / type / view / frame.png` 预期
- 本轮只删除了真正为空的 `view` 目录，没有删非空目录

对应材料：

- `reports/casia_b_empty_views.md`

## Pretreatment 真执行

- OpenGait 根目录：
  - `external/OpenGait`
- 执行环境：
  - `.venvs/opengait-gpu/bin/python`
- 补丁状态：
  - `external/OpenGait/datasets/pretreatment.py` 中仍存在 `is unreadable` warning + skip 逻辑
- 真执行输出：
  - `datasets/processed/CASIA-B-pkl`
- pretreatment 日志：
  - `datasets/processed/casia_b_pretreatment.log`

本轮还做了一个最小修复：

- `scripts/run_opengait_pretreatment.py`
  - 让相对 `--python-bin` 在切换 `cwd=external/OpenGait` 后仍能正确解析为绝对路径

日志观察：

- 本轮日志中可见大量 `has no data` 与 `has less than 5 valid data` warning
- 本轮 pretreatment 没有因为这些 warning 中断
- 当前这次下载的数据里未观察到需要触发 `is unreadable` 的 fatal 场景，但补丁仍已保留

## 真实 CASIA-B-pkl 校验

- 当前数据目录：
  - `datasets/processed/CASIA-B-pkl`
- 校验结论：
  - `classification = likely_real`
  - `real_data_status = ready`
- 明确区别于第二轮合成最小数据：
  - 第二轮：`subject_count = 3`, `pkl_count = 3`
  - 本轮：`subject_count = 124`, `pkl_count = 13592`

对应材料：

- `reports/real_data_validation.md`

## GPU 环境复用结果

- 当前首选环境：
  - `.venvs/opengait-gpu/bin/python`
- 当前关键状态：
  - `torch 2.5.1+cu118`
  - `cuda_available = True`
  - `nccl_available = True`
- 机器：
  - `NVIDIA GeForce RTX 2070 SUPER`

对应材料：

- `reports/gpu_compatibility_report.md`
- `reports/linux_third_round/linux_gpu_compat_real_data.json`

## 基于真实数据的 OpenGait 结果

### 标准 smoke

- 报告：
  - `reports/linux_third_round/opengait_smoke_test.md`
- 结果：
  - `import_check = passed`
  - `artifact_check = passed`
  - `data_smoke = passed`
  - `entry_smoke = passed`

### GPU probe

- 报告：
  - `reports/linux_third_round/real_gpu_probe.md`
- 结果：
  - `status = passed`
  - `total_iter = 2`
  - `checkpoint_count = 2`
- 输出目录：
  - `external/OpenGait/output/CASIA-B/GaitSet/real_gpu_probe_2iter`

### 严格受限 short-run

- 报告：
  - `reports/linux_third_round/real_gpu_short_run.md`
- 结果：
  - `status = passed`
  - `total_iter = 5`
  - `checkpoint_count = 5`
- 输出目录：
  - `external/OpenGait/output/CASIA-B/GaitSet/real_gpu_short_run_5iter`

### Baseline small

- 报告：
  - `reports/linux_baseline_small.md`
  - `reports/linux_third_round/baseline_small_real.md`
- 结果：
  - `status = passed`
  - `total_iter = 20`
  - `checkpoint_count = 4`
- 输出目录：
  - `external/OpenGait/output/CASIA-B/GaitSet/baseline_small_real`

### Checkpoint resume

- 报告：
  - `reports/checkpoint_resume_validation.md`
  - `reports/linux_third_round/baseline_small_real_resume.md`
- 结果：
  - `status = passed`
  - `restore_hint = 20`
  - `total_iter = 25`
  - `checkpoint_count = 5`
- 新增 checkpoint：
  - `baseline_small_real-00025.pt`

### 受控中程验证

- 报告：
  - `reports/linux_midrun_validation.md`
  - `reports/training_stability_assessment.md`
  - `reports/linux_third_round/baseline_midrun_50.md`
  - `reports/linux_third_round/baseline_midrun_100.md`
- `25 -> 50` 结果：
  - `status = passed`
  - `restore_hint = 25`
  - `total_iter = 50`
  - `checkpoint_count = 10`
  - 新增 checkpoint：
    - `baseline_small_real-00030.pt`
    - `baseline_small_real-00035.pt`
    - `baseline_small_real-00040.pt`
    - `baseline_small_real-00045.pt`
    - `baseline_small_real-00050.pt`
- `50 -> 100` 结果：
  - `status = passed`
  - `restore_hint = 50`
  - `total_iter = 100`
  - `checkpoint_count = 20`
  - 新增 checkpoint：
    - `baseline_small_real-00055.pt`
    - `...`
    - `baseline_small_real-00100.pt`
- 连续性检查：
  - logs 覆盖 `1 -> 20`、`21 -> 25`、`26 -> 50`、`51 -> 100`
  - TensorBoard event 覆盖 `1 -> 100`
  - `learning_rate` 在 `21 -> 100` 保持 `0.001`，说明 resume 后 scheduler 没有重置
- 资源观察：
  - 对齐真实训练窗口后，显存峰值约 `2769 ~ 2772 MiB`
  - 对齐真实训练窗口后，GPU 利用率峰值约 `42%`
  - 温度约 `54 ~ 57C`
  - 未见 OOM、CUDA error、NCCL error 或 DataLoader 崩溃
  - stderr 中持续存在 PyTorch 2.4+ 的 `ProcessGroupNCCL` 退出清理 warning，但没有导致训练失败

### 受控长跑验证

- 报告：
  - `reports/linux_longrun_validation.md`
  - `reports/night_run_assessment.md`
  - `reports/linux_third_round/baseline_longrun_300.md`
  - `reports/linux_third_round/baseline_longrun_500.md`
- `100 -> 300` 结果：
  - `status = passed`
  - `restore_hint = 100`
  - `total_iter = 300`
  - `checkpoint_count = 60`
  - 新增 checkpoint：`00105 -> 00300`，共 `40` 个
- `300 -> 500` 结果：
  - `status = passed`
  - `restore_hint = 300`
  - `total_iter = 500`
  - `checkpoint_count = 100`
  - 新增 checkpoint：`00305 -> 00500`，共 `40` 个
- 连续性检查：
  - logs 覆盖 `1 -> 20`、`21 -> 25`、`26 -> 50`、`51 -> 100`、`101 -> 300`、`301 -> 500`
  - TensorBoard event 覆盖到 `500`
  - `learning_rate` 在 `101 -> 500` 保持 `0.001`
- 资源观察：
  - `100 -> 300` 有效训练窗口内：`mem 668 ~ 2818 MiB`、`util 13% ~ 37%`、`temp 53 ~ 57C`
  - `300 -> 500` 有效训练窗口内：`mem 677 ~ 2797 MiB`、`util 9% ~ 40%`、`temp 50 ~ 57C`
  - 未见 OOM、CUDA error、NCCL error 或 DataLoader 崩溃
  - stderr 中仍只有退出阶段的 `ProcessGroupNCCL` warning
  - `101 -> 300` 现场保留了两组成功日志/event；主结论以最新一次成功记录和独立 `300 -> 500` 续跑为准

## 当前结论

- 当前 Linux 服务器已经完成“真实数据下载 -> 解压 -> 清理 -> pretreatment -> 真实校验 -> smoke -> probe -> short-run”的整条最小真实链路
- 当前 Linux 服务器已经进一步完成“小规模 baseline -> checkpoint resume -> 中程 100 iter -> 长跑 500 iter”的可回溯验证
- 当前所有核心结论都基于真实 `CASIA-B-pkl`，不是合成最小数据
- 当前已经完成训练前夜跑验证，并具备启动“第一轮正式保守训练”的条件
- 当前仍不应把这份夜跑结果当成正式实验结论

## 下一步最值得做什么

1. 保留并复用当前 `.venvs/opengait-gpu` 环境
2. 如果转入第一轮正式保守训练，核心安全组合继续保持：
   - `batch = [2, 2]`
   - `frames = 16`
   - `num_workers = 1`
   - `save_iter = 5`
3. 暂时不要同时改 `batch / frames / workers / lr / restore` 这几类关键参数
4. 如需继续推进，优先查看：
   - `reports/linux_longrun_validation.md`
   - `reports/night_run_assessment.md`
   - `reports/linux_midrun_validation.md`
   - `reports/training_stability_assessment.md`
