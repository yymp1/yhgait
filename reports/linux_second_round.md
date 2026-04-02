# Linux 第二轮落地结果

## 总结

- 当前 Linux 服务器上未找到真实 `CASIA-B-pkl`
- 当前仓库里的 `datasets/processed/CASIA-B-pkl` 只有 `3` 个 `pkl`，属于第一轮合成最小数据
- 因此，本轮**未能完成基于真实数据的 smoke / probe**
- 但这台机器已经完成 GPU/NCCL 路线验证：
  - 独立 GPU 环境已建立：`/home/bb/gait/yhgait/.venvs/opengait-gpu-cu118`
  - `torch.cuda.is_available() = True`
  - `torch.distributed.is_nccl_available() = True`
  - 官方 `OpenGait` 入口已在最小合成数据上通过 `1` 步 smoke

## 真实数据接通结果

通过本机文件系统搜索与脚本复核，当前只找到下面这一份 `CASIA-B-pkl`：

- `/home/bb/gait/yhgait/datasets/processed/CASIA-B-pkl`

但它的结构是：

- `subject_count = 3`
- `type_count = 1`
- `view_count = 1`
- `pkl_count = 3`

并且抽样文件只有：

- `001/nm-01/000/000.pkl`
- `002/nm-01/000/000.pkl`
- `075/nm-01/000/000.pkl`

结论：

- 这不是历史上那份完整的真实 pretreatment 产物
- 当前 Linux 机器上**真实 `CASIA-B-pkl` 尚未同步到位**
- 当前 Linux 文件系统扫描也**没有找到**另一份真实 `CASIA-B-pkl`、`datasets/external/casia_b` 原始目录或 `GaitDatasetB-silh.zip`

对应材料：

- `reports/linux_second_round/find_real_casiab_paths.json`
- `reports/real_data_smoke.md`

## 标准 smoke / probe 结果

### 基于真实数据

- 未执行
- 原因：真实 `CASIA-B-pkl` 缺失

### 基于当前最小合成数据的辅助诊断

#### CPU 环境

- 环境：`/home/bb/gait/yhgait/.venvs/opengait-cpu`
- 标准 smoke：`entry_smoke = failed`
- 原因：`torch 2.11.0+cpu` 不带 NCCL
- CPU probe：`passed`

对应材料：

- `reports/linux_first_round/opengait_smoke_test.md`
- `reports/linux_first_round/cpu_probe_smoke.md`

#### GPU 环境

- 环境：`/home/bb/gait/yhgait/.venvs/opengait-gpu-cu118`
- 标准 smoke：`entry_smoke = passed`
- 结果：成功进入模型初始化并跑完 `Iteration 00001`

对应材料：

- `reports/linux_second_round/opengait_smoke_test.md`
- `reports/linux_second_round/opengait_smoke_test.json`

## 当前结论

这台 Linux 服务器当前已经不再被 GPU/NCCL 阻塞。

当前主阻塞点已经切换为：

1. 真实 `CASIA-B-pkl` 还没同步到 Linux
2. 因此还不能把 GPU 成功 smoke 误当成“真实数据训练链路已打通”

## 下一步最优先

1. 把历史上那份真实 `datasets/processed/CASIA-B-pkl` 同步到 Linux
2. 或者把真实 `dataset_root` 改到 Linux 上现有的真实路径
3. 再用 GPU 环境重跑：
   - `scripts/run_opengait_smoke_test.py`
   - `scripts/run_opengait_cpu_probe.py`
4. 若真实数据 smoke 通过，再考虑更正式的训练配置
