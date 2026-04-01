# OpenGait Smoke Test 报告

- Python: `/Users/yymp/.venvs/person_orientation_demo/opengait_pretreatment/bin/python`
- OpenGait: `/Volumes/Data/person_orientation_demo/external/OpenGait`
- 配置: `/Volumes/Data/person_orientation_demo/configs/opengait_casiab_smoke.yaml`
- 训练入口端口: `29531`

## 依赖检查
- 状态：`passed`
- `torch`: `ok` 2.11.0
- `yaml`: `ok` 6.0.3
- `tensorboard`: `ok` 2.20.0
- `torchvision`: `ok` 0.26.0
- `einops`: `ok` 0.8.2
- `kornia`: `ok` 0.8.2
- `matplotlib`: `ok` 3.10.8
- `imageio`: `ok` 2.37.3
- `sklearn`: `ok` 1.8.0
- `cv2`: `ok` 4.13.0
- `numpy`: `ok` 2.4.4
- `tqdm`: `ok` 4.67.3

## Pretreatment 产物检查
- 状态：`passed`
- 数据根目录：`/Volumes/Data/person_orientation_demo/datasets/processed/CASIA-B-pkl`
- subject 数：`124`
- type 数：`10`
- view 数：`11`
- pkl 数：`13592`
- 抽样：
  - `/Volumes/Data/person_orientation_demo/datasets/processed/CASIA-B-pkl/001/bg-01/000/000.pkl` shape=[89, 64, 64] dtype=uint8
  - `/Volumes/Data/person_orientation_demo/datasets/processed/CASIA-B-pkl/062/nm-06/162/162.pkl` shape=[77, 64, 64] dtype=uint8
  - `/Volumes/Data/person_orientation_demo/datasets/processed/CASIA-B-pkl/124/nm-06/180/180.pkl` shape=[74, 64, 64] dtype=uint8

## 数据集 Smoke
- 状态：`passed`
- train 序列数：`8107`
- test 序列数：`5485`
- train 身份数：`74`
- 样例序列：`['001', 'bg-01', '000', ['../../datasets/processed/CASIA-B-pkl/001/bg-01/000/000.pkl']]`
- batch 形状：`[[[8, 64, 64], [8, 64, 64]]]`

## 官方训练入口 Smoke
- 状态：`failed`
- 返回码：`1`

```bash
/Users/yymp/.venvs/person_orientation_demo/opengait_pretreatment/bin/python opengait/main.py --cfgs /Volumes/Data/person_orientation_demo/configs/opengait_casiab_smoke.yaml --phase train --log_to_file
```

```text
xFormers not available
xFormers not available
/Users/yymp/.venvs/person_orientation_demo/opengait_pretreatment/lib/python3.11/site-packages/torch/distributed/distributed_c10d.py:1788: UserWarning: Attempted to get default timeout for nccl backend, but NCCL support is not compiled
  timeout = _get_default_timeout(backend)
Traceback (most recent call last):
  File "/Volumes/Data/person_orientation_demo/external/OpenGait/opengait/main.py", line 62, in <module>
    torch.distributed.init_process_group('nccl', init_method='env://')
  File "/Users/yymp/.venvs/person_orientation_demo/opengait_pretreatment/lib/python3.11/site-packages/torch/distributed/c10d_logger.py", line 83, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/Users/yymp/.venvs/person_orientation_demo/opengait_pretreatment/lib/python3.11/site-packages/torch/distributed/c10d_logger.py", line 97, in wrapper
    func_return = func(*args, **kwargs)
                  ^^^^^^^^^^^^^^^^^^^^^
  File "/Users/yymp/.venvs/person_orientation_demo/opengait_pretreatment/lib/python3.11/site-packages/torch/distributed/distributed_c10d.py", line 1838, in init_process_group
    default_pg, _ = _new_process_group_helper(
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/yymp/.venvs/person_orientation_demo/opengait_pretreatment/lib/python3.11/site-packages/torch/distributed/distributed_c10d.py", line 2099, in _new_process_group_helper
    raise RuntimeError("Distributed package doesn't have NCCL built in")
RuntimeError: Distributed package doesn't have NCCL built in
```
