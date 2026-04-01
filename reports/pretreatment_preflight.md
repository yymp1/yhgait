# OpenGait pretreatment 预检报告

- 预检结论：`ready`

## 检查项
- `opengait_root`：`passed`，OpenGait 仓库和关键文件齐全
  说明：`repo=/Volumes/Data/person_orientation_demo/external/OpenGait`
- `casia_b_layout`：`passed`，CASIA-B 原始目录检查通过
  说明：`subject_count=124; type_count=10; view_count=11; png_count=1118371; readiness=ready`
- `empty_view_scan`：`passed`，空目录和 suspicious 目录检查通过
  说明：`empty_view_count=0; suspicious_view_count=0; affected_subject_count=0; affected_type_count=0`
- `pretreatment_dry_run`：`passed`，pretreatment dry-run 命令已生成
  说明：`/Library/Developer/CommandLineTools/usr/bin/python3 /Volumes/Data/person_orientation_demo/external/OpenGait/datasets/pretreatment.py --input_path /Volumes/Data/person_orientation_demo/datasets/external/casia_b --output_path /Volumes/Data/person_orientation_demo/datasets/processed/CASIA-B-pkl --dataset CASIAB --n_workers 4 --img_size 64 --log_file /Volumes/Data/person_orientation_demo/datasets/processed/casia_b_pretreatment.log`

## 下一条 dry-run 命令

```bash
/Library/Developer/CommandLineTools/usr/bin/python3 /Volumes/Data/person_orientation_demo/scripts/run_opengait_pretreatment.py --opengait-root external/OpenGait --input-path datasets/external/casia_b --output-path datasets/processed/CASIA-B-pkl --dataset-name CASIAB --n-workers 4 --img-size 64 --dry-run
```

## OpenGait 原始 pretreatment 命令

```bash
/Library/Developer/CommandLineTools/usr/bin/python3 /Volumes/Data/person_orientation_demo/external/OpenGait/datasets/pretreatment.py --input_path /Volumes/Data/person_orientation_demo/datasets/external/casia_b --output_path /Volumes/Data/person_orientation_demo/datasets/processed/CASIA-B-pkl --dataset CASIAB --n_workers 4 --img_size 64 --log_file /Volumes/Data/person_orientation_demo/datasets/processed/casia_b_pretreatment.log
```

## 通过后可执行命令

```bash
/Library/Developer/CommandLineTools/usr/bin/python3 /Volumes/Data/person_orientation_demo/scripts/run_opengait_pretreatment.py --opengait-root external/OpenGait --input-path datasets/external/casia_b --output-path datasets/processed/CASIA-B-pkl --dataset-name CASIAB --n-workers 4 --img-size 64 --execute
```
