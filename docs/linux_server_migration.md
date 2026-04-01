# Linux 服务器迁移计划

## 目标

把当前项目迁到一台有 `Linux + NVIDIA GPU + CUDA/NCCL` 的服务器，先跑最小验证，再决定是否进入正式训练。

## 最少需要带走什么

### 代码

直接带整个仓库，但不需要把大体量数据和生成物提交到 GitHub。

### 必带目录

- `configs/`
- `docs/`
- `patches/`
- `reports/`
- `scripts/`
- `src/`
- `README.md`
- `requirements.txt`
- `Dockerfile`

### 必带数据

如果目标是直接训练，不需要重新下载或重新做 pretreatment，只需要拷：

- `datasets/processed/CASIA-B-pkl/`

如果后面还想在服务器上重跑 pretreatment，再额外拷：

- `datasets/external/casia_b/`

## 服务器上最简单的目录准备

```bash
mkdir -p ~/work
cd ~/work
git clone <你的-github-仓库-url> person_orientation_demo
cd person_orientation_demo
git clone https://github.com/ShiqiYu/OpenGait.git external/OpenGait
```

如果 GitHub 还没推上去，也可以先用本地生成的 `git bundle` 兜底迁移：

```bash
scp /Volumes/Data/person_orientation_demo.bundle <user>@<linux-host>:~/work/
ssh <user>@<linux-host>
cd ~/work
git clone person_orientation_demo.bundle person_orientation_demo
cd person_orientation_demo
git clone https://github.com/ShiqiYu/OpenGait.git external/OpenGait
```

如果你只打算直接训练，不打算重跑 pretreatment，那么 `external/OpenGait` 克隆完就够了，不需要把原始 `casia_b` 再复制上去。

## 从这台 Mac 往 Linux 拷数据

最简单的是 `rsync`：

```bash
rsync -avh --progress \
  /Volumes/Data/person_orientation_demo/datasets/processed/CASIA-B-pkl/ \
  <user>@<linux-host>:~/work/person_orientation_demo/datasets/processed/CASIA-B-pkl/
```

如果后面还要重跑 pretreatment，再拷原始数据：

```bash
rsync -avh --progress \
  /Volumes/Data/person_orientation_demo/datasets/external/casia_b/ \
  <user>@<linux-host>:~/work/person_orientation_demo/datasets/external/casia_b/
```

## 服务器环境：先装最小必需项

### 系统层

先确认：

```bash
nvidia-smi
python3 --version
```

如果 `nvidia-smi` 不通，先不要开始 OpenGait。

### Python 环境

建议单独建一个 venv：

```bash
cd ~/work/person_orientation_demo
python3 -m venv .venvs/opengait_train
source .venvs/opengait_train/bin/activate
python -m pip install --upgrade pip
```

### 先装 PyTorch

这一步要按服务器的 CUDA 版本来，不要照搬 Mac 上的 CPU 轮子。先到 PyTorch 官方页面选择和服务器 CUDA 匹配的安装命令。

### 再装当前项目已经验证过的其余依赖

```bash
pip install torchvision pyyaml tensorboard einops kornia matplotlib imageio scikit-learn opencv-python-headless tqdm
```

如果你想尽量和本地保持一致，可参考：

- `reports/opengait_training_env_freeze.txt`

## 服务器上先跑什么验证

### 1. 先确认 GPU 可见

```bash
python - <<'PY'
import torch
print("cuda_available =", torch.cuda.is_available())
print("device_count =", torch.cuda.device_count())
PY
```

成功标准：

- `cuda_available = True`
- `device_count >= 1`

### 2. 先限制只用 1 张卡做 smoke

当前 OpenGait 这份 `main.py` 会检查：

- `torch.distributed.get_world_size() == torch.cuda.device_count()`

所以第一次验证最稳的方式是只暴露一张卡：

```bash
export CUDA_VISIBLE_DEVICES=0
```

### 3. 跑当前仓库自带的训练入口 smoke

```bash
cd ~/work/person_orientation_demo
source .venvs/opengait_train/bin/activate

python scripts/run_opengait_smoke_test.py \
  --python-bin .venvs/opengait_train/bin/python \
  --opengait-root external/OpenGait \
  --config-path configs/opengait_casiab_smoke.yaml \
  --reports-dir reports
```

成功标准：

- `reports/opengait_smoke_test.md` 里看到
  - `import_check = passed`
  - `artifact_check = passed`
  - `data_smoke = passed`
  - `entry_smoke = passed`

### 4. 如果你要重跑 pretreatment

只有在服务器上重新处理原始 CASIA-B 时才需要这一步。

先应用本地补丁：

```bash
cd ~/work/person_orientation_demo/external/OpenGait
git apply ../../patches/pretreatment_skip_bad_frames.patch
```

然后再按 README 里的 pretreatment 命令去跑。

## 看到什么结果算迁移成功

满足下面 4 条，就算迁移第一阶段成功：

1. Linux 上 `torch.cuda.is_available() == True`
2. `datasets/processed/CASIA-B-pkl/` 被正确识别
3. `scripts/run_opengait_smoke_test.py` 的 `entry_smoke` 通过
4. `external/OpenGait/output/...` 下开始出现日志和 checkpoint

## 最后再进入正式训练

只有在 smoke 通过之后，才建议切换到更完整的训练配置。第一次不要一上来就跑全量长训。
