# 人物背影/姿态自动识别 Demo

这是一个面向 `macOS Apple Silicon` 的本地最小可运行 demo，功能只覆盖：

1. 读取本地视频文件
2. 检测视频中的 `person`
3. 对每个 `person` 做姿态关键点估计
4. 基于规则输出 `front / back / side / unknown`
5. 在视频帧上绘制 `bbox`、`skeleton`、`orientation label`
6. 把结果视频写到 `outputs/`

## 目录

- `app.py`：命令行入口
- `src/detector.py`：`ultralytics` 人体检测
- `src/pose_estimator.py`：`mediapipe pose` 姿态估计
- `src/orientation.py`：规则式朝向判断
- `src/visualize.py`：框、骨架、标签绘制
- `scripts/batch_collect.py`：批量步态采集处理
- `scripts/organize_tracks.py`：单视频轨迹整理
- `scripts/prepare_for_opengait.py`：把当前 `data/` 整理成更接近 OpenGait 的目录
- `scripts/check_casia_b_layout.py`：检查 CASIA-B 原始目录是否接近 OpenGait 预处理期望结构
- `scripts/clean_casia_b_empty_views.py`：扫描并可选清理 CASIA-B 中真正为空的 view 目录
- `scripts/setup_opengait_pretreatment_env.py`：创建 OpenGait pretreatment 专用最小 Python 环境
- `scripts/run_opengait_pretreatment.py`：生成或执行外部 OpenGait pretreatment 命令
- `scripts/preflight_opengait_pretreatment.py`：pretreatment 真执行前的一键预检入口
- `configs/`：OpenGait 准备阶段的路径模板
- `datasets/external/`：公开 gait 数据集预留目录
- `datasets/processed/`：公开数据集 pretreatment 输出目录
- `samples/`：输入视频目录
- `outputs/`：输出视频目录

## 依赖选择说明

- 运行时核心依赖是 `ultralytics + mediapipe + opencv-python + numpy`
- `mediapipe` 会额外带上 `opencv-contrib-python`，这是它自身依赖，属于正常现象
- 姿态模块默认使用 `MediaPipe Pose model_complexity=1`，避免 `model_complexity=0` 首次运行时额外下载 `pose_landmark_lite.tflite`
- Docker 里不用默认的 `pip install` 方案，而是改用 `uv pip --torch-backend cpu`
  原因：在 `linux/arm64` 下默认解析 `torch` 时，可能会把一串 `nvidia-*` 包带进镜像；这里显式固定为 `CPU-only`，更符合本项目的 CPU-first 目标

## 本地运行

如果系统已经有 `python3.11`：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py --input samples/test.mp4 --output outputs/result.mp4
```

如果本机没有现成的 `python3.11`，推荐用 `uv`：

```bash
uv python install 3.11
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python app.py --input samples/test.mp4 --output outputs/result.mp4
```

说明：

- 仓库已经附带了一个最小烟测视频 `samples/test.mp4`，可直接用于验证整条处理链路
- 这个样例视频是由 `ultralytics` 自带的 `bus.jpg` 生成的短循环片段，适合做功能验证；如果你要测试真实“背影/侧身/正面”判断，建议换成自己的真实人物视频
- 第一次运行时，`ultralytics` 可能会自动下载 `yolov8n.pt` 权重

## 步态数据采集

如果你的目标是后续接入 `OpenGait` 或其他 gait recognition 框架，建议尽量采集下面这种更接近标准步态数据的数据：

- 单人、全身入镜、头到脚尽量完整
- 相机尽量固定，避免大幅平移和变焦
- 背景尽量简单，光照尽量稳定
- 人物和相机保持相对稳定距离，不要忽远忽近
- 同一个人尽量覆盖多条序列，而不是只录一条
- 尽量包含正后方、偏后方、偏侧方等不同角度，但仍要保证步行过程连续
- 每段序列尽量包含完整走路过程，不要只截取很短的一小段

更具体一点，后续用于 gait 模型的数据，最好满足“可分割成稳定连续序列”的条件，而不是零散碎片视频。

## 当前适用范围

如果你现在只有“每人一段背面或偏侧面自拍视频”，这个项目更适合做下面这些事：

- 自动检测人物、加 `track_id`、切出连续序列
- 生成后续可人工整理的 `track clip` 或逐帧 `crop`
- 给每段序列打上 `front / back / side / unknown` 这种粗粒度朝向标签
- 帮你做步态数据前处理、筛选、归档和人工命名

它不适合直接做这些事：

- 不适合直接训练 gait 身份识别模型
- 不适合把少量自拍视频当成完整训练集
- 不适合拿单条背面视频去学“这个人就是谁”

原因很简单：

- gait recognition 通常需要更多序列、更多步态变化、更多覆盖角度
- 只有一段自拍视频时，样本量太小，模型很容易过拟合
- 背景、衣物、拍摄设备、裁切方式都会成为强干扰因素
- 你当前的数据更适合做“采集、清洗、组织、筛选”，而不是直接训练

## Tracking 与导出

现在可以直接用当前项目做 gait 前处理导出：

```bash
python app.py \
  --input samples/test.mp4 \
  --output outputs/result.mp4 \
  --enable-tracking \
  --export-tracks \
  --export-format clip
```

也可以导出逐帧图片：

```bash
python app.py \
  --input samples/test.mp4 \
  --output outputs/result.mp4 \
  --enable-tracking \
  --export-tracks \
  --export-format frames
```

建议的过滤条件如下：

- `--min-track-length 20`：最小连续帧数
- `--min-bbox-size 64`：最小 bbox 尺寸
- `--track-conf 0.35`：检测或跟踪置信度阈值
- `--tracker-max-missing 10`：短暂丢失多少帧后断开轨迹

导出后的目录通常可以组织成这样：

```text
artifacts/
  clips/
    track_0001.mp4
    track_0002.mp4
  manifest.tsv
  tracks/
    0001/
      frame_000012.jpg
      frame_000013.jpg
      metadata.json
    0002/
      frame_000120.jpg
      frame_000121.jpg
      metadata.json
```

如果你要进一步做人工整理，可以再把导出的轨迹搬到这样的结构：

```text
data/
  unknown_id/
    manifest.tsv
    seq_001/
      clip.mp4
      frames/
    seq_002/
  person_001/
    seq_001/
```

仓库里提供了 `scripts/organize_tracks.py`，可以把 `artifacts/` 里的轨迹整理成便于人工改名和后续接 gait 框架的结构：

```bash
python scripts/organize_tracks.py --source artifacts --output data
```

## 批量采集

如果你要把 `samples/` 下面的真人视频一次性都跑完，推荐按下面的批量流程处理：

```bash
python scripts/batch_collect.py \
  --input-dir samples \
  --outputs-dir outputs \
  --artifacts-dir artifacts \
  --data-dir data \
  --reports-dir reports
```

这类批处理通常会做下面几件事：

1. 自动扫描 `samples/` 下的常见视频格式，例如 `mp4 / mov / m4v`
2. 对每个视频分别执行现有主流程，开启 `tracking` 和 `export`
3. 统一生成每个视频自己的结果文件
4. 自动调用 `scripts/organize_tracks.py`，把导出的轨迹整理到 `data/`
5. 输出汇总报告，方便你快速筛选较差序列
6. 如果结果已存在，默认记为 `skipped`；加 `--overwrite` 可强制重跑

建议的输出目录结构如下：

```text
outputs/
  <video_stem>_result.mp4
artifacts/
  <video_stem>/
    clips/
    tracks/
    manifest.tsv
data/
  <video_stem>/
    unknown_id/
      seq_001/
      seq_002/
reports/
  summary.tsv
  summary.md
```

查看结果时，可以按这个顺序看：

1. 先看 `outputs/<video_stem>_result.mp4`，确认检测、`track_id` 和骨架有没有明显跑偏
2. 再看 `artifacts/<video_stem>/manifest.tsv`，确认每条 track 的帧数和导出是否正常
3. 最后看 `data/<video_stem>/unknown_id/`，检查是否已经整理成后续可人工命名的序列结构

汇总报告里，建议重点关注这些字段：

- `source_video`
- `track_id`
- `frame_count`
- `avg_bbox_area`
- `avg_confidence`
- `kept` 或 `pass`
- `status`

筛掉较差轨迹时，可以优先排除这些情况：

- `frame_count` 太短，通常说明序列不够连续
- `avg_bbox_area` 太小，通常说明人太远或检测框不稳定
- `avg_confidence` 偏低，通常说明跟踪质量不稳或遮挡较重
- 结果视频里明显断轨、跳 ID、裁切错误的序列

当前这套轻量 tracking 的已知限制也要一起记住：

- 它是单视频内的轻量 `IoU` 跟踪，不做 ReID
- 多人交叉、长遮挡、快速转身、出画再入画时，`track_id` 可能切换
- 低分辨率、远景、背景复杂的视频，导出序列可能偏短或被过滤掉
- 它更适合做步态数据的前处理、筛选和整理，不适合做高精度 MOT

如果你想强制重跑已经处理过的视频，可以加：

```bash
python scripts/batch_collect.py \
  --input-dir samples \
  --outputs-dir outputs \
  --artifacts-dir artifacts \
  --data-dir data \
  --reports-dir reports \
  --overwrite
```

## OpenGait 对接准备

这一章把当前项目和 OpenGait 常见数据形态对齐，方便你后续真正接训练工程时少走弯路。

### 当前项目导出格式

本项目现在导出的主产物是这几类：

```text
outputs/<video_stem>_result.mp4
artifacts/<video_stem>/
  manifest.tsv
  clips/track_0001.mp4
  tracks/0001/frame_000012.jpg
  tracks/0001/metadata.json
data/<video_stem>/unknown_id/
  manifest.tsv
  seq_001/
    clip.mp4
    metadata.json
    frames/
reports/summary.tsv
reports/summary.md
```

这些结果更偏向“视频侧前处理”：

- `outputs/` 是可视化结果，给你做人眼检查
- `artifacts/` 是单视频的 track 导出，中间态更适合排错
- `data/` 是按 `identity / sequence` 组织后的序列集合，适合后续人工重命名和再加工
- `reports/` 是批量汇总，用来快速筛掉低质量轨迹

### OpenGait 常见输入

从 OpenGait 官方仓库的 `datasets/CASIA-B/README.md` 可以看到，CASIA-B 原始数据的层级大致是：

```text
CASIA-B/
  001 (subject)
    bg-01 (type)
      000 (view)
        001-bg-01-000-001.png
        ...
```

OpenGait 里处理后的 CASIA-B 还会被预处理成 pkl 结构，例如：

```text
casiab-128-end2end/
  001/
    bg-01/
      000/
        000-aligned-sils.pkl
        000-ratios.pkl
        000-rgbs.pkl
        000-sils.pkl
```

也就是说，OpenGait 不是只看“有没有一个视频文件”，而是更看重：

- `subject / identity`
- `type / sequence`
- `view`
- 经过统一预处理后的序列文件

### 现在还差什么

和 OpenGait 的常见输入相比，我们当前项目还差这些步骤：

1. 把当前 `data/` 里的 `seq_*` 进一步整理成更稳定的 `subject / sequence / view` 结构
2. 给每个序列补一个明确的视角占位，例如 `000 / 018 / 036 / 054 / 072 / 090 / 108 / 126 / 144 / 162 / 180`，或者先统一用 `000`
3. 为后续训练工程准备 `raw` 与 `prepared` 两层目录，避免把前处理结果和训练输入混在一起
4. 如果以后要走标准 OpenGait 预处理，还要把 crop / silhouette / ratio / aligned-silhouette 这些中间结果补出来

所以，当前项目的定位不是“直接产出可训练 OpenGait 数据”，而是“把原始自拍视频先整理成更接近 OpenGait 的前处理输入”。

### 当前脚本如何整理成更接近 OpenGait 的结构

现在可以直接把现有 `data/` 转成一版更规整的 `subject / sequence / view / frame` 目录：

```bash
python scripts/prepare_for_opengait.py --input-dir data --output-dir opengait_ready --copy-clips
```

默认输出会是：

```text
opengait_ready/
  manifest.tsv
  subject_map.tsv
  subject_001/
    seq_001/
      000/
        000001.jpg
        000002.jpg
      clip.mp4
      metadata.json
```

这里有几个关键点：

- `subject_001` 是脚本生成的标准化身份目录，具体映射关系写在 `subject_map.tsv`
- `seq_001` 是同一身份下的序列编号
- `000` 是当前阶段的视角占位，后续如果你知道真实视角，可以再人工改成 `018 / 036 / 090 / 180` 这类编号
- `manifest.tsv` 会保留 `source_video / source_identity / source_sequence / frame_count / status` 等字段，方便你继续回溯来源

如果你已经把 `data/` 里的 `identity` 人工改成了跨视频一致的名字，例如都改成 `person_001`，可以用：

```bash
python scripts/prepare_for_opengait.py \
  --input-dir data \
  --output-dir opengait_ready \
  --subject-key-mode identity \
  --copy-clips
```

这样同名身份就会被合并到同一个 `subject_xxx` 下。默认模式是 `auto`，会把 `unknown_id` 按视频拆开，避免误把不同视频里的陌生人混成一个身份。

如果你的 `data/` 目录里还保留了早期 smoke 数据或临时目录，也可以显式排除：

```bash
python scripts/prepare_for_opengait.py \
  --input-dir data \
  --output-dir opengait_ready \
  --exclude-collections unknown_id \
  --copy-clips
```

### CASIA-B 与 OUMVLP 怎么选

这两个公开数据集都已经被 OpenGait 支持，OpenGait 官方也明确把它们列在支持列表里。

CASIA-B 更适合你先上手，原因是：

- 目录结构更经典，`subject / type / view` 的概念直观
- 很多 gait 论文和工具都以它作为入门基准
- 你用当前项目做目录对齐时，更容易理解“一个人、多次序列、多视角”到底是什么意思

OUMVLP 更适合你在 CASIA-B 跑通后再接，原因是：

- 数据规模更大，更接近“训练工程”而不是“快速入门”
- 对数据组织、预处理和存储要求更高
- 更适合在你已经确认 OpenGait 管线跑通后，再做规模化验证

如果你只想先选一个公开数据集来对接 OpenGait，我建议先从 `CASIA-B` 开始，再扩到 `OUMVLP`。

### 公开数据集和自拍视频各自适合什么

公开数据集更适合：

- 训练 gait 模型
- 做标准评测
- 对比不同模型和不同预处理方式
- 验证 OpenGait 的完整训练 / 测试流水线

你自己的少量自拍视频更适合：

- track 导出和可视化检查
- demo 展示
- 给训练后的模型做小规模验证
- 看检索效果、看失败样例、看姿态和朝向标签是否合理

它们不适合被混成一类来用，因为自拍视频通常只有少量样本、视角单一、遮挡和背景变化也不够丰富，直接拿来训练 gait 模型很容易过拟合。

### 后续真正训练前还差什么

在真正接 OpenGait 训练前，通常还要补齐这些东西：

- 明确身份目录，不再长期使用 `unknown_id`
- 明确每个序列的视角编号
- 统一裁切和分辨率
- 明确是用 RGB、silhouette，还是两者都用
- 把 OpenGait 需要的目录结构和配置路径写成单独的训练工程配置
- 先用公开数据集跑通训练与评测，再把你自己的自拍视频作为补充验证集

如果你现在手头还是只有少量自拍视频，最合理的顺序是：

1. 用本项目做前处理和整理
2. 用 CASIA-B 跑通 OpenGait
3. 再把 OUMVLP 接进来做规模验证
4. 最后再把你自己的自拍视频放进验证流程里观察效果

`configs/opengait_prep_example.yaml` 和 `configs/dataset_paths.example.yaml` 是这一步的路径模板，`datasets/external/` 则预留了后续放 `CASIA-B` 和 `OUMVLP` 原始数据的位置。

## CASIA-B 实际接入流程

如果你准备开始真正对接 OpenGait，我建议第一步先做 `CASIA-B`，而不是先碰 `OUMVLP`。原因很简单：

- `CASIA-B` 的层级和概念最经典，最容易把“身份 / 序列 / 视角”这三个核心维度对齐
- 它的数据规模比 `OUMVLP` 小，更适合先验证目录、路径和预处理命令是否正确
- 你当前项目已经能把自拍视频整理成 `subject / sequence / view` 的占位结构，拿它去对照 `CASIA-B` 最容易发现差异

### 建议放哪里

建议把原始公开数据和预处理结果分开存放，不要和当前自拍视频的 `data/` 混在一起：

```text
datasets/
  external/
    casia_b/
  processed/
    CASIA-B-pkl/
```

和当前项目已有目录对应起来，建议就是：

- 原始数据：`datasets/external/casia_b/`
- pretreatment 输出：`datasets/processed/CASIA-B-pkl/`
- 自拍视频整理结果：`data/` 和 `opengait_ready/`

这三类目录不要混在一起。`data/` / `opengait_ready/` 是你自己的视频前处理结果，`datasets/external/casia_b/` 是公开数据集原始目录，`datasets/processed/CASIA-B-pkl/` 则是 OpenGait pretreatment 产物。

### 自动下载公开资源

这次已经实际完成过一轮公开资源自动化，范围只到“拉取和落盘”，还不包含训练：

- 外部 `OpenGait` 仓库已经克隆到 `external/OpenGait`
- `CASIA-B` 的 silhouette 压缩包已经下载到 `/Volumes/Data/download_cache/person_orientation_demo/GaitDatasetB-silh.zip`
- 解压后的原始数据已经落到 `datasets/external/casia_b`

本轮实际使用的就是这类命令：

```bash
git clone --depth 1 https://github.com/ShiqiYu/OpenGait.git external/OpenGait
curl -L -C - -o /Volumes/Data/download_cache/person_orientation_demo/GaitDatasetB-silh.zip http://www.cbsr.ia.ac.cn/GaitDatasetB-silh.zip
```

这份 silhouette 压缩包的真实结构不是“直接一层 PNG 目录”，而是：

```text
GaitDatasetB-silh.zip
  GaitDatasetB-silh/
    001.tar.gz
    002.tar.gz
    ...
```

所以自动落盘时需要两层展开：

1. 先解开外层 zip
2. 再把每个 subject 的 `*.tar.gz` 展开到 `datasets/external/casia_b/`

最终实际得到的目录会是：

```text
datasets/external/casia_b/
  001/
    bg-01/
      000/
        001-bg-01-000-001.png
```

如果之后公开直链失效，处理方式也很简单：

- 先不要改当前项目目录结构，仍然保持目标目录是 `datasets/external/casia_b`
- 重新从 `CASIA-B` 官方申请页、学校镜像页，或你手头可用的公开镜像拿到同一份 silhouette zip
- 把 zip 放回 `/Volumes/Data/download_cache/person_orientation_demo/GaitDatasetB-silh.zip`，再按同样目标路径重新解压

这一层的目标仍然只是把公开数据和外部代码准备好，当前项目里仍然不启动训练。

还有一个真实现象需要提前说明：这份已下载的数据在初次解压后，可能会在少数 `subject/type/view` 目录下出现空目录。遇到这种情况，`python scripts/check_casia_b_layout.py --input-dir datasets/external/casia_b` 的结果通常会是 `readiness: needs_attention`。这表示“目录里有需要人工关注的空 view”，不表示下载失败；先做空目录筛查和清理，再进入 pretreatment 会更稳妥。

### 先检查什么

在跑 OpenGait 的 pretreatment 之前，先确认目录层级和命名是否对得上：

```text
CASIA-B/
  001/
    bg-01/
      000/
        *.png
```

现在项目里已经补了目录检查脚本，可以直接先做 dry-run 检查：

```bash
python scripts/check_casia_b_layout.py --input-dir datasets/external/casia_b
```

检查时重点看这三件事：

1. `subject` 是否是三位或类似的身份编号
2. `type` 是否能对应到 OpenGait 里常见的 `bg / nm / cl` 这类序列类型
3. `view` 是否按固定角度编号组织，而不是散落的文件名

如果你的原始数据已经解压，但还没接训练工程，最重要的是先确认“目录能被脚本稳定扫到”，而不是急着看模型效果。

### CASIA-B 空目录筛查与清理

在真正进入 pretreatment 之前，建议先做一次空 `view` 目录筛查。

原因很简单：

- `check_casia_b_layout.py` 会把“存在 view 目录但没有 png 帧文件”标成 warning
- 这些 warning 会让 `readiness` 变成 `needs_attention`
- 如果直接带着这类空目录进入 pretreatment，后续更容易把问题放大

项目里现在已经补了一个保守工具：

```bash
python scripts/clean_casia_b_empty_views.py --input-dir datasets/external/casia_b
```

它只会把目录分成两类：

- `empty`：目录里完全没有任何文件和子目录
- `suspicious`：目录里没有 png，但仍然存在隐藏文件、异常文件或子目录，这类目录不会自动删除

只扫描：

```bash
python scripts/clean_casia_b_empty_views.py --input-dir datasets/external/casia_b
```

dry-run：

```bash
python scripts/clean_casia_b_empty_views.py --input-dir datasets/external/casia_b --dry-run
```

真正删除：

```bash
python scripts/clean_casia_b_empty_views.py --input-dir datasets/external/casia_b --delete-empty-views
```

这个脚本会把扫描结果写到：

- `reports/casia_b_empty_views.tsv`
- `reports/casia_b_empty_views.md`

删除后，建议立刻再跑一次目录检查：

```bash
python scripts/check_casia_b_layout.py --input-dir datasets/external/casia_b
```

如果删后 warning 消失，说明这批“空 view”问题已经被清掉；如果还存在别的 warning，就继续按 `issues` 列表逐项看。

不建议自动删除的情况：

- 目录里有隐藏文件
- 目录里有非 png 文件
- 目录里还有子目录
- 你还不确定这些异常文件是不是后续流程需要保留

遇到这些情况，宁可先保留并人工检查，也不要为了追求“全绿”直接删掉。

### pretreatment 前一键预检

在真正执行 OpenGait pretreatment 之前，建议先跑一遍统一的 `preflight` 入口，而不是手动分散执行多个脚本：

```bash
python scripts/preflight_opengait_pretreatment.py \
  --opengait-root external/OpenGait \
  --input-path datasets/external/casia_b \
  --output-path datasets/processed/CASIA-B-pkl \
  --reports-dir reports
```

先跑这一步的原因很简单：

- 它会把进入 pretreatment 前最关键的几项检查串起来，避免你漏掉某一步
- 如果只单独跑目录检查，很容易忽略 OpenGait 路径、空目录状态和最终 pretreatment 命令本身
- 先把阻塞项一次性暴露出来，会比中途失败后回头排查更省时间

这一条 `preflight` 主要检查：

- `external/OpenGait` 是否存在
- `external/OpenGait/datasets/pretreatment.py` 是否存在
- `external/OpenGait/datasets/CASIA-B/README.md` 是否存在
- `datasets/external/casia_b` 的 layout 是否为 `ready`
- 当前是否还有空 `view` 目录或 `suspicious` 目录
- 当前真实路径下的 pretreatment dry-run 命令是否能正确生成

怎么看结果：

- `preflight_status: ready`：可以进入下一步的 pretreatment dry-run
- `preflight_status: blocked`：当前还不适合进入 pretreatment，先按失败项逐条修正

运行后还会额外生成两份报告：

- `reports/pretreatment_preflight.tsv`
- `reports/pretreatment_preflight.md`

它们会把每一项检查的通过/失败状态，以及下一条建议执行的命令都写清楚。

这个 `preflight` 通过后，下一条建议执行的 dry-run 命令就是：

```bash
python scripts/run_opengait_pretreatment.py \
  --opengait-root external/OpenGait \
  --input-path datasets/external/casia_b \
  --output-path datasets/processed/CASIA-B-pkl \
  --dry-run
```

### OpenGait pretreatment 运行环境

不要直接拿系统 Python 去跑 `external/OpenGait/datasets/pretreatment.py`，原因有两个：

- 系统 Python 默认没有 `cv2`，这正是这次真执行时已经实际遇到的报错来源
- 当前仓库在外置卷 `/Volumes/Data` 下，如果把 venv 也建在这个卷里，可能会被 `._*.pth` 这类 AppleDouble sidecar 污染，导致 Python 启动阶段异常

本项目现在提供了一个最小环境脚本，建议直接用 `python3.11` 运行，并默认把 pretreatment 专用环境建到用户主目录：

```bash
python3.11 scripts/setup_opengait_pretreatment_env.py --recreate
```

默认环境路径：

```text
~/.venvs/person_orientation_demo/opengait_pretreatment
```

这个脚本只安装当前 `CASIA-B silhouette -> OpenGait pretreatment` 真正需要的最小依赖：

- `opencv-python-headless`
- `numpy`
- `tqdm`

脚本执行完成后，会直接打印：

- `python_bin`
- 实际安装的最小依赖版本
- `cv2 / numpy / tqdm` 的 import 检查结果

如果你想单独手动再验一次，可以这样：

```bash
PRETREAT_PY="$HOME/.venvs/person_orientation_demo/opengait_pretreatment/bin/python"
"$PRETREAT_PY" -c "import cv2, numpy, tqdm; print(cv2.__version__); print(numpy.__version__); print(tqdm.__version__)"
```

验证通过后，再用这个专用环境去重跑 pretreatment：

```bash
PRETREAT_PY="$HOME/.venvs/person_orientation_demo/opengait_pretreatment/bin/python"
"$PRETREAT_PY" scripts/run_opengait_pretreatment.py \
  --opengait-root external/OpenGait \
  --input-path datasets/external/casia_b \
  --output-path datasets/processed/CASIA-B-pkl \
  --python-bin "$PRETREAT_PY" \
  --execute
```

如果这一步不再报 `ModuleNotFoundError: No module named 'cv2'`，但在 OpenGait 官方脚本里又遇到：

```text
AttributeError: 'NoneType' object has no attribute 'sum'
```

优先检查输入集中是否有空的 `png` 帧文件：

```bash
find datasets/external/casia_b -type f -name '*.png' -size 0
```

这次真实跑到全量 pretreatment 时，已经实际发现下载得到的 `CASIA-B silhouette` 目录里存在少量 `0B` 的坏帧。`cv2.imread()` 对这类文件会返回 `None`。

为保证当前仓库里的 pretreatment 能跑完，本地 `external/OpenGait/datasets/pretreatment.py` 已经做了一个最小健壮性补丁：遇到不可读帧会记 warning 并跳过，不会再因为 `img is None` 在 `img.sum()` 处中断整批处理。

如果你的仓库不在外置卷，或者你确实想把环境建到别的位置，也可以显式指定：

```bash
python3 scripts/setup_opengait_pretreatment_env.py --venv-dir /your/path/opengait_pretreatment
```

### 如何生成 pretreatment 命令

本项目不把 OpenGait 代码拷进当前仓库，而是通过一个轻量封装脚本帮你生成或执行外部 OpenGait 的 pretreatment 命令。

先用 dry-run 看计划命令：

```bash
python scripts/run_opengait_pretreatment.py \
  --opengait-root /path/to/OpenGait \
  --input-path datasets/external/casia_b \
  --output-path datasets/processed/CASIA-B-pkl \
  --python-bin /path/to/pretreatment-venv/bin/python \
  --dry-run
```

这个脚本默认会生成接近 OpenGait 官方 `datasets/pretreatment.py` 当前参数风格的命令，核心就是：

```bash
python /path/to/OpenGait/datasets/pretreatment.py \
  --input_path /abs/path/to/datasets/external/casia_b \
  --output_path /abs/path/to/datasets/processed/CASIA-B-pkl \
  --dataset CASIAB \
  --n_workers 4 \
  --img_size 64 \
  --log_file /abs/path/to/datasets/processed/casia_b_pretreatment.log
```

如果你显式确认要执行，再把 `--dry-run` 改成 `--execute`，并优先配合上面的 pretreatment 专用环境一起用。

### pretreatment 跑完后预期长什么样

按照 OpenGait 官方 `datasets/README.md` 当前约定，pretreatment 之后的核心结果会接近下面这种形式：

```text
DATASET_ROOT/
  001/
    bg-01/
      000/
        000.pkl
```

也就是说，对 CASIA-B 来说，处理后最关键的是每个 `subject / type / view` 目录里会出现一个对应视角的 `pkl` 序列文件。你在这个项目里建议把它落到：

```text
datasets/processed/CASIA-B-pkl/
  001/
    bg-01/
      000/
        000.pkl
```

核心判断标准不变：

- 已经从“原始视频或图片目录”变成“可被训练工程直接读取的预处理结果”
- 目录层级里能稳定表达身份、序列和视角
- 预处理结果和原始数据分离，后续可以单独清理、重建或替换

### 本阶段为什么不训练

本阶段只做接入准备，不训练，原因是：

- 你当前最需要的是把“目录、路径、预处理、人工整理”这条链路打通
- 训练一旦开始，问题会立刻扩展到算力、超参、显存、数据完整性和评测策略，不利于先把接入做稳
- 你的自拍视频样本太少，不适合作为训练起点，更适合作为整理、筛选和小规模验证材料

### 自拍视频和 CASIA-B 的边界

这两类数据最好明确分工，不要混用：

- `CASIA-B` 负责标准训练、标准评测、流程验证
- 你的自拍视频负责前处理演示、结果检查、目录整理、后续模型的少量验证

更直白一点说：

- `CASIA-B` 是“让 OpenGait 真正跑起来”的主数据
- 你自己的自拍视频是“验证这个流程在你真实场景里有没有用”的补充数据

所以，本阶段最合理的顺序仍然是：

1. 先检查 `datasets/external/casia_b/` 是否符合目录预期
2. 再用 `run_opengait_pretreatment.py` 生成并核对 pretreatment 命令
3. 然后再在外部 OpenGait 仓库里真正跑通 pretreatment
4. 再确认 `datasets/processed/CASIA-B-pkl/` 的目录和 OpenGait 配置是否稳定
5. 最后把你自己的自拍视频作为 demo 和后续验证集使用

这样做能避免把少量自拍视频误当成完整训练集，也能让后续接 OpenGait 时少走弯路。

配套模板文件可以直接看：

- `configs/casia_b_paths.example.yaml`
- `configs/opengait_pretreatment.example.yaml`

### OpenGait 训练前最后检查与 smoke test

当前项目已经把 `CASIA-B silhouette -> OpenGait pretreatment -> CASIA-B-pkl` 这条链路打通，并做过一次训练前最后检查。

pretreatment 产物当前的真实状态是：

- `datasets/processed/CASIA-B-pkl/` 已生成
- `subject_count = 124`
- `pkl_count = 13592`
- 已抽样确认多个 `pkl` 可正常读取，形状为 `n x 64 x 64`

对应的检查摘要在：

- `reports/casiab_pkl_check.md`
- `reports/opengait_smoke_test.md`

当前本地 OpenGait 仓库下，真正的训练入口是：

- `external/OpenGait/opengait/main.py`

这次最小 smoke test 参考的官方配置基线是：

- `external/OpenGait/configs/gaitset/gaitset.yaml`

项目里新增的最小 smoke 配置是：

- `configs/opengait_casiab_smoke.yaml`

这个 smoke 配置只做了很小的收缩：

- `total_iter = 1`
- `num_workers = 0`
- 更小的 `batch_size`
- `with_test = false`
- `dataset_root` 指向当前项目内的 `datasets/processed/CASIA-B-pkl`

要复用当前已经建好的专用环境，并补齐训练 smoke 所需的最小附加依赖，可以运行：

```bash
python3.11 scripts/setup_opengait_pretreatment_env.py --with-train-smoke
```

这一步会在原有 pretreatment 专用环境基础上，补齐当前 smoke test 实际用到的额外依赖，包括：

- `torch`
- `torchvision`
- `PyYAML`
- `tensorboard`
- `einops`
- `kornia`
- `matplotlib`
- `imageio`
- `scikit-learn`

依赖可以按下面三类理解：

- 必须安装：
  `torch`、`torchvision`、`PyYAML`、`tensorboard`、`numpy`、`opencv-python-headless`、`tqdm`
- 当前这份 OpenGait 仓库直接 import 时也会实际拉起：
  `einops`、`kornia`、`matplotlib`、`imageio`、`scikit-learn`
- 可选但当前不是阻塞项：
  `xformers`

这里要特别说明一点：虽然 `GaitSet + CASIA-B` 核心路径本身不一定直接需要 `imageio` 或 `scikit-learn`，但当前 OpenGait 仓库的 `modeling/models/__init__.py` 会自动导入整包模型，BigGait 相关模块会顺带触发这些依赖。因此，对“当前官方仓库直接 import”来说，它们仍然是实际需要装的。

相反，下面这些在这台机器上不值得继续投入时间：

- CUDA 版 PyTorch
- NCCL
- 任何 NVIDIA GPU 训练栈

因为当前机器没有 NVIDIA/CUDA 条件，继续折腾也不会把官方训练入口变成稳定的正式训练环境。

当前项目里的一键 smoke 脚本是：

- `scripts/run_opengait_smoke_test.py`

真实执行命令示例：

```bash
SMOKE_PY="$HOME/.venvs/person_orientation_demo/opengait_pretreatment/bin/python"
"$SMOKE_PY" scripts/run_opengait_smoke_test.py \
  --python-bin "$SMOKE_PY" \
  --opengait-root external/OpenGait \
  --config-path configs/opengait_casiab_smoke.yaml \
  --reports-dir reports
```

这个脚本会依次做四件事：

- 检查训练入口依赖是否齐全
- 检查 `datasets/processed/CASIA-B-pkl` 的目录和抽样 `pkl`
- 用 OpenGait 自己的 `DataSet` 和 `CollateFn` 做一次最小数据加载 smoke
- 真正调用一次官方训练入口 `external/OpenGait/opengait/main.py`

如果你显式传了 `--master-port`，脚本会优先使用该端口；如果端口已被占用，会自动回退到一个空闲端口。

当前这次真实 smoke 结果是：

- `import_check = passed`
- `artifact_check = passed`
- `data_smoke = passed`
- `entry_smoke = failed`

也就是说：

- 依赖已补齐
- pretreatment 产物已确认可读
- 数据读取已打通
- 最小 batch 已成功组装
- 官方训练入口最终真实失败在分布式后端要求

对应的真实报错是：

```text
RuntimeError: Distributed package doesn't have NCCL built in
```

这说明当前这份 OpenGait 代码在 `main.py` 里默认直接走：

- `torch.distributed.init_process_group('nccl', init_method='env://')`
- CUDA / NCCL 风格的分布式训练入口

因此，这台 `macOS Apple Silicon` 机器更适合继续做：

- 数据准备
- pretreatment
- 配置核对
- smoke test

而不适合作为 OpenGait 正式训练主机。真正进入训练阶段，更建议迁移到：

- `Linux`
- `NVIDIA GPU`
- 可用 `CUDA + NCCL` 的 PyTorch 环境

### 本机 CPU/gloo 兼容探针

在官方训练入口因为 `NCCL` 受阻之后，项目里额外补了一条“低风险本机兼容探针”路径，用来继续验证：

- 数据加载是否稳定
- 模型是否可实例化
- 前向 / 反向 / optimizer step 是否能执行
- 日志 / summary / checkpoint 是否能落盘

对应脚本：

- `scripts/run_opengait_cpu_probe.py`

这条路径不是 OpenGait 官方支持的正式训练方案，它只是为了让当前这台 `macOS Apple Silicon` 机器继续向前验证一点点，而不是停在 `NCCL` 前一动不动。

真实命令示例：

```bash
SMOKE_PY="$HOME/.venvs/person_orientation_demo/opengait_pretreatment/bin/python"

"$SMOKE_PY" scripts/run_opengait_cpu_probe.py \
  --python-bin "$SMOKE_PY" \
  --opengait-root external/OpenGait \
  --config-path configs/opengait_casiab_smoke.yaml \
  --reports-dir reports \
  --mode smoke \
  --cleanup-appledouble

"$SMOKE_PY" scripts/run_opengait_cpu_probe.py \
  --python-bin "$SMOKE_PY" \
  --opengait-root external/OpenGait \
  --config-path configs/opengait_casiab_smoke.yaml \
  --reports-dir reports \
  --mode short-run \
  --total-iter 5 \
  --save-name cpu_probe_short_run \
  --cleanup-appledouble
```

当前真实结果是：

- `reports/cpu_probe_smoke.md`：`passed`
- `reports/cpu_probe_short_run.md`：`passed`

这两次运行都已经真实生成了：

- checkpoint
- log
- tensorboard summary

但要特别注意：

- 这是兼容探针，不是官方正式训练入口
- 这条路径使用了本地 monkeypatch，把 OpenGait 里硬编码的 `cuda/ddp` 包装绕开了
- 它适合验证“这套数据和模型在本机还能不能再往前走几步”
- 它不适合替代 `Linux + NVIDIA GPU + CUDA/NCCL` 上的正式训练

### 本地 pretreatment.py 补丁说明

当前项目里的 `external/OpenGait/datasets/pretreatment.py` 已经带了一个本地最小健壮性补丁，用于跳过 `CASIA-B silhouette` 里的少量坏帧。

补丁摘要和保留方式已经单独写到：

- `docs/opengait_local_patch.md`
- `patches/pretreatment_skip_bad_frames.patch`
- `reports/opengait_local_patch_summary.md`
- `reports/opengait_pretreatment_local.patch`

如果你后续重新 clone 了 `external/OpenGait`，记得把这个补丁重新应用，否则再次碰到 `0B png` 时，pretreatment 可能会在 `img.sum()` 处中断。

## Docker 构建

推荐在 Apple Silicon 上显式构建 `linux/arm64` 镜像：

```bash
docker build --platform linux/arm64 -t person-orientation-demo .
```

## Docker 运行

如果你把测试视频放在当前目录的 `samples/test.mp4`：

```bash
docker run --rm \
  --platform linux/arm64 \
  -v "$(pwd)/samples:/app/samples" \
  -v "$(pwd)/outputs:/app/outputs" \
  person-orientation-demo \
  --input samples/test.mp4 \
  --output outputs/result.mp4
```

如果你的视频在别的目录，例如 `/Users/you/Videos`：

```bash
docker run --rm \
  --platform linux/arm64 \
  -v "/Users/you/Videos:/app/input_videos" \
  -v "$(pwd)/outputs:/app/outputs" \
  person-orientation-demo \
  --input /app/input_videos/test.mp4 \
  --output outputs/result.mp4
```

输出视频会写到宿主机的 `outputs/` 目录。

## 完整清理

删除本 demo 相关容器：

```bash
docker ps -a --filter ancestor=person-orientation-demo
docker rm -f $(docker ps -aq --filter ancestor=person-orientation-demo)
```

删除镜像：

```bash
docker rmi -f person-orientation-demo
```

清理构建缓存：

```bash
docker builder prune -f
```

如果你用 `uv` 创建了本地虚拟环境，也可以直接删除：

```bash
rm -rf .venv
```

## 排错

- `yolov8n.pt` 下载失败：先确认网络可用，或者把本地权重路径传给 `--model`
- `mediapipe` 安装失败：优先确认你使用的是 `Python 3.11`，以及 Docker 构建平台是 `linux/arm64`
- 输出视频打不开：检查输入视频是否存在、输出目录是否可写、以及容器挂载路径是否正确
- 检测结果偏少：可把 `--det-conf` 从默认 `0.35` 调低到 `0.25`
- 如果当前项目目录位于外部磁盘，且本地建 `.venv` 时出现 `._*` 相关报错：这是 macOS 元数据文件干扰复制导致的，改在 `/tmp` 或 `~/venvs` 下创建虚拟环境即可
