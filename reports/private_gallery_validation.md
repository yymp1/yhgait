# 私有步态库 Demo 真实验证记录

## 验证目标

确认下面这条主链路已经真实打通：

```text
真实视频 -> track -> silhouette -> embedding -> 录入私有库
真实视频 -> track -> silhouette -> embedding -> 私有库识别
```

## 已完成的两类验证

### 1. 核心链路验证

验证产物：

- `reports/private_gallery_validation_result.json`

这轮验证使用了真实视频：

```text
/home/bb/文档/xwechat_files/wxid_srx1t4h1evm822_f5b4/msg/video/2025-10/eb19955971e2cad2909ca8f6aad25d19.mp4
```

结果是：

- 成功导出 `2` 条 track
- 成功把 `track 0001 / 0002` 录入私有库
- 成功做识别
- 对应 track 的 Top-1 分别返回 `person_001 / person_002`

这个验证说明：

- 私有库录入逻辑是真实可用的
- 私有库识别逻辑是真实可用的

但它不是跨场景泛化结论，因为录入和识别使用的是同一来源视频。

### 2. GUI 主流程验证

本轮补做了 GUI 背后函数级验证，真实跑了：

- `prepare_enrollment_video(...)`
- `enroll_selected_track(...)`
- `run_private_recognition(...)`
- `select_recognition_track(...)`

本轮落地产物目录：

- 录入：`reports/private_gallery_runs/enroll/eb19955971e2cad2909ca8f6aad25d19_20260402_161525/`
- 识别：`reports/private_gallery_runs/recognize/eb19955971e2cad2909ca8f6aad25d19_20260402_161539/`

GUI 验证结果：

- `ENROLL_MODE = private_gallery_enrollment`
- `IDENTIFY_MODE = private_gallery_recognition`
- 录入页成功返回 `2` 条 track
- 识别页成功返回 `2` 条 track
- 识别页成功返回 Top-K gallery 和结果表

当前默认私有库里已经累积了多个 demo identity，因此这次识别时：

- `track 0001` 的 Top-1 是 `demo_person_001`
- `track 0002` 的 Top-1 是 `demo_person_002`

同时，这次结果也真实触发了“建议人工复核”，原因包括：

- Top-1 与 Top-2 距离差过小
- Top-1 身份样本数仍然太少

这说明 GUI 不只是能给“像谁”，也能给出“当前不建议强认”的提示。

## 当前结论

当前私有步态库 demo 已经达到“可演示”状态：

- 能录入真实人物视频
- 能查看库内 identity 和样本
- 能上传新视频做识别
- 能给出 Top-K 和人工复核提示

但当前验证更多说明“工程链路已经通”，并不等于“真实业务精度已经可靠”。
