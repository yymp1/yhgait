#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from private_gallery_core import (  # noqa: E402
    delete_identity_from_private_gallery,
    enroll_track_to_private_gallery,
    list_identity_rows,
    list_sample_rows,
    load_private_gallery_index,
    private_gallery_identity_choices,
    rebuild_private_gallery,
)
from video_gait_core import (  # noqa: E402
    process_private_gallery_enrollment,
    process_private_gallery_recognition,
)

VIDEO_MAX_TOPK = 10
DEVICE_CHOICES = [("自动", "auto"), ("CPU", "cpu"), ("GPU", "cuda:0")]
TRACK_TABLE_COLUMNS = ["目标编号", "状态", "质量", "总帧数", "有效帧", "可用帧", "目标尺寸", "置信度"]
IDENTITY_TABLE_COLUMNS = ["人员编号", "样本数", "首次录入", "最近更新", "代表样本"]
SAMPLE_TABLE_COLUMNS = ["样本编号", "人员编号", "录入时间", "目标编号", "总帧数", "可用帧", "源视频", "备注"]
RESULT_TABLE_COLUMNS = ["排名", "人员编号", "相似度分数", "参考分值", "最近样本分数", "样本数"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动姿态识别系统界面。")
    parser.add_argument(
        "--cfg-path",
        default=str(ROOT / "configs" / "opengait_casiab_formal_conservative_eval.yaml"),
        help="真实视频推理时使用的 OpenGait 配置。",
    )
    parser.add_argument("--checkpoint-iter", type=int, default=1000, help="录入和识别使用的 checkpoint iter。")
    parser.add_argument(
        "--private-gallery-root",
        default=str(ROOT / "data" / "private_gallery"),
        help="私有步态库根目录。",
    )
    parser.add_argument(
        "--video-run-root",
        default=str(ROOT / "reports" / "private_gallery_runs"),
        help="录入/识别过程的中间产物目录。",
    )
    parser.add_argument(
        "--dataset-summary-path",
        default=str(ROOT / "reports" / "effect_analysis_summary.md"),
        help="效果分析页展示的说明文档路径。",
    )
    parser.add_argument("--server-name", default="0.0.0.0", help="Gradio 监听地址。")
    parser.add_argument("--server-port", type=int, default=7860, help="Gradio 端口。")
    parser.add_argument("--topk-default", type=int, default=5, help="默认 Top-K。")
    parser.add_argument("--share", action="store_true", help="是否开启 Gradio share。")
    return parser.parse_args()


def safe_df(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=columns)


def load_markdown_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def resolve_video_input(uploaded_video: str | None, local_video_path: str | None) -> Path:
    if uploaded_video:
        return Path(uploaded_video).resolve()
    if local_video_path and local_video_path.strip():
        return Path(local_video_path.strip()).expanduser().resolve()
    raise gr.Error("请上传视频，或输入服务器上的视频路径。")


def humanize_track_status(status: str) -> str:
    mapping = {
        "ok": "可用",
        "skipped": "未进入处理",
        "too_short": "可用帧不足",
        "no_valid_silhouette": "轮廓提取失败",
        "no_frames": "无可用画面",
    }
    return mapping.get(str(status), str(status) or "-")


def humanize_quality_note(note: str) -> str:
    text = str(note or "")
    mapping = {
        "ok": "符合要求",
        "export_filtered": "未进入后续处理",
        "no_crop_frames": "未提取到有效画面",
        "segmentation_failed": "轮廓提取失败",
    }
    if text.startswith("valid_run_too_short<"):
        return f"连续可用帧不足（至少需要 {text.split('<', 1)[-1]} 帧）"
    return mapping.get(text, text or "-")


def humanize_review_reason(reason: str) -> str:
    mapping = {
        "当前私有步态库为空，请先录入至少一个 identity。": "当前样本库为空，请先录入至少一位人员。",
        "当前私有库身份数不足 2 个。": "当前样本库中的人员较少，识别结果仅供参考。",
        "Top-1 与 Top-2 的距离差小于 0.03。": "当前前两名结果接近，建议人工复核。",
        "Top-1 身份当前样本数少于 2。": "当前样本较少，识别结果仅供参考。",
        "Top-1 距离偏大。": "当前相似度分数偏弱，建议结合人工判断。",
    }
    return mapping.get(str(reason), str(reason))


def select_best_track_id(session: dict[str, Any]) -> str | None:
    track_id = session.get("selected_track_id")
    if track_id:
        return str(track_id)
    for track in session.get("tracks", []):
        if track.get("status") == "ok":
            return str(track.get("track_id"))
    if session.get("tracks"):
        return str(session["tracks"][0].get("track_id"))
    return None


def build_track_table(session: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for track in session.get("tracks", []):
        rows.append(
            {
                "目标编号": track.get("track_id", ""),
                "状态": humanize_track_status(track.get("status", "")),
                "质量": humanize_quality_note(track.get("quality_note", "")),
                "总帧数": int(track.get("frame_count", 0)),
                "有效帧": int(track.get("valid_frames", 0)),
                "可用帧": int(track.get("used_frames", 0)),
                "目标尺寸": round(float(track.get("avg_bbox_area", 0.0)), 1),
                "置信度": round(float(track.get("avg_confidence", 0.0)), 4),
            }
        )
    return safe_df(rows, TRACK_TABLE_COLUMNS)


def track_preview_markdown(track: dict[str, Any], title: str) -> str:
    return "\n".join(
        [
            f"### {title}",
            f"- 目标编号：`{track.get('track_id', '')}`",
            f"- 状态：`{humanize_track_status(track.get('status', ''))}`",
            f"- 质量说明：`{humanize_quality_note(track.get('quality_note', ''))}`",
            f"- 总帧数：`{track.get('frame_count', 0)}`",
            f"- 有效帧数：`{track.get('valid_frames', 0)}`",
            f"- 可用帧数：`{track.get('used_frames', 0)}`",
            f"- 目标片段：`{'已生成' if track.get('clip_path') else '未生成'}`",
            f"- 轮廓摘要：`{'已生成' if track.get('silhouette_strip') else '未生成'}`",
        ]
    )


def track_quality_markdown(track: dict[str, Any]) -> str:
    if track.get("status") == "ok":
        return "当前目标质量合格，可用于保存样本或进行识别。"
    return "\n".join(
        [
            "当前目标暂不建议用于保存或识别。",
            f"原因：`{humanize_quality_note(track.get('quality_note', ''))}`",
            "常见原因包括：目标过小、轮廓提取不稳定、连续可用帧不足，或视频本身不适合识别。",
        ]
    )


def private_gallery_overview_markdown(gallery_root: Path) -> str:
    index = load_private_gallery_index(gallery_root)
    identity_count = len(index.get("identities", {}))
    sample_count = len(index.get("samples", {}))
    return "\n".join(
        [
            "## 样本库状态",
            f"- 样本库位置：`{gallery_root}`",
            f"- 已录入人员数：`{identity_count}`",
            f"- 样本总数：`{sample_count}`",
            "- 先录入人员样本，再用新视频进行识别比对。",
            "- 当前识别结果仅与本地样本库进行匹配。",
        ]
    )


def private_identity_df(gallery_root: Path) -> pd.DataFrame:
    rows = list_identity_rows(gallery_root)
    mapped_rows = [
        {
            "人员编号": row.get("identity", ""),
            "样本数": row.get("sample_count", 0),
            "首次录入": row.get("created_at", ""),
            "最近更新": row.get("updated_at", ""),
            "代表样本": row.get("representative_sample_id", ""),
        }
        for row in rows
    ]
    return safe_df(mapped_rows, IDENTITY_TABLE_COLUMNS)


def private_sample_df(gallery_root: Path, identity: str | None = None) -> pd.DataFrame:
    rows = list_sample_rows(gallery_root, identity)
    mapped_rows = [
        {
            "样本编号": row.get("sample_id", ""),
            "人员编号": row.get("identity", ""),
            "录入时间": row.get("created_at", ""),
            "目标编号": row.get("track_id", ""),
            "总帧数": row.get("frame_count", 0),
            "可用帧": row.get("used_frames", 0),
            "源视频": row.get("input_video", ""),
            "备注": row.get("note", ""),
        }
        for row in rows
    ]
    return safe_df(mapped_rows, SAMPLE_TABLE_COLUMNS)


def identity_dropdown_update(gallery_root: Path, current: str | None = None):
    identities = private_gallery_identity_choices(gallery_root)
    value = current if current in identities else (identities[0] if identities else None)
    return gr.update(choices=identities, value=value)


def enrollment_session_markdown(session: dict[str, Any], gallery_root: Path) -> str:
    selected_track = select_best_track_id(session) or "无"
    ok_tracks = sum(1 for track in session.get("tracks", []) if track.get("status") == "ok")
    return "\n".join(
        [
            "## 样本处理结果",
            f"- 源视频：`{session.get('input_video', '')}`",
            f"- 处理预览：`{session.get('annotated_video', '')}`",
            f"- 可用目标数 / 检测目标数：`{ok_tracks} / {len(session.get('tracks', []))}`",
            f"- 当前推荐目标：`{selected_track}`",
            f"- 样本库位置：`{gallery_root}`",
        ]
    )


def select_enrollment_track(session: dict[str, Any], track_id: str):
    if not session or not session.get("tracks"):
        raise gr.Error("请先处理一段样本视频。")
    track = next((item for item in session["tracks"] if item.get("track_id") == track_id), None)
    if track is None:
        raise gr.Error(f"找不到目标：{track_id}")
    return (
        track_preview_markdown(track, "待保存目标"),
        track.get("clip_path", "") or None,
        track.get("silhouette_strip", "") or None,
        track_quality_markdown(track),
    )


def prepare_enrollment_video(
    uploaded_video: str | None,
    local_video_path: str | None,
    detection_device: str,
    gallery_root: Path,
    cfg_path: Path,
    checkpoint_iter: int,
    run_root: Path,
):
    source_video = resolve_video_input(uploaded_video, local_video_path)
    session = process_private_gallery_enrollment(
        video_path=source_video,
        cfg_path=cfg_path,
        checkpoint_iter=checkpoint_iter,
        run_root=run_root / "enroll",
        detection_device=None if detection_device == "auto" else detection_device,
    )
    selected_track = select_best_track_id(session)
    track_update = gr.update(
        choices=[track.get("track_id", "") for track in session.get("tracks", [])],
        value=selected_track,
    )
    if selected_track:
        track_info, track_clip, track_image, track_status = select_enrollment_track(session, selected_track)
    else:
        track_info, track_clip, track_image, track_status = ("没有可用目标。", None, None, "当前没有通过质量筛选的目标。")
    return (
        session,
        track_update,
        build_track_table(session),
        session.get("annotated_video"),
        enrollment_session_markdown(session, gallery_root),
        track_info,
        track_clip,
        track_image,
        track_status,
    )


def enroll_selected_track(
    session: dict[str, Any],
    track_id: str,
    identity: str,
    note: str,
    gallery_root: Path,
    selected_identity: str | None,
):
    if not session:
        raise gr.Error("请先处理样本视频。")
    if not track_id:
        raise gr.Error("请先选择一个目标。")
    sample_meta = enroll_track_to_private_gallery(session, track_id, identity, gallery_root, note=note)
    dropdown = identity_dropdown_update(gallery_root, sample_meta["identity"])
    return (
        "\n".join(
            [
                "## 样本保存成功",
                f"- 人员编号：`{sample_meta['identity']}`",
                f"- 样本编号：`{sample_meta['sample_id']}`",
                f"- 目标编号：`{sample_meta['track_id']}`",
                "- 当前目标已保存到样本库，并已更新该人员的样本特征。",
            ]
        ),
        private_gallery_overview_markdown(gallery_root),
        private_identity_df(gallery_root),
        dropdown,
        private_sample_df(gallery_root, sample_meta["identity"] or selected_identity),
    )


def refresh_private_gallery_views(gallery_root: Path, selected_identity: str | None):
    dropdown = identity_dropdown_update(gallery_root, selected_identity)
    current_identity = dropdown["value"] if isinstance(dropdown, dict) else selected_identity
    return (
        private_gallery_overview_markdown(gallery_root),
        private_identity_df(gallery_root),
        dropdown,
        private_sample_df(gallery_root, current_identity),
    )


def delete_identity_action(gallery_root: Path, identity: str | None):
    if not identity:
        raise gr.Error("请先选择一个人员编号。")
    delete_identity_from_private_gallery(gallery_root, identity)
    dropdown = identity_dropdown_update(gallery_root, None)
    current_identity = dropdown["value"] if isinstance(dropdown, dict) else None
    return (
        f"已删除人员 `{identity}`。",
        private_gallery_overview_markdown(gallery_root),
        private_identity_df(gallery_root),
        dropdown,
        private_sample_df(gallery_root, current_identity),
    )


def rebuild_gallery_action(gallery_root: Path, selected_identity: str | None):
    rebuild_private_gallery(gallery_root)
    dropdown = identity_dropdown_update(gallery_root, selected_identity)
    current_identity = dropdown["value"] if isinstance(dropdown, dict) else selected_identity
    return (
        "已重新整理样本特征。",
        private_gallery_overview_markdown(gallery_root),
        private_identity_df(gallery_root),
        dropdown,
        private_sample_df(gallery_root, current_identity),
    )


def private_result_gallery_items(rows: list[dict[str, Any]]) -> list[tuple]:
    items = []
    for row in rows:
        preview_path = row.get("preview_path", "")
        if not preview_path:
            continue
        caption = (
            f"第 {row['rank']} 名 | {row['identity']} | "
            f"相似度分数 {row['distance']:.4f} | 参考分值 {row['score']:.4f} | "
            f"样本数 {row['sample_count']}"
        )
        items.append((preview_path, caption))
    return items


def private_result_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "排名": row["rank"],
                "人员编号": row["identity"],
                "相似度分数": round(row["distance"], 4),
                "参考分值": round(row["score"], 4),
                "最近样本分数": round(row["best_sample_distance"], 4),
                "样本数": int(row["sample_count"]),
            }
            for row in rows
        ],
        columns=RESULT_TABLE_COLUMNS,
    )


def recognition_session_markdown(session: dict[str, Any], gallery_root: Path) -> str:
    selected_track = select_best_track_id(session) or "无"
    ok_tracks = sum(1 for track in session.get("tracks", []) if track.get("status") == "ok")
    return "\n".join(
        [
            "## 识别处理结果",
            f"- 源视频：`{session.get('input_video', '')}`",
            f"- 处理预览：`{session.get('annotated_video', '')}`",
            f"- 可识别目标数 / 检测目标数：`{ok_tracks} / {len(session.get('tracks', []))}`",
            f"- 当前推荐目标：`{selected_track}`",
            f"- 样本库位置：`{gallery_root}`",
        ]
    )


def recognition_status_markdown(track: dict[str, Any], topk: int) -> str:
    if track.get("status") != "ok":
        return track_quality_markdown(track)
    rows = track.get("rows", [])
    if not rows:
        return "当前目标已处理完成，但样本库中暂无可比对人员。"
    review_line = "当前前两名结果接近，建议人工复核。" if track.get("review_required") else "当前识别结果相对稳定，可作为初步参考。"
    lines = [
        f"最相近人员：`{track.get('top1_identity', '')}`",
        f"已返回前 {topk} 个识别结果。",
        f"相似度分数：`{track.get('top1_distance', 0.0):.4f}`",
        f"结果提示：`{review_line}`",
        "- 相似度分数用于展示接近程度，数值越小表示越接近。",
    ]
    for reason in track.get("review_reasons", []):
        lines.append(f"- 提示：{humanize_review_reason(reason)}")
    return "\n".join(lines)


def select_recognition_track(session: dict[str, Any], track_id: str, topk: int):
    if not session or not session.get("tracks"):
        raise gr.Error("请先处理一段识别视频。")
    track = next((item for item in session["tracks"] if item.get("track_id") == track_id), None)
    if track is None:
        raise gr.Error(f"找不到目标：{track_id}")
    rows = track.get("rows", [])[:topk]
    return (
        track_preview_markdown(track, "待识别目标"),
        track.get("clip_path", "") or None,
        track.get("silhouette_strip", "") or None,
        private_result_gallery_items(rows),
        private_result_table(rows) if rows else safe_df([], RESULT_TABLE_COLUMNS),
        recognition_status_markdown(track, topk),
    )


def run_private_recognition(
    uploaded_video: str | None,
    local_video_path: str | None,
    topk: int,
    detection_device: str,
    gallery_root: Path,
    cfg_path: Path,
    checkpoint_iter: int,
    run_root: Path,
):
    if not private_gallery_identity_choices(gallery_root):
        raise gr.Error("当前样本库为空，请先录入至少一位人员。")
    source_video = resolve_video_input(uploaded_video, local_video_path)
    session = process_private_gallery_recognition(
        video_path=source_video,
        gallery_root=gallery_root,
        cfg_path=cfg_path,
        checkpoint_iter=checkpoint_iter,
        run_root=run_root / "recognize",
        topk=VIDEO_MAX_TOPK,
        detection_device=None if detection_device == "auto" else detection_device,
    )
    selected_track = select_best_track_id(session)
    track_update = gr.update(
        choices=[track.get("track_id", "") for track in session.get("tracks", [])],
        value=selected_track,
    )
    if selected_track:
        track_info, track_clip, track_image, gallery_items, result_table, status_md = select_recognition_track(session, selected_track, topk)
    else:
        track_info, track_clip, track_image, gallery_items, result_table, status_md = (
            "没有可用目标。",
            None,
            None,
            [],
            safe_df([], RESULT_TABLE_COLUMNS),
            "当前未检测到适合识别的目标，请更换视频后重试。",
        )
    return (
        session,
        track_update,
        build_track_table(session),
        session.get("annotated_video"),
        recognition_session_markdown(session, gallery_root),
        track_info,
        track_clip,
        track_image,
        gallery_items,
        result_table,
        status_md,
    )


def build_dataset_aux_tab(dataset_summary_path: Path) -> None:
    with gr.TabItem("效果分析"):
        summary_md = load_markdown_if_exists(dataset_summary_path)
        if summary_md:
            gr.Markdown(
                "\n".join(
                    [
                        "## 结果说明",
                        "- 这里用于查看当前模型在标准样本上的识别表现，仅作效果参考。",
                        "- 这里的结果主要用于对比不同训练阶段的效果差异。",
                        "- 这里展示的是固定样本下的识别表现，不代表真实业务场景效果。",
                    ]
                )
            )
            gr.Markdown(summary_md)
        else:
            gr.Markdown(
                "\n".join(
                    [
                        "## 效果分析暂未加载",
                        "- 当前系统默认使用“样本录入”和“视频识别”页面。",
                        "- 如分析结果未准备完成，这里将保持简要说明。",
                    ]
                )
            )


def build_demo(
    cfg_path: Path,
    checkpoint_iter: int,
    video_run_root: Path,
    private_gallery_root: Path,
    dataset_summary_path: Path,
    topk_default: int,
) -> gr.Blocks:
    initial_identity_choices = private_gallery_identity_choices(private_gallery_root)
    initial_identity = initial_identity_choices[0] if initial_identity_choices else None

    with gr.Blocks(title="姿态识别系统") as demo:
        gr.Markdown(
            "\n".join(
                [
                    "# 姿态识别系统",
                    "- 先录入人员样本，再上传新视频完成识别比对。",
                    "- 默认使用“样本录入”和“视频识别”两个页面。",
                    "- 当前结果来自本地样本库，仅用于演示识别流程。",
                ]
            )
        )

        with gr.Tabs():
            with gr.TabItem("样本录入"):
                enroll_state = gr.State({})

                gr.Markdown("上传一段清晰的行走视频，系统会自动提取可用目标，并保存到样本库。")

                with gr.Row():
                    with gr.Column(scale=1):
                        enroll_uploaded_video = gr.File(label="上传样本视频", file_types=["video"], type="filepath")
                        enroll_local_video_path = gr.Textbox(label="或输入服务器上的视频路径", placeholder="/abs/path/to/video.mp4")
                        enroll_identity = gr.Textbox(label="人员编号", placeholder="例如：001、张三、访客A")
                        enroll_note = gr.Textbox(label="备注信息（可选）", placeholder="例如：医院内走廊 / 背面行走 / 上午采集")
                        enroll_detection_device = gr.Dropdown(choices=DEVICE_CHOICES, value="auto", label="处理方式")
                        enroll_prepare_btn = gr.Button("开始处理样本视频", variant="primary")
                        enroll_commit_btn = gr.Button("保存当前目标为样本")
                    with gr.Column(scale=1):
                        private_gallery_overview = gr.Markdown(private_gallery_overview_markdown(private_gallery_root))
                        enroll_session_md = gr.Markdown("请先处理一段样本视频。")
                        enroll_track_dropdown = gr.Dropdown(choices=[], label="选择要保存的目标")
                        enroll_track_table = gr.Dataframe(headers=TRACK_TABLE_COLUMNS, interactive=False, wrap=True, label="候选目标列表")
                        enroll_load_track_btn = gr.Button("查看当前目标")
                        enroll_action_status = gr.Markdown("")

                enroll_annotated_video = gr.Video(label="样本处理预览", interactive=False)
                with gr.Row():
                    with gr.Column(scale=1):
                        enroll_track_info = gr.Markdown()
                        enroll_track_clip = gr.Video(label="当前目标片段", interactive=False)
                        enroll_track_image = gr.Image(label="目标轮廓摘要", type="filepath", interactive=False)
                        enroll_track_status = gr.Markdown()
                    with gr.Column(scale=1):
                        manage_identity_dropdown = gr.Dropdown(choices=initial_identity_choices, value=initial_identity, label="选择人员")
                        gallery_identity_table = gr.Dataframe(value=private_identity_df(private_gallery_root), interactive=False, wrap=True, label="已录入人员列表")
                        gallery_sample_table = gr.Dataframe(value=private_sample_df(private_gallery_root, initial_identity), interactive=False, wrap=True, label="当前人员样本列表")
                        with gr.Row():
                            gallery_refresh_btn = gr.Button("刷新样本库")
                            gallery_rebuild_btn = gr.Button("重新整理样本特征")
                        gallery_delete_btn = gr.Button("删除当前人员", variant="stop")

                enroll_prepare_btn.click(
                    fn=lambda uploaded, local_path, det_dev: prepare_enrollment_video(
                        uploaded,
                        local_path,
                        det_dev,
                        private_gallery_root,
                        cfg_path,
                        checkpoint_iter,
                        video_run_root,
                    ),
                    inputs=[enroll_uploaded_video, enroll_local_video_path, enroll_detection_device],
                    outputs=[enroll_state, enroll_track_dropdown, enroll_track_table, enroll_annotated_video, enroll_session_md, enroll_track_info, enroll_track_clip, enroll_track_image, enroll_track_status],
                )
                enroll_load_track_btn.click(
                    fn=select_enrollment_track,
                    inputs=[enroll_state, enroll_track_dropdown],
                    outputs=[enroll_track_info, enroll_track_clip, enroll_track_image, enroll_track_status],
                )
                enroll_commit_btn.click(
                    fn=lambda session, track_id, identity, note, selected_identity: enroll_selected_track(
                        session,
                        track_id,
                        identity,
                        note,
                        private_gallery_root,
                        selected_identity,
                    ),
                    inputs=[enroll_state, enroll_track_dropdown, enroll_identity, enroll_note, manage_identity_dropdown],
                    outputs=[enroll_action_status, private_gallery_overview, gallery_identity_table, manage_identity_dropdown, gallery_sample_table],
                )
                gallery_refresh_btn.click(
                    fn=lambda selected_identity: refresh_private_gallery_views(private_gallery_root, selected_identity),
                    inputs=[manage_identity_dropdown],
                    outputs=[private_gallery_overview, gallery_identity_table, manage_identity_dropdown, gallery_sample_table],
                )
                gallery_rebuild_btn.click(
                    fn=lambda selected_identity: rebuild_gallery_action(private_gallery_root, selected_identity),
                    inputs=[manage_identity_dropdown],
                    outputs=[enroll_action_status, private_gallery_overview, gallery_identity_table, manage_identity_dropdown, gallery_sample_table],
                )
                gallery_delete_btn.click(
                    fn=lambda identity: delete_identity_action(private_gallery_root, identity),
                    inputs=[manage_identity_dropdown],
                    outputs=[enroll_action_status, private_gallery_overview, gallery_identity_table, manage_identity_dropdown, gallery_sample_table],
                )
                manage_identity_dropdown.change(
                    fn=lambda identity: private_sample_df(private_gallery_root, identity),
                    inputs=[manage_identity_dropdown],
                    outputs=[gallery_sample_table],
                )

            with gr.TabItem("视频识别"):
                recognize_state = gr.State({})
                gr.Markdown("上传视频后，系统会自动提取目标片段，并与样本库中的人员进行比对。")
                with gr.Row():
                    with gr.Column(scale=1):
                        recog_uploaded_video = gr.File(label="上传识别视频", file_types=["video"], type="filepath")
                        recog_local_video_path = gr.Textbox(label="或输入服务器上的视频路径", placeholder="/abs/path/to/video.mp4")
                        recog_topk = gr.Slider(minimum=1, maximum=VIDEO_MAX_TOPK, value=min(topk_default, VIDEO_MAX_TOPK), step=1, label="返回结果数量（Top-K）")
                        recog_detection_device = gr.Dropdown(choices=DEVICE_CHOICES, value="auto", label="处理方式")
                        recog_process_btn = gr.Button("开始识别", variant="primary")
                    with gr.Column(scale=1):
                        recog_session_md = gr.Markdown(private_gallery_overview_markdown(private_gallery_root))
                        recog_track_dropdown = gr.Dropdown(choices=[], label="选择要识别的目标")
                        recog_track_table = gr.Dataframe(headers=TRACK_TABLE_COLUMNS, interactive=False, wrap=True, label="候选目标列表")
                        recog_load_track_btn = gr.Button("查看当前目标结果")

                recog_annotated_video = gr.Video(label="识别处理预览", interactive=False)
                with gr.Row():
                    with gr.Column(scale=1):
                        recog_track_info = gr.Markdown()
                        recog_track_clip = gr.Video(label="当前目标片段", interactive=False)
                        recog_track_image = gr.Image(label="目标轮廓摘要", type="filepath", interactive=False)
                        recog_status = gr.Markdown()
                    with gr.Column(scale=2):
                        recog_gallery = gr.Gallery(label="相似结果", columns=2, object_fit="contain", height="auto")
                        recog_table = gr.Dataframe(headers=RESULT_TABLE_COLUMNS, interactive=False, wrap=True, label="识别结果列表")

                recog_process_btn.click(
                    fn=lambda uploaded, local_path, k, det_dev: run_private_recognition(
                        uploaded,
                        local_path,
                        int(k),
                        det_dev,
                        private_gallery_root,
                        cfg_path,
                        checkpoint_iter,
                        video_run_root,
                    ),
                    inputs=[recog_uploaded_video, recog_local_video_path, recog_topk, recog_detection_device],
                    outputs=[recognize_state, recog_track_dropdown, recog_track_table, recog_annotated_video, recog_session_md, recog_track_info, recog_track_clip, recog_track_image, recog_gallery, recog_table, recog_status],
                )
                recog_load_track_btn.click(
                    fn=lambda session, track_id, k: select_recognition_track(session, track_id, int(k)),
                    inputs=[recognize_state, recog_track_dropdown, recog_topk],
                    outputs=[recog_track_info, recog_track_clip, recog_track_image, recog_gallery, recog_table, recog_status],
                )

            build_dataset_aux_tab(dataset_summary_path)

    return demo


def main() -> int:
    args = parse_args()
    demo = build_demo(
        cfg_path=Path(args.cfg_path).resolve(),
        checkpoint_iter=args.checkpoint_iter,
        video_run_root=Path(args.video_run_root).resolve(),
        private_gallery_root=Path(args.private_gallery_root).resolve(),
        dataset_summary_path=Path(args.dataset_summary_path).resolve(),
        topk_default=args.topk_default,
    )
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
