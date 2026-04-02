#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gait_demo_core import (  # noqa: E402
    PROBE_TYPES,
    build_probe_key,
    case_map,
    comparison_summary,
    list_gallery_views,
    make_sequence_strip,
    load_cache,
)
from video_gait_core import process_real_video_demo  # noqa: E402

VIDEO_MAX_TOPK = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动步态检索闭环分析与真实视频 demo。")
    parser.add_argument(
        "--primary-cache-path",
        default=str(ROOT / "reports" / "gait_demo_cache_formal_conservative_real_iter01000.npz"),
        help="主 checkpoint 的 embedding cache 路径。",
    )
    parser.add_argument(
        "--compare-cache-path",
        default=str(ROOT / "reports" / "gait_demo_cache_formal_conservative_real_iter00500.npz"),
        help="对照 checkpoint 的 embedding cache 路径。",
    )
    parser.add_argument(
        "--primary-analysis-path",
        default=str(ROOT / "reports" / "gait_demo_analysis_iter01000.json"),
        help="主 checkpoint 的分析 JSON 路径。",
    )
    parser.add_argument(
        "--compare-analysis-path",
        default=str(ROOT / "reports" / "gait_demo_analysis_iter00500.json"),
        help="对照 checkpoint 的分析 JSON 路径。",
    )
    parser.add_argument(
        "--cfg-path",
        default=str(ROOT / "configs" / "opengait_casiab_formal_conservative_eval.yaml"),
        help="构建 cache 与真实视频 probe embedding 时使用的评估配置。",
    )
    parser.add_argument("--primary-checkpoint-iter", type=int, default=1000, help="主 checkpoint iter。")
    parser.add_argument("--compare-checkpoint-iter", type=int, default=500, help="对照 checkpoint iter。")
    parser.add_argument(
        "--run-root",
        default=str(ROOT / "reports" / "real_video_demo_runs"),
        help="真实视频 demo 的输出根目录。",
    )
    parser.add_argument("--server-name", default="0.0.0.0", help="Gradio 监听地址。")
    parser.add_argument("--server-port", type=int, default=7860, help="Gradio 端口。")
    parser.add_argument("--topk-default", type=int, default=5, help="默认 Top-K。")
    parser.add_argument("--share", action="store_true", help="是否开启 Gradio share。")
    return parser.parse_args()


def ensure_cache(cache_path: Path, cfg_path: Path, checkpoint_iter: int, master_port: int) -> None:
    if cache_path.exists():
        return
    cmd = [
        str(ROOT / ".venvs" / "opengait-gpu" / "bin" / "python"),
        str(ROOT / "scripts" / "build_gait_gallery.py"),
        "--cfg-path",
        str(cfg_path),
        "--checkpoint-iter",
        str(checkpoint_iter),
        "--cache-path",
        str(cache_path),
        "--master-port",
        str(master_port),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)


def ensure_analysis(analysis_path: Path, cache_path: Path, topk: int) -> None:
    if analysis_path.exists():
        return
    cmd = [
        str(ROOT / ".venvs" / "opengait-gpu" / "bin" / "python"),
        str(ROOT / "scripts" / "build_gait_retrieval_analysis.py"),
        "--cache-path",
        str(cache_path),
        "--analysis-path",
        str(analysis_path),
        "--topk",
        str(topk),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)


def load_analysis(analysis_path: str | Path) -> dict[str, Any]:
    with Path(analysis_path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def percentage(rate: float) -> str:
    return f"{rate * 100:.2f}%"


def analysis_label(analysis: dict[str, Any]) -> str:
    restore_hint = analysis.get("restore_hint", "?")
    checkpoint_name = Path(analysis["checkpoint_path"]).name
    return f"iter {restore_hint} ({checkpoint_name})"


def summary_rows(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    row = {
        "checkpoint": analysis_label(analysis),
        "probe_count": int(analysis["probe_count"]),
        "Top-1": percentage(analysis["overall"]["top1_rate"]),
        "Top-5": percentage(analysis["overall"]["top5_rate"]),
    }
    for group_name in ("NM", "BG", "CL"):
        stats = analysis["groups"][group_name]
        row[f"{group_name} Top-1"] = percentage(stats["top1_rate"])
        row[f"{group_name} Top-5"] = percentage(stats["top5_rate"])
    return [row]


def make_summary_table(primary_analysis: dict[str, Any], compare_analysis: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(summary_rows(compare_analysis) + summary_rows(primary_analysis))


def fixed_definition_markdown(primary_analysis: dict[str, Any], compare_analysis: dict[str, Any]) -> str:
    probe_count = int(primary_analysis["probe_count"])
    primary_top1 = percentage(primary_analysis["overall"]["top1_rate"])
    primary_top5 = percentage(primary_analysis["overall"]["top5_rate"])
    compare_top1 = percentage(compare_analysis["overall"]["top1_rate"])
    compare_top5 = percentage(compare_analysis["overall"]["top5_rate"])
    return "\n".join(
        [
            "## “80%” 到底是什么意思",
            f"- 这页统一按 **固定测试集 probe** 统计，不是随机抽样印象值。样本量是 `{probe_count}`。",
            "- 固定测试集包含 `NM / BG / CL` 三组，具体 probe 类型是 `nm-05/06`、`bg-01/02`、`cl-01/02`。",
            "- gallery 协议固定为同视角 `nm-01~04`，这和我们当前检索 GUI/对照评估使用的是同一套协议。",
            f"- 当前 `01000` 的固定测试集 Top-5 = `{primary_top5}`，Top-1 = `{primary_top1}`。",
            f"- 对照组 `00500` 的固定测试集 Top-5 = `{compare_top5}`，Top-1 = `{compare_top1}`。",
            "- 这些数值是 GUI 检索协议下的命中率，不等于官方 `NM/BG/CL Rank-1` 评测值。",
            "- 所以这里展示的是 **数据集检索效果**，不是现实业务视频系统效果。",
        ]
    )


def case_choices(analysis: dict[str, Any], mode: str, limit: int = 80) -> list[str]:
    keys = analysis["case_lists"]["success_top1"] if mode == "success" else analysis["case_lists"]["failure_top1"]
    return keys[:limit]


def improved_case_choices(compare_pairs: dict[str, Any], limit: int = 80) -> list[str]:
    improved_keys = [
        pair["probe_key"]
        for pair in compare_pairs["pairs"]
        if (not pair["baseline_top1_hit"]) and pair["target_top1_hit"]
    ]
    return improved_keys[:limit]


def list_probe_labels(primary_analysis: dict[str, Any]) -> list[str]:
    labels = {case["probe"]["label"] for case in primary_analysis["cases"]}
    return sorted(labels)


def list_probe_views(primary_analysis: dict[str, Any]) -> list[str]:
    views = {case["probe"]["view"] for case in primary_analysis["cases"]}
    return sorted(views)


def hit_text(value: Any) -> str:
    if value is None:
        return "n/a"
    return "hit" if bool(value) else "miss"


def make_gallery_items(rows: list[dict[str, Any]]) -> list[tuple]:
    items = []
    for row in rows:
        caption = (
            f"Rank {row['rank']} | {row['subject_id']} | {row['sequence_type']} | {row['view']} | "
            f"dist={row['distance']:.4f} | {hit_text(row.get('hit'))}"
        )
        items.append((make_sequence_strip(row["pkl_path"]), caption))
    return items


def make_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rank": row["rank"],
                "subject_id": row["subject_id"],
                "sequence_type": row["sequence_type"],
                "view": row["view"],
                "distance": round(row["distance"], 4),
                "hit": hit_text(row.get("hit")),
            }
            for row in rows
        ]
    )


def probe_markdown(case: dict[str, Any], analysis: dict[str, Any]) -> str:
    probe = case["probe"]
    return "\n".join(
        [
            f"**Probe**: `{probe['label']} / {probe['type']} / {probe['view']}`",
            f"**Probe Group**: `{case['probe_group']}`",
            f"**Checkpoint**: `{Path(analysis['checkpoint_path']).name}`",
            f"**数据集路径**: `{analysis['dataset_root']}`",
            f"**协议**: `{analysis['protocol']}`",
        ]
    )


def checkpoint_status_markdown(case: dict[str, Any], analysis: dict[str, Any], topk: int) -> str:
    checkpoint_name = Path(analysis["checkpoint_path"]).name
    rows = case["rows"][:topk]
    return "\n".join(
        [
            f"**{checkpoint_name}**",
            f"- Top-1: `{'hit' if case['top1_hit'] else 'miss'}`",
            f"- Top-{topk}: `{'hit' if any(row['hit'] for row in rows) else 'miss'}`",
            f"- gallery 数量: `{case['gallery_count']}`",
            f"- Top-1 候选: `{case['top1_subject']}`",
            f"- Top-1 distance: `{case['top1_distance']:.4f}`" if case["top1_distance"] is not None else "- Top-1 distance: `N/A`",
        ]
    )


def compare_markdown(
    baseline_case: dict[str, Any],
    target_case: dict[str, Any],
    baseline_analysis: dict[str, Any],
    target_analysis: dict[str, Any],
    topk: int,
) -> str:
    baseline_name = Path(baseline_analysis["checkpoint_path"]).name
    target_name = Path(target_analysis["checkpoint_path"]).name
    return "\n".join(
        [
            "### 同一 Probe 的 checkpoint 对比",
            f"- `{baseline_name}`: Top-1 = `{'hit' if baseline_case['top1_hit'] else 'miss'}`，Top-{topk} = `{'hit' if any(row['hit'] for row in baseline_case['rows'][:topk]) else 'miss'}`，Top-1 distance = `{baseline_case['top1_distance']:.4f}`" if baseline_case["top1_distance"] is not None else f"- `{baseline_name}`: 无结果",
            f"- `{target_name}`: Top-1 = `{'hit' if target_case['top1_hit'] else 'miss'}`，Top-{topk} = `{'hit' if any(row['hit'] for row in target_case['rows'][:topk]) else 'miss'}`，Top-1 distance = `{target_case['top1_distance']:.4f}`" if target_case["top1_distance"] is not None else f"- `{target_name}`: 无结果",
            "- score 是平均欧式距离，越小越相似。",
        ]
    )


def comparison_overview_markdown(compare_pairs: dict[str, Any]) -> str:
    return "\n".join(
        [
            "## `00500` vs `01000` 固定测试集对照",
            f"- 共享 probe 数量: `{compare_pairs['probe_count']}`",
            f"- Top-1 改善数: `{compare_pairs['improved_top1']}`",
            f"- Top-1 回退数: `{compare_pairs['regressed_top1']}`",
            f"- Top-1 不变数: `{compare_pairs['unchanged_top1']}`",
            f"- Top-5 改善数: `{compare_pairs['improved_top5']}`",
            f"- Top-5 回退数: `{compare_pairs['regressed_top5']}`",
        ]
    )


def result_for_probe_key(
    probe_key: str,
    baseline_lookup: dict[str, Any],
    target_lookup: dict[str, Any],
    baseline_analysis: dict[str, Any],
    target_analysis: dict[str, Any],
    topk: int,
):
    baseline_case = baseline_lookup[probe_key]
    target_case = target_lookup[probe_key]
    probe_image = make_sequence_strip(target_case["probe"]["pkl_path"])
    return (
        probe_markdown(target_case, target_analysis),
        probe_image,
        compare_markdown(baseline_case, target_case, baseline_analysis, target_analysis, topk),
        make_gallery_items(baseline_case["rows"][:topk]),
        make_table(baseline_case["rows"][:topk]),
        checkpoint_status_markdown(baseline_case, baseline_analysis, topk),
        make_gallery_items(target_case["rows"][:topk]),
        make_table(target_case["rows"][:topk]),
        checkpoint_status_markdown(target_case, target_analysis, topk),
    )


def run_manual_query(
    subject: str,
    sequence_type: str,
    view: str,
    topk: int,
    baseline_lookup: dict[str, Any],
    target_lookup: dict[str, Any],
    baseline_analysis: dict[str, Any],
    target_analysis: dict[str, Any],
):
    probe_key = build_probe_key(subject, sequence_type, view)
    if probe_key not in target_lookup:
        raise gr.Error(f"找不到 probe: {probe_key}")
    outputs = result_for_probe_key(
        probe_key,
        baseline_lookup,
        target_lookup,
        baseline_analysis,
        target_analysis,
        topk,
    )
    return subject, sequence_type, view, *outputs


def choose_random_key(keys: list[str]) -> str:
    if not keys:
        raise gr.Error("当前没有可用案例。")
    return random.choice(keys)


def parse_probe_key(probe_key: str) -> tuple[str, str, str]:
    parts = [part.strip() for part in probe_key.split("|")]
    if len(parts) != 3:
        raise ValueError(f"probe key 格式不正确: {probe_key}")
    return parts[0], parts[1], parts[2]


def run_case_key(
    probe_key: str,
    topk: int,
    baseline_lookup: dict[str, Any],
    target_lookup: dict[str, Any],
    baseline_analysis: dict[str, Any],
    target_analysis: dict[str, Any],
):
    subject, sequence_type, view = parse_probe_key(probe_key)
    outputs = result_for_probe_key(
        probe_key,
        baseline_lookup,
        target_lookup,
        baseline_analysis,
        target_analysis,
        topk,
    )
    return subject, sequence_type, view, *outputs


def pick_default_probe_key(target_analysis: dict[str, Any], compare_pairs: dict[str, Any]) -> str:
    improved = improved_case_choices(compare_pairs, limit=1)
    if improved:
        return improved[0]
    failures = case_choices(target_analysis, "failure", limit=1)
    if failures:
        return failures[0]
    successes = case_choices(target_analysis, "success", limit=1)
    if successes:
        return successes[0]
    raise RuntimeError("没有可用 probe 样本。")


def build_track_table(session: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for track in session.get("tracks", []):
        rows.append(
            {
                "track_id": track["track_id"],
                "status": track["status"],
                "quality_note": track["quality_note"],
                "frames": track["frame_count"],
                "valid_masks": track.get("valid_frames", 0),
                "used_frames": track.get("used_frames", 0),
                "avg_bbox_area": round(float(track.get("avg_bbox_area", 0.0)), 1),
                "avg_conf": round(float(track.get("avg_confidence", 0.0)), 4),
            }
        )
    return pd.DataFrame(rows)


def video_session_markdown(session: dict[str, Any]) -> str:
    selected_track = session.get("selected_track_id") or "无"
    ok_tracks = sum(1 for track in session.get("tracks", []) if track["status"] == "ok")
    return "\n".join(
        [
            "## 真实视频 demo 说明",
            "- 这里展示的是 `真实视频 -> 跟踪 -> silhouette -> 当前 checkpoint 检索` 的最小演示链路。",
            "- 它不是官方 benchmark，也不是完整业务系统。",
            f"- 输入视频: `{session.get('input_video', '')}`",
            f"- 标注预览视频: `{session.get('annotated_video', '')}`",
            f"- 当前可用 track 数: `{ok_tracks}` / 总 track 数 `{len(session.get('tracks', []))}`",
            f"- 当前自动选择 track: `{selected_track}`",
            f"- gallery 视角过滤: `{session.get('gallery_view', 'all')}`",
        ]
    )


def video_probe_markdown(track: dict[str, Any], session: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"**Track**: `{track['track_id']}`",
            f"**状态**: `{track['status']}`",
            f"**质量说明**: `{track['quality_note']}`",
            f"**原始帧数**: `{track['frame_count']}`",
            f"**有效分割帧数**: `{track.get('valid_frames', 0)}`",
            f"**用于检索的连续帧数**: `{track.get('used_frames', 0)}`",
            f"**track clip**: `{track.get('clip_path', '') or '-'}`",
            f"**silhouette pkl**: `{track.get('silhouette_pkl', '') or '-'}`",
        ]
    )


def video_status_markdown(track: dict[str, Any], session: dict[str, Any], topk: int) -> str:
    if track["status"] != "ok":
        return "\n".join(
            [
                "该 track 当前还不能进入检索。",
                f"quality_note = `{track['quality_note']}`",
                "常见原因是：人框太小、分割失败、连续有效帧不足、视频本身不适合步态序列。",
            ]
        )
    top1_subject = track.get("top1_subject", "")
    top1_distance = track.get("top1_distance")
    distance_text = f"{top1_distance:.4f}" if top1_distance is not None else "N/A"
    expected_subject = session.get("expected_subject", "") or "未提供"
    return "\n".join(
        [
            f"Top-{topk} 基于当前 `formal_conservative_real-01000.pt` 的真实 embedding 检索。",
            f"当前 gallery 视角过滤：`{session.get('gallery_view', 'all')}`",
            f"预期 subject（可选）：`{expected_subject}`",
            f"Top-1 候选：`{top1_subject}`，distance=`{distance_text}`",
            "score 是平均欧式距离，越小越相似。",
        ]
    )


def video_track_choices(session: dict[str, Any]) -> list[str]:
    return [track["track_id"] for track in session.get("tracks", [])]


def select_track(session: dict[str, Any], track_id: str, topk: int):
    if not session or not session.get("tracks"):
        raise gr.Error("请先处理一段真实视频。")
    track = next((item for item in session["tracks"] if item["track_id"] == track_id), None)
    if track is None:
        raise gr.Error(f"找不到 track: {track_id}")
    probe_image = track.get("silhouette_strip", "")
    probe_value = probe_image if probe_image else None
    track_clip = track.get("clip_path", "")
    track_clip_value = track_clip if track_clip else None
    rows = track.get("rows", [])[:topk]
    return (
        video_probe_markdown(track, session),
        track_clip_value,
        probe_value,
        make_gallery_items(rows) if rows else [],
        make_table(rows) if rows else pd.DataFrame(columns=["rank", "subject_id", "sequence_type", "view", "distance", "hit"]),
        video_status_markdown(track, session, topk),
    )


def resolve_video_input(uploaded_video: str | None, local_video_path: str | None) -> Path:
    if uploaded_video:
        return Path(uploaded_video).resolve()
    if local_video_path and local_video_path.strip():
        return Path(local_video_path.strip()).expanduser().resolve()
    raise gr.Error("请上传真实视频，或者填写服务器上的本地视频绝对路径。")


def run_real_video(
    uploaded_video: str | None,
    local_video_path: str | None,
    gallery_view: str,
    expected_subject: str,
    topk: int,
    detection_device: str,
    gallery_cache: dict[str, Any],
    cfg_path: Path,
    checkpoint_iter: int,
    run_root: Path,
):
    source_video = resolve_video_input(uploaded_video, local_video_path)
    session = process_real_video_demo(
        video_path=source_video,
        gallery_cache=gallery_cache,
        cfg_path=cfg_path,
        checkpoint_iter=checkpoint_iter,
        run_root=run_root,
        topk=VIDEO_MAX_TOPK,
        gallery_view=None if gallery_view == "all" else gallery_view,
        expected_subject=expected_subject.strip() or None,
        detection_device=None if detection_device == "auto" else detection_device,
    )
    track_choices = video_track_choices(session)
    selected_track = session.get("selected_track_id")
    if selected_track is None and track_choices:
        selected_track = track_choices[0]
        session["selected_track_id"] = selected_track
    track_update = gr.update(choices=track_choices, value=selected_track)
    probe_info, track_clip, probe_image, gallery_items, table_df, status_md = (
        select_track(session, selected_track, topk) if selected_track else (
            "没有可用 track。",
            None,
            None,
            [],
            pd.DataFrame(columns=["rank", "subject_id", "sequence_type", "view", "distance", "hit"]),
            "当前没有通过质量筛选并可进入检索的 track。",
        )
    )
    return (
        session,
        track_update,
        build_track_table(session),
        session.get("annotated_video"),
        video_session_markdown(session),
        probe_info,
        track_clip,
        probe_image,
        gallery_items,
        table_df,
        status_md,
    )


def build_demo(
    primary_cache: dict[str, Any],
    primary_analysis: dict[str, Any],
    compare_analysis: dict[str, Any],
    topk_default: int,
    cfg_path: Path,
    checkpoint_iter: int,
    run_root: Path,
) -> gr.Blocks:
    baseline_analysis = compare_analysis
    target_analysis = primary_analysis
    baseline_lookup = case_map(baseline_analysis)
    target_lookup = case_map(target_analysis)
    compare_pairs = comparison_summary(baseline_analysis, target_analysis)

    labels = list_probe_labels(target_analysis)
    views = list_probe_views(target_analysis)
    success_keys = case_choices(target_analysis, "success")
    failure_keys = case_choices(target_analysis, "failure")
    improved_keys = improved_case_choices(compare_pairs)
    max_topk = min(int(target_analysis["topk"]), int(baseline_analysis["topk"]))
    default_probe_key = pick_default_probe_key(target_analysis, compare_pairs)
    default_subject, default_type, default_view = parse_probe_key(default_probe_key)
    default_success = success_keys[0] if success_keys else None
    default_failure = failure_keys[0] if failure_keys else None
    default_improved = improved_keys[0] if improved_keys else None
    gallery_view_choices = ["all"] + list_gallery_views(primary_cache)

    with gr.Blocks(title="步态检索分析与真实视频 Demo") as demo:
        gr.Markdown(
            "\n".join(
                [
                    "# 步态检索分析与真实视频 Demo",
                    "同一个入口里包含两部分：上半是数据集检索闭环分析，下半是把真实视频接入当前 checkpoint 的最小演示。",
                ]
            )
        )

        with gr.Tabs():
            with gr.TabItem("数据集检索分析"):
                summary_md = gr.Markdown(fixed_definition_markdown(target_analysis, baseline_analysis))
                summary_table = gr.Dataframe(
                    value=make_summary_table(target_analysis, baseline_analysis),
                    interactive=False,
                    wrap=True,
                    label="固定测试集 Top-1 / Top-5 汇总",
                )
                compare_overview = gr.Markdown(comparison_overview_markdown(compare_pairs))

                with gr.Row():
                    with gr.Column(scale=1):
                        subject = gr.Dropdown(choices=labels, value=default_subject, label="Probe Subject")
                        sequence_type = gr.Dropdown(choices=list(PROBE_TYPES), value=default_type, label="Probe Type")
                        view = gr.Dropdown(choices=views, value=default_view, label="Probe View")
                        topk = gr.Slider(minimum=1, maximum=max_topk, value=min(topk_default, max_topk), step=1, label="Top-K")
                        manual_btn = gr.Button("按指定 Probe 对比", variant="primary")
                    with gr.Column(scale=1):
                        success_case = gr.Dropdown(choices=success_keys, value=default_success, label="01000 成功案例")
                        failure_case = gr.Dropdown(choices=failure_keys, value=default_failure, label="01000 失败案例")
                        improved_case = gr.Dropdown(choices=improved_keys, value=default_improved, label="00500->01000 改善案例")
                        with gr.Row():
                            load_success_btn = gr.Button("加载成功案例")
                            load_failure_btn = gr.Button("加载失败案例")
                        with gr.Row():
                            load_improved_btn = gr.Button("加载改善案例")
                            random_btn = gr.Button("随机 Probe")

                probe_info = gr.Markdown()
                compare_info = gr.Markdown()

                with gr.Row():
                    with gr.Column(scale=1):
                        probe_image = gr.Image(label="Probe 序列摘要", type="pil", interactive=False)
                    with gr.Column(scale=1):
                        gr.Markdown(
                            "\n".join(
                                [
                                    "### 页面说明",
                                    "- 左下和右下是同一个 probe 在两个 checkpoint 下的 Top-K 检索结果。",
                                    "- `hit/miss` 只看是不是同一 subject。",
                                    "- score 是 distance，越小越相似。",
                                ]
                            )
                        )

                with gr.Row():
                    with gr.Column(scale=1):
                        baseline_status = gr.Markdown()
                        baseline_gallery = gr.Gallery(label=f"{analysis_label(baseline_analysis)} Top-K", columns=2, object_fit="contain", height="auto")
                        baseline_table = gr.Dataframe(
                            headers=["rank", "subject_id", "sequence_type", "view", "distance", "hit"],
                            interactive=False,
                            wrap=True,
                            label=f"{analysis_label(baseline_analysis)} 结果表",
                        )
                    with gr.Column(scale=1):
                        target_status = gr.Markdown()
                        target_gallery = gr.Gallery(label=f"{analysis_label(target_analysis)} Top-K", columns=2, object_fit="contain", height="auto")
                        target_table = gr.Dataframe(
                            headers=["rank", "subject_id", "sequence_type", "view", "distance", "hit"],
                            interactive=False,
                            wrap=True,
                            label=f"{analysis_label(target_analysis)} 结果表",
                        )

                dataset_outputs = [
                    subject,
                    sequence_type,
                    view,
                    probe_info,
                    probe_image,
                    compare_info,
                    baseline_gallery,
                    baseline_table,
                    baseline_status,
                    target_gallery,
                    target_table,
                    target_status,
                ]

                manual_btn.click(
                    fn=lambda s, t, v, k: run_manual_query(
                        s,
                        t,
                        v,
                        int(k),
                        baseline_lookup,
                        target_lookup,
                        baseline_analysis,
                        target_analysis,
                    ),
                    inputs=[subject, sequence_type, view, topk],
                    outputs=dataset_outputs,
                )
                load_success_btn.click(
                    fn=lambda key, k: run_case_key(
                        key,
                        int(k),
                        baseline_lookup,
                        target_lookup,
                        baseline_analysis,
                        target_analysis,
                    ),
                    inputs=[success_case, topk],
                    outputs=dataset_outputs,
                )
                load_failure_btn.click(
                    fn=lambda key, k: run_case_key(
                        key,
                        int(k),
                        baseline_lookup,
                        target_lookup,
                        baseline_analysis,
                        target_analysis,
                    ),
                    inputs=[failure_case, topk],
                    outputs=dataset_outputs,
                )
                load_improved_btn.click(
                    fn=lambda key, k: run_case_key(
                        key,
                        int(k),
                        baseline_lookup,
                        target_lookup,
                        baseline_analysis,
                        target_analysis,
                    ),
                    inputs=[improved_case, topk],
                    outputs=dataset_outputs,
                )
                random_btn.click(
                    fn=lambda k: run_case_key(
                        choose_random_key(list(target_lookup.keys())),
                        int(k),
                        baseline_lookup,
                        target_lookup,
                        baseline_analysis,
                        target_analysis,
                    ),
                    inputs=[topk],
                    outputs=dataset_outputs,
                )

                demo.load(
                    fn=lambda: run_case_key(
                        default_probe_key,
                        min(topk_default, max_topk),
                        baseline_lookup,
                        target_lookup,
                        baseline_analysis,
                        target_analysis,
                    ),
                    outputs=dataset_outputs,
                )

            with gr.TabItem("真实视频 Demo"):
                gr.Markdown(
                    "\n".join(
                        [
                            "## 真实视频 -> 步态检索 Demo",
                            "- 这里复用当前真实 checkpoint 和真实 retrieval 逻辑。",
                            "- 它展示的是 **真实视频 demo**，不是官方 benchmark。",
                            "- 如果你知道这段视频对应的 CASIA-B subject，可选填 `expected subject`，页面就会标记 hit/miss；不知道就留空。",
                        ]
                    )
                )
                video_state = gr.State({})

                with gr.Row():
                    with gr.Column(scale=1):
                        uploaded_video = gr.File(label="上传真实视频", file_types=["video"], type="filepath")
                        local_video_path = gr.Textbox(
                            label="或填写服务器上的本地视频绝对路径",
                            placeholder="/abs/path/to/video.mp4",
                        )
                        gallery_view = gr.Dropdown(choices=gallery_view_choices, value="all", label="Gallery 视角过滤")
                        expected_subject = gr.Textbox(label="Expected Subject（可选）", placeholder="例如 091")
                        video_topk = gr.Slider(minimum=1, maximum=VIDEO_MAX_TOPK, value=min(topk_default, VIDEO_MAX_TOPK), step=1, label="Top-K")
                        detection_device = gr.Dropdown(choices=["auto", "cpu", "cuda:0"], value="auto", label="视频检测设备")
                        process_btn = gr.Button("处理视频并检索", variant="primary")
                    with gr.Column(scale=1):
                        video_session_md = gr.Markdown(
                            "\n".join(
                                [
                                    "处理前说明：",
                                    "- 建议单人、全身尽量完整、连续行走。",
                                    "- 画面过暗、遮挡重、人物太小，都会直接影响 silhouette 质量。",
                                ]
                            )
                        )
                        track_dropdown = gr.Dropdown(choices=[], label="选择 track")
                        track_table = gr.Dataframe(
                            headers=["track_id", "status", "quality_note", "frames", "valid_masks", "used_frames", "avg_bbox_area", "avg_conf"],
                            interactive=False,
                            wrap=True,
                            label="Track 汇总",
                        )
                        load_track_btn = gr.Button("加载当前 track 结果")

                annotated_video = gr.Video(label="标注预览视频", interactive=False)
                with gr.Row():
                    with gr.Column(scale=1):
                        video_probe_info = gr.Markdown()
                        video_track_clip = gr.Video(label="当前选中 track clip", interactive=False)
                        video_probe_image = gr.Image(label="视频 track 的 silhouette 序列摘要", type="filepath", interactive=False)
                        video_status = gr.Markdown()
                    with gr.Column(scale=2):
                        video_gallery = gr.Gallery(label="Top-K 检索结果", columns=2, object_fit="contain", height="auto")
                        video_table = gr.Dataframe(
                            headers=["rank", "subject_id", "sequence_type", "view", "distance", "hit"],
                            interactive=False,
                            wrap=True,
                            label="真实视频检索结果表",
                        )

                process_btn.click(
                    fn=lambda uploaded, local_path, g_view, expected, k, det_dev: run_real_video(
                        uploaded,
                        local_path,
                        g_view,
                        expected,
                        int(k),
                        det_dev,
                        primary_cache,
                        cfg_path,
                        checkpoint_iter,
                        run_root,
                    ),
                    inputs=[uploaded_video, local_video_path, gallery_view, expected_subject, video_topk, detection_device],
                    outputs=[video_state, track_dropdown, track_table, annotated_video, video_session_md, video_probe_info, video_track_clip, video_probe_image, video_gallery, video_table, video_status],
                )
                load_track_btn.click(
                    fn=lambda session, track_id, k: select_track(session, track_id, int(k)),
                    inputs=[video_state, track_dropdown, video_topk],
                    outputs=[video_probe_info, video_track_clip, video_probe_image, video_gallery, video_table, video_status],
                )

    return demo


def main() -> int:
    args = parse_args()
    primary_cache_path = Path(args.primary_cache_path).resolve()
    compare_cache_path = Path(args.compare_cache_path).resolve()
    primary_analysis_path = Path(args.primary_analysis_path).resolve()
    compare_analysis_path = Path(args.compare_analysis_path).resolve()
    cfg_path = Path(args.cfg_path).resolve()
    run_root = Path(args.run_root).resolve()

    ensure_cache(primary_cache_path, cfg_path, args.primary_checkpoint_iter, 29712)
    ensure_cache(compare_cache_path, cfg_path, args.compare_checkpoint_iter, 29713)
    ensure_analysis(primary_analysis_path, primary_cache_path, args.topk_default)
    ensure_analysis(compare_analysis_path, compare_cache_path, args.topk_default)

    primary_cache = load_cache(primary_cache_path)
    primary_analysis = load_analysis(primary_analysis_path)
    compare_analysis = load_analysis(compare_analysis_path)
    demo = build_demo(
        primary_cache,
        primary_analysis,
        compare_analysis,
        args.topk_default,
        cfg_path,
        args.primary_checkpoint_iter,
        run_root,
    )
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
