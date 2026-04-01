from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from app import ExportSummary, process_video
from organize_tracks import organize_tracks


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
SUMMARY_HEADERS = [
    "status",
    "source_video",
    "result_video",
    "artifacts_dir",
    "data_dir",
    "track_id",
    "frame_count",
    "kept",
    "avg_bbox_area",
    "avg_confidence",
    "first_frame_idx",
    "last_frame_idx",
    "quality_pass",
    "quality_note",
    "error",
]


@dataclass(slots=True)
class VideoRunResult:
    source_video: str
    status: str
    result_video: Path
    artifacts_dir: Path
    data_dir: Path
    track_total: int = 0
    track_kept: int = 0
    track_low_quality: int = 0
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量步态采集处理脚本")
    parser.add_argument("--input-dir", default="samples", help="输入视频目录")
    parser.add_argument("--outputs-dir", default="outputs", help="标注结果视频目录")
    parser.add_argument("--artifacts-dir", default="artifacts", help="轨迹导出根目录")
    parser.add_argument("--data-dir", default="data", help="整理后的数据根目录")
    parser.add_argument("--reports-dir", default="reports", help="汇总报告目录")
    parser.add_argument("--identity", default="unknown_id", help="整理数据时使用的默认 identity 名称")
    parser.add_argument("--copy-mode", choices=["copy", "symlink"], default="copy", help="整理数据时复制或软链接")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的结果")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO 权重路径或名称")
    parser.add_argument("--det-conf", type=float, default=0.35, help="检测置信度阈值")
    parser.add_argument("--pose-conf", type=float, default=0.4, help="姿态检测置信度阈值")
    parser.add_argument("--device", default="cpu", help="推理设备")
    parser.add_argument("--export-format", choices=["frames", "clip", "both"], default="both", help="轨迹导出形式")
    parser.add_argument("--track-iou-threshold", type=float, default=0.3, help="tracking 的 IoU 阈值")
    parser.add_argument("--tracker-max-missing", dest="track_max_age", type=int, default=15, help="track 丢失后保留的最大帧数")
    parser.add_argument("--min-track-length", dest="min_track_frames", type=int, default=20, help="轨迹最少连续帧数")
    parser.add_argument("--min-bbox-size", type=int, default=40, help="最小 bbox 尺寸")
    parser.add_argument("--track-conf", dest="min_track_confidence", type=float, default=0.25, help="导出轨迹的最小检测置信度")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    outputs_dir = Path(args.outputs_dir)
    artifacts_dir = Path(args.artifacts_dir)
    data_dir = Path(args.data_dir)
    reports_dir = Path(args.reports_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    outputs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    videos = discover_videos(input_dir)
    if not videos:
        raise FileNotFoundError(f"在 {input_dir} 下没有找到可处理的视频文件")

    all_track_rows: list[dict[str, str]] = []
    video_results: list[VideoRunResult] = []

    for video_path in videos:
        source_video = video_path.name
        video_stem = video_path.stem
        result_video = outputs_dir / f"{video_stem}_result.mp4"
        video_artifacts_dir = artifacts_dir / video_stem
        video_data_dir = data_dir / video_stem
        data_manifest = video_data_dir / args.identity / "manifest.tsv"

        if not args.overwrite and is_complete_run(result_video, video_artifacts_dir, data_manifest):
            rows = load_track_rows_from_manifest(
                status="skipped",
                source_video=source_video,
                result_video=result_video,
                artifacts_dir=video_artifacts_dir,
                data_dir=video_data_dir,
                manifest_path=video_artifacts_dir / "manifest.tsv",
                min_track_frames=args.min_track_frames,
                min_bbox_size=args.min_bbox_size,
                min_track_confidence=args.min_track_confidence,
            )
            if not rows:
                rows = [empty_track_row("skipped", source_video, result_video, video_artifacts_dir, video_data_dir, "")]
            all_track_rows.extend(rows)
            video_results.append(build_video_run_result(source_video, "skipped", result_video, video_artifacts_dir, video_data_dir, rows))
            continue

        cleanup_path(result_video)
        cleanup_path(video_artifacts_dir)
        cleanup_path(video_data_dir)

        try:
            summaries = process_video(
                input_path=video_path,
                output_path=result_video,
                model_path=args.model,
                det_conf=args.det_conf,
                pose_conf=args.pose_conf,
                device=args.device,
                enable_tracking=True,
                export_tracks=True,
                export_format=args.export_format,
                artifacts_dir=video_artifacts_dir,
                track_iou_threshold=args.track_iou_threshold,
                track_max_age=args.track_max_age,
                min_track_frames=args.min_track_frames,
                min_bbox_size=args.min_bbox_size,
                min_track_confidence=args.min_track_confidence,
            )
            kept_summaries = [summary for summary in summaries if summary.kept]
            if kept_summaries:
                organize_tracks(
                    source_root=video_artifacts_dir,
                    target_root=video_data_dir,
                    identity=args.identity,
                    copy_mode=args.copy_mode,
                )
            else:
                write_empty_data_manifest(video_data_dir, args.identity)

            rows = build_track_rows(
                status="success",
                summaries=summaries,
                result_video=result_video,
                artifacts_dir=video_artifacts_dir,
                data_dir=video_data_dir,
                min_track_frames=args.min_track_frames,
                min_bbox_size=args.min_bbox_size,
                min_track_confidence=args.min_track_confidence,
            )
            if not rows:
                rows = [empty_track_row("success", source_video, result_video, video_artifacts_dir, video_data_dir, "")]
            all_track_rows.extend(rows)
            video_results.append(build_video_run_result(source_video, "success", result_video, video_artifacts_dir, video_data_dir, rows))
        except Exception as exc:
            error_message = str(exc)
            row = empty_track_row("failure", source_video, result_video, video_artifacts_dir, video_data_dir, error_message)
            all_track_rows.append(row)
            video_results.append(build_video_run_result(source_video, "failure", result_video, video_artifacts_dir, video_data_dir, [row]))

    write_summary_tsv(reports_dir / "summary.tsv", all_track_rows)
    write_summary_md(
        reports_dir / "summary.md",
        input_dir=input_dir,
        video_results=video_results,
        track_rows=all_track_rows,
    )
    cleanup_appledouble_files(outputs_dir)
    cleanup_appledouble_files(artifacts_dir)
    cleanup_appledouble_files(data_dir)
    cleanup_appledouble_files(reports_dir)

    success_count = sum(1 for item in video_results if item.status == "success")
    skipped_count = sum(1 for item in video_results if item.status == "skipped")
    failure_count = sum(1 for item in video_results if item.status == "failure")
    total_kept = sum(item.track_kept for item in video_results)
    print(
        f"批量处理完成: videos={len(video_results)} success={success_count} skipped={skipped_count} "
        f"failure={failure_count} kept_tracks={total_kept}",
        flush=True,
    )
    print(f"报告已写入: {reports_dir / 'summary.tsv'}", flush=True)
    print(f"报告已写入: {reports_dir / 'summary.md'}", flush=True)


def discover_videos(input_dir: Path) -> list[Path]:
    videos = []
    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        videos.append(path)
    return videos


def is_complete_run(result_video: Path, artifacts_dir: Path, data_manifest: Path) -> bool:
    return result_video.exists() and (artifacts_dir / "manifest.tsv").exists() and data_manifest.exists()


def cleanup_path(path: Path) -> None:
    if not path.exists():
        sibling = path.parent / f"._{path.name}"
        if sibling.exists():
            sibling.unlink(missing_ok=True)
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)
    sibling = path.parent / f"._{path.name}"
    if sibling.exists():
        sibling.unlink(missing_ok=True)


def build_track_rows(
    *,
    status: str,
    summaries: list[ExportSummary],
    result_video: Path,
    artifacts_dir: Path,
    data_dir: Path,
    min_track_frames: int,
    min_bbox_size: int,
    min_track_confidence: float,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for summary in summaries:
        quality_note = evaluate_quality(summary, min_track_frames, min_bbox_size, min_track_confidence)
        quality_pass = "1" if summary.kept and quality_note == "ok" else "0"
        rows.append(
            {
                "status": status,
                "source_video": summary.source_video,
                "result_video": relpath(result_video),
                "artifacts_dir": relpath(artifacts_dir),
                "data_dir": relpath(data_dir),
                "track_id": f"{summary.track_id:04d}",
                "frame_count": str(summary.frame_count),
                "kept": "1" if summary.kept else "0",
                "avg_bbox_area": f"{summary.avg_bbox_area:.2f}",
                "avg_confidence": f"{summary.avg_confidence:.4f}",
                "first_frame_idx": str(summary.first_frame_idx or ""),
                "last_frame_idx": str(summary.last_frame_idx or ""),
                "quality_pass": quality_pass,
                "quality_note": quality_note,
                "error": "",
            }
        )
    return rows


def load_track_rows_from_manifest(
    *,
    status: str,
    source_video: str,
    result_video: Path,
    artifacts_dir: Path,
    data_dir: Path,
    manifest_path: Path,
    min_track_frames: int,
    min_bbox_size: int,
    min_track_confidence: float,
) -> list[dict[str, str]]:
    if not manifest_path.exists():
        return []
    rows: list[dict[str, str]] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for record in reader:
            kept = record.get("kept", "0") in {"1", "true", "True"}
            frame_count = int(record.get("frame_count") or 0)
            avg_bbox_area = float(record.get("avg_bbox_area") or 0.0)
            avg_confidence = float(record.get("avg_confidence") or 0.0)
            quality_note = evaluate_quality_from_values(
                kept=kept,
                frame_count=frame_count,
                avg_bbox_area=avg_bbox_area,
                avg_confidence=avg_confidence,
                min_track_frames=min_track_frames,
                min_bbox_size=min_bbox_size,
                min_track_confidence=min_track_confidence,
            )
            rows.append(
                {
                    "status": status,
                    "source_video": record.get("source_video") or source_video,
                    "result_video": relpath(result_video),
                    "artifacts_dir": relpath(artifacts_dir),
                    "data_dir": relpath(data_dir),
                    "track_id": record.get("track_id", ""),
                    "frame_count": str(frame_count),
                    "kept": "1" if kept else "0",
                    "avg_bbox_area": f"{avg_bbox_area:.2f}",
                    "avg_confidence": f"{avg_confidence:.4f}",
                    "first_frame_idx": record.get("first_frame_idx", ""),
                    "last_frame_idx": record.get("last_frame_idx", ""),
                    "quality_pass": "1" if kept and quality_note == "ok" else "0",
                    "quality_note": quality_note,
                    "error": "",
                }
            )
    return rows


def build_video_run_result(
    source_video: str,
    status: str,
    result_video: Path,
    artifacts_dir: Path,
    data_dir: Path,
    rows: list[dict[str, str]],
) -> VideoRunResult:
    track_rows = [row for row in rows if row.get("track_id")]
    track_total = len(track_rows)
    track_kept = sum(1 for row in track_rows if row.get("kept") == "1")
    track_low_quality = sum(1 for row in track_rows if row.get("quality_pass") != "1")
    error = ""
    for row in rows:
        if row.get("error"):
            error = row["error"]
            break
    return VideoRunResult(
        source_video=source_video,
        status=status,
        result_video=result_video,
        artifacts_dir=artifacts_dir,
        data_dir=data_dir,
        track_total=track_total,
        track_kept=track_kept,
        track_low_quality=track_low_quality,
        error=error,
    )


def empty_track_row(
    status: str,
    source_video: str,
    result_video: Path,
    artifacts_dir: Path,
    data_dir: Path,
    error: str,
) -> dict[str, str]:
    return {
        "status": status,
        "source_video": source_video,
        "result_video": relpath(result_video),
        "artifacts_dir": relpath(artifacts_dir),
        "data_dir": relpath(data_dir),
        "track_id": "",
        "frame_count": "0",
        "kept": "0",
        "avg_bbox_area": "0.00",
        "avg_confidence": "0.0000",
        "first_frame_idx": "",
        "last_frame_idx": "",
        "quality_pass": "0",
        "quality_note": "no_tracks" if not error else "error",
        "error": error,
    }


def write_empty_data_manifest(data_dir: Path, identity: str) -> None:
    identity_dir = data_dir / identity
    identity_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = identity_dir / "manifest.tsv"
    manifest_path.write_text("identity\tsequence\ttrack_id\tframes\tclip\tsource_video\n", encoding="utf-8")
    cleanup_appledouble_files(data_dir)


def evaluate_quality(
    summary: ExportSummary,
    min_track_frames: int,
    min_bbox_size: int,
    min_track_confidence: float,
) -> str:
    return evaluate_quality_from_values(
        kept=summary.kept,
        frame_count=summary.frame_count,
        avg_bbox_area=summary.avg_bbox_area,
        avg_confidence=summary.avg_confidence,
        min_track_frames=min_track_frames,
        min_bbox_size=min_bbox_size,
        min_track_confidence=min_track_confidence,
    )


def evaluate_quality_from_values(
    *,
    kept: bool,
    frame_count: int,
    avg_bbox_area: float,
    avg_confidence: float,
    min_track_frames: int,
    min_bbox_size: int,
    min_track_confidence: float,
) -> str:
    reasons: list[str] = []
    if not kept or frame_count < min_track_frames:
        reasons.append("too_short")
    elif frame_count < max(min_track_frames + 5, int(min_track_frames * 1.5)):
        reasons.append("borderline_short")
    if avg_bbox_area > 0 and avg_bbox_area < float((min_bbox_size * min_bbox_size) * 2):
        reasons.append("small_bbox")
    if avg_confidence > 0 and avg_confidence < min_track_confidence + 0.10:
        reasons.append("low_confidence")
    return ",".join(reasons) if reasons else "ok"


def write_summary_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_HEADERS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_md(
    path: Path,
    *,
    input_dir: Path,
    video_results: list[VideoRunResult],
    track_rows: list[dict[str, str]],
) -> None:
    success_count = sum(1 for item in video_results if item.status == "success")
    skipped_count = sum(1 for item in video_results if item.status == "skipped")
    failure_count = sum(1 for item in video_results if item.status == "failure")
    kept_tracks = sum(item.track_kept for item in video_results)
    low_quality_tracks = sum(item.track_low_quality for item in video_results)
    suspicious_rows = [
        row for row in track_rows if row.get("track_id") and (row.get("quality_pass") != "1" or row.get("kept") != "1")
    ]
    failed_rows = [row for row in track_rows if row.get("status") == "failure"]

    lines = [
        "# 批量采集汇总",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 输入目录：`{relpath(input_dir)}`",
        f"- 扫描到视频数：{len(video_results)}",
        f"- 成功：{success_count}",
        f"- 跳过：{skipped_count}",
        f"- 失败：{failure_count}",
        f"- 导出轨迹数：{kept_tracks}",
        f"- 可能质量较差轨迹数：{low_quality_tracks}",
        "",
        "## 视频汇总",
        "",
        "| 视频 | 状态 | 导出轨迹 | 可疑轨迹 | 结果视频 | artifacts | data | 备注 |",
        "| --- | --- | ---: | ---: | --- | --- | --- | --- |",
    ]

    for item in video_results:
        note = item.error or ("已跳过已有结果" if item.status == "skipped" else "")
        lines.append(
            f"| {item.source_video} | {item.status} | {item.track_kept} | {item.track_low_quality} | "
            f"`{relpath(item.result_video)}` | `{relpath(item.artifacts_dir)}` | `{relpath(item.data_dir)}` | {note} |"
        )

    lines.extend(
        [
            "",
            "## 可疑轨迹",
            "",
        ]
    )

    if suspicious_rows:
        lines.extend(
            [
                "| 视频 | track_id | 帧数 | kept | avg_bbox_area | avg_confidence | 说明 |",
                "| --- | ---: | ---: | --- | ---: | ---: | --- |",
            ]
        )
        for row in suspicious_rows:
            lines.append(
                f"| {row['source_video']} | {row['track_id']} | {row['frame_count']} | {row['kept']} | "
                f"{row['avg_bbox_area']} | {row['avg_confidence']} | {row['quality_note']} |"
            )
    else:
        lines.append("没有发现明显的低质量轨迹。")

    lines.extend(["", "## 失败视频", ""])
    if failed_rows:
        lines.extend(["| 视频 | 错误 |", "| --- | --- |"])
        for row in failed_rows:
            lines.append(f"| {row['source_video']} | {row['error']} |")
    else:
        lines.append("没有失败视频。")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def cleanup_appledouble_files(root: Path) -> None:
    sibling = root.parent / f"._{root.name}"
    if sibling.exists():
        sibling.unlink(missing_ok=True)
    if not root.exists():
        return
    for path in root.rglob("._*"):
        if path.is_file() or path.is_symlink():
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
