from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from src.detector import DetectorConfig, PersonDetector
from src.models import Detection, PersonAnnotation
from src.orientation import OrientationClassifier, OrientationConfig
from src.pose_estimator import PoseConfig, PoseEstimator
from src.tracker import SimpleIoUTracker
from src.visualize import annotate_frame


@dataclass(slots=True)
class ExportConfig:
    root_dir: Path
    export_crops: bool = True
    export_clips: bool = False
    min_track_frames: int = 20
    min_bbox_size: int = 40
    min_confidence: float = 0.25
    clip_size: tuple[int, int] = (256, 256)
    crop_padding: float = 0.15


@dataclass(slots=True)
class ExportSummary:
    track_id: int
    source_video: str
    frame_count: int
    kept: bool
    avg_bbox_area: float
    avg_confidence: float
    first_frame_idx: int | None
    last_frame_idx: int | None
    crop_dir: Path | None
    clip_path: Path | None


@dataclass(slots=True)
class _TrackWriter:
    track_id: int
    base_dir: Path
    export_crops: bool
    export_clips: bool
    clip_size: tuple[int, int]
    fps: float
    source_video: str
    frame_count: int = 0
    bbox_area_sum: float = 0.0
    confidence_sum: float = 0.0
    crop_dir: Path | None = None
    clip_path: Path | None = None
    clip_writer: cv2.VideoWriter | None = None
    first_frame_idx: int | None = None
    last_frame_idx: int | None = None

    def observe(self, frame_idx: int, detection: Detection, crop_bgr: np.ndarray) -> None:
        if self.first_frame_idx is None:
            self.first_frame_idx = frame_idx
        self.last_frame_idx = frame_idx
        self.frame_count += 1
        x1, y1, x2, y2 = detection.bbox_xyxy
        self.bbox_area_sum += max(0.0, (x2 - x1) * (y2 - y1))
        self.confidence_sum += max(0.0, detection.confidence)

        if self.export_crops:
            self._ensure_crop_dir()
            assert self.crop_dir is not None
            frame_path = self.crop_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(frame_path), crop_bgr)

        if self.export_clips:
            self._ensure_clip_writer()
            assert self.clip_writer is not None
            self.clip_writer.write(_fit_to_size(crop_bgr, self.clip_size))

    def finalize(self, min_track_frames: int) -> ExportSummary:
        self._close_clip_writer()
        kept = self.frame_count >= min_track_frames

        if not kept:
            if self.crop_dir and self.crop_dir.exists():
                shutil.rmtree(self.crop_dir, ignore_errors=True)
            if self.clip_path and self.clip_path.exists():
                self.clip_path.unlink()
            return ExportSummary(
                track_id=self.track_id,
                source_video=self.source_video,
                frame_count=self.frame_count,
                kept=False,
                avg_bbox_area=self._avg_bbox_area(),
                avg_confidence=self._avg_confidence(),
                first_frame_idx=self.first_frame_idx,
                last_frame_idx=self.last_frame_idx,
                crop_dir=None,
                clip_path=None,
            )

        if self.crop_dir is not None:
            metadata = {
                "track_id": self.track_id,
                "source_video": self.source_video,
                "frame_count": self.frame_count,
                "kept": True,
                "avg_bbox_area": self._avg_bbox_area(),
                "avg_confidence": self._avg_confidence(),
                "first_frame_idx": self.first_frame_idx,
                "last_frame_idx": self.last_frame_idx,
            }
            with open(self.crop_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

        return ExportSummary(
            track_id=self.track_id,
            source_video=self.source_video,
            frame_count=self.frame_count,
            kept=True,
            avg_bbox_area=self._avg_bbox_area(),
            avg_confidence=self._avg_confidence(),
            first_frame_idx=self.first_frame_idx,
            last_frame_idx=self.last_frame_idx,
            crop_dir=self.crop_dir,
            clip_path=self.clip_path,
        )

    def _ensure_crop_dir(self) -> None:
        if self.crop_dir is None:
            self.crop_dir = self.base_dir / "tracks" / f"{self.track_id:04d}"
            self.crop_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_clip_writer(self) -> None:
        if self.clip_writer is not None:
            return
        clips_dir = self.base_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        self.clip_path = clips_dir / f"track_{self.track_id:04d}.mp4"
        self.clip_writer = cv2.VideoWriter(
            str(self.clip_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            self.clip_size,
        )
        if not self.clip_writer.isOpened():
            raise RuntimeError(f"无法创建轨迹视频: {self.clip_path}")

    def _close_clip_writer(self) -> None:
        if self.clip_writer is not None:
            self.clip_writer.release()
            self.clip_writer = None

    def _avg_bbox_area(self) -> float:
        return self.bbox_area_sum / self.frame_count if self.frame_count else 0.0

    def _avg_confidence(self) -> float:
        return self.confidence_sum / self.frame_count if self.frame_count else 0.0


class _TrackExporter:
    def __init__(self, config: ExportConfig, fps: float, source_video: str) -> None:
        self.config = config
        self.fps = fps
        self.source_video = source_video
        self._writers: dict[int, _TrackWriter] = {}
        _cleanup_appledouble_files(self.config.root_dir)
        if self.config.root_dir.exists():
            shutil.rmtree(self.config.root_dir, ignore_errors=True)
        self.config.root_dir.mkdir(parents=True, exist_ok=True)

    def observe_frame(self, frame_idx: int, frame_bgr: np.ndarray, detections: Sequence[Detection]) -> None:
        for detection in detections:
            if detection.track_id is None:
                continue
            if not self._should_export(detection):
                continue
            crop = _extract_crop(frame_bgr, detection.bbox_xyxy, self.config.crop_padding)
            if crop.size == 0:
                continue
            writer = self._writers.get(detection.track_id)
            if writer is None:
                writer = _TrackWriter(
                    track_id=detection.track_id,
                    base_dir=self.config.root_dir,
                    export_crops=self.config.export_crops,
                    export_clips=self.config.export_clips,
                    clip_size=self.config.clip_size,
                    fps=self.fps,
                    source_video=self.source_video,
                )
                self._writers[detection.track_id] = writer
            writer.observe(frame_idx, detection, crop)

    def finalize(self) -> list[ExportSummary]:
        summaries: list[ExportSummary] = []
        for track_id in sorted(self._writers):
            summaries.append(self._writers[track_id].finalize(self.config.min_track_frames))
        self._write_manifest(summaries)
        _cleanup_appledouble_files(self.config.root_dir)
        return summaries

    def _should_export(self, detection: Detection) -> bool:
        x1, y1, x2, y2 = detection.bbox_xyxy
        width = x2 - x1
        height = y2 - y1
        return (
            detection.confidence >= self.config.min_confidence
            and width >= self.config.min_bbox_size
            and height >= self.config.min_bbox_size
        )

    def _write_manifest(self, summaries: Sequence[ExportSummary]) -> None:
        lines = [
            "track_id\tsource_video\tframe_count\tkept\tavg_bbox_area\tavg_confidence\tfirst_frame_idx\tlast_frame_idx\tcrop_dir\tclip_path"
        ]
        for summary in summaries:
            lines.append(
                "\t".join(
                    [
                        f"{summary.track_id:04d}",
                        summary.source_video,
                        str(summary.frame_count),
                        "1" if summary.kept else "0",
                        f"{summary.avg_bbox_area:.2f}",
                        f"{summary.avg_confidence:.4f}",
                        str(summary.first_frame_idx) if summary.first_frame_idx is not None else "",
                        str(summary.last_frame_idx) if summary.last_frame_idx is not None else "",
                        str(summary.crop_dir) if summary.crop_dir else "",
                        str(summary.clip_path) if summary.clip_path else "",
                    ]
                )
            )
        (self.config.root_dir / "manifest.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地人物背影/姿态识别与步态前处理 demo")
    parser.add_argument("--input", required=True, help="输入视频路径")
    parser.add_argument("--output", required=True, help="输出视频路径")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO 权重路径或名称")
    parser.add_argument("--det-conf", type=float, default=0.35, help="检测置信度阈值")
    parser.add_argument("--pose-conf", type=float, default=0.4, help="姿态检测置信度阈值")
    parser.add_argument("--device", default="cpu", help="推理设备，默认 cpu")
    parser.add_argument("--enable-tracking", action="store_true", help="启用单视频 track_id 跟踪")
    parser.add_argument("--export-tracks", action="store_true", help="导出轨迹序列到 artifacts/")
    parser.add_argument(
        "--export-format",
        choices=["frames", "clip", "both", "crops", "clips"],
        default="frames",
        help="轨迹导出形式",
    )
    parser.add_argument("--artifacts-dir", default="artifacts", help="轨迹导出目录")
    parser.add_argument("--track-iou-threshold", type=float, default=0.3, help="tracking 的 IoU 匹配阈值")
    parser.add_argument(
        "--track-max-age",
        "--tracker-max-missing",
        dest="track_max_age",
        type=int,
        default=15,
        help="track 丢失后保留的最大帧数",
    )
    parser.add_argument(
        "--min-track-frames",
        "--min-track-length",
        dest="min_track_frames",
        type=int,
        default=20,
        help="导出轨迹的最少连续帧数",
    )
    parser.add_argument("--min-bbox-size", type=int, default=40, help="导出轨迹的最小 bbox 尺寸")
    parser.add_argument(
        "--min-track-confidence",
        "--track-conf",
        dest="min_track_confidence",
        type=float,
        default=0.25,
        help="导出轨迹的最小检测置信度",
    )
    return parser.parse_args()


def process_video(
    input_path: Path,
    output_path: Path,
    model_path: str,
    det_conf: float,
    pose_conf: float,
    device: str | None,
    *,
    enable_tracking: bool = False,
    export_tracks: bool = False,
    export_format: str = "frames",
    artifacts_dir: Path | None = None,
    track_iou_threshold: float = 0.3,
    track_max_age: int = 15,
    min_track_frames: int = 20,
    min_bbox_size: int = 40,
    min_track_confidence: float = 0.25,
) -> list[ExportSummary]:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开输入视频: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        fps = 25.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"无法创建输出视频: {output_path}")

    detector = PersonDetector(DetectorConfig(model_path=model_path, conf=det_conf, device=device))
    pose_estimator = PoseEstimator(PoseConfig(min_detection_confidence=pose_conf))
    orientation_classifier = OrientationClassifier(OrientationConfig())
    tracker = SimpleIoUTracker(iou_threshold=track_iou_threshold, max_age=track_max_age) if enable_tracking or export_tracks else None

    exporter: _TrackExporter | None = None
    if export_tracks:
        normalized_export_format = {"crops": "frames", "clips": "clip"}.get(export_format, export_format)
        export_crops = normalized_export_format in ("frames", "both")
        export_clips = normalized_export_format in ("clip", "both")
        exporter = _TrackExporter(
            ExportConfig(
                root_dir=Path(artifacts_dir or Path("artifacts")),
                export_crops=export_crops,
                export_clips=export_clips,
                min_track_frames=min_track_frames,
                min_bbox_size=min_bbox_size,
                min_confidence=min_track_confidence,
            ),
            fps=fps,
            source_video=input_path.name,
        )

    summaries: list[ExportSummary] = []
    try:
        frame_idx = 1
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            detections = detector.detect(frame)
            if tracker is not None:
                tracked_detections = tracker.update(detections, frame_idx)
                detections = [item.detection.with_track_id(item.track_id) for item in tracked_detections]

            annotations: list[PersonAnnotation] = []
            for detection in detections:
                pose = pose_estimator.estimate(frame, detection.bbox_xyxy)
                orientation = orientation_classifier.classify(pose)
                annotations.append(PersonAnnotation(detection=detection, pose=pose, orientation=orientation))

            if exporter is not None:
                exporter.observe_frame(frame_idx, frame, detections)

            writer.write(annotate_frame(frame, annotations))
            frame_idx += 1
    finally:
        pose_estimator.close()
        writer.release()
        capture.release()
        if exporter is not None:
            summaries = exporter.finalize()
            kept = [item for item in summaries if item.kept]
            print(f"已导出轨迹: {len(kept)} 条", flush=True)
            for summary in kept:
                print(
                    f"track={summary.track_id:04d} source={summary.source_video} frames={summary.frame_count} "
                    f"avg_area={summary.avg_bbox_area:.2f} avg_conf={summary.avg_confidence:.4f} "
                    f"range={summary.first_frame_idx}-{summary.last_frame_idx} "
                    f"frames_dir={summary.crop_dir or '-'} clip={summary.clip_path or '-'}",
                    flush=True,
                )
    return summaries


def _fit_to_size(image_bgr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = size
    if image_bgr.size == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    src_h, src_w = image_bgr.shape[:2]
    scale = min(target_w / max(1, src_w), target_h / max(1, src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=resized.dtype)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
    return canvas


def _extract_crop(frame_bgr: np.ndarray, bbox_xyxy: Sequence[float], padding_ratio: float) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = box_w * padding_ratio
    pad_y = box_h * padding_ratio
    left = max(0, int(round(x1 - pad_x)))
    top = max(0, int(round(y1 - pad_y)))
    right = min(width, int(round(x2 + pad_x)))
    bottom = min(height, int(round(y2 + pad_y)))
    if right <= left or bottom <= top:
        return np.empty((0, 0, 3), dtype=frame_bgr.dtype)
    return frame_bgr[top:bottom, left:right]


def _cleanup_appledouble_files(root: Path) -> None:
    sibling = root.parent / f"._{root.name}"
    if sibling.exists():
        sibling.unlink(missing_ok=True)
    if not root.exists():
        return
    for path in root.rglob("._*"):
        if path.is_file() or path.is_symlink():
            path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    process_video(
        input_path=input_path,
        output_path=output_path,
        model_path=args.model,
        det_conf=args.det_conf,
        pose_conf=args.pose_conf,
        device=args.device,
        enable_tracking=args.enable_tracking or args.export_tracks,
        export_tracks=args.export_tracks,
        export_format=args.export_format,
        artifacts_dir=Path(args.artifacts_dir),
        track_iou_threshold=args.track_iou_threshold,
        track_max_age=args.track_max_age,
        min_track_frames=args.min_track_frames,
        min_bbox_size=args.min_bbox_size,
        min_track_confidence=args.min_track_confidence,
    )


if __name__ == "__main__":
    main()
