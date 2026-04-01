from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .models import Detection


@dataclass(slots=True)
class TrackExportConfig:
    output_root: Path = Path("artifacts")
    export_frames: bool = True
    export_clips: bool = False
    min_track_length: int = 20
    min_bbox_width: int = 40
    min_bbox_height: int = 40
    min_confidence: float = 0.25
    track_padding: float = 0.15
    clip_fps: float = 25.0
    clip_size: tuple[int, int] | None = None


@dataclass(slots=True)
class TrackSummary:
    track_id: int
    accepted_frames: int
    source_video: str
    frame_dir: Path | None
    clip_path: Path | None
    first_frame_index: int | None = None
    last_frame_index: int | None = None
    kept: bool = False


@dataclass(slots=True)
class _TrackWriter:
    track_id: int
    base_dir: Path
    export_frames: bool
    export_clips: bool
    clip_fps: float
    clip_size: tuple[int, int] | None
    source_video: str
    frame_count: int = 0
    first_frame_index: int | None = None
    last_frame_index: int | None = None
    frame_dir: Path | None = None
    clip_path: Path | None = None
    clip_writer: cv2.VideoWriter | None = None
    writer_size: tuple[int, int] | None = None

    def append(self, crop_bgr: np.ndarray, frame_index: int) -> None:
        if self.first_frame_index is None:
            self.first_frame_index = frame_index
        self.last_frame_index = frame_index
        self.frame_count += 1

        if self.export_frames:
            self._ensure_frame_dir()
            frame_path = self.frame_dir / f"frame_{frame_index:06d}.jpg"
            cv2.imwrite(str(frame_path), crop_bgr)

        if self.export_clips:
            self._ensure_clip_writer(crop_bgr)
            clip_frame = crop_bgr
            if self.writer_size is not None and (crop_bgr.shape[1], crop_bgr.shape[0]) != self.writer_size:
                clip_frame = fit_to_size(crop_bgr, self.writer_size)
            self.clip_writer.write(clip_frame)

    def finalize(self, min_track_length: int) -> TrackSummary:
        self._close_writer()
        kept = self.frame_count >= min_track_length
        if not kept:
            if self.frame_dir and self.frame_dir.exists():
                shutil.rmtree(self.frame_dir, ignore_errors=True)
            if self.clip_path and self.clip_path.exists():
                self.clip_path.unlink()
            return TrackSummary(
                track_id=self.track_id,
                accepted_frames=self.frame_count,
                source_video=self.source_video,
                frame_dir=None,
                clip_path=None,
                first_frame_index=self.first_frame_index,
                last_frame_index=self.last_frame_index,
                kept=False,
            )

        self._write_metadata()
        return TrackSummary(
            track_id=self.track_id,
            accepted_frames=self.frame_count,
            source_video=self.source_video,
            frame_dir=self.frame_dir,
            clip_path=self.clip_path,
            first_frame_index=self.first_frame_index,
            last_frame_index=self.last_frame_index,
            kept=True,
        )

    def _ensure_frame_dir(self) -> None:
        if self.frame_dir is None:
            self.frame_dir = self.base_dir / "tracks" / f"{self.track_id:04d}"
            self.frame_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_clip_writer(self, crop_bgr: np.ndarray) -> None:
        if self.clip_writer is not None:
            return
        clips_dir = self.base_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        self.clip_path = clips_dir / f"track_{self.track_id:04d}.mp4"
        self.writer_size = self.clip_size or (crop_bgr.shape[1], crop_bgr.shape[0])
        self.clip_writer = cv2.VideoWriter(
            str(self.clip_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.clip_fps,
            self.writer_size,
        )
        if not self.clip_writer.isOpened():
            raise RuntimeError(f"无法创建轨迹视频: {self.clip_path}")

    def _close_writer(self) -> None:
        if self.clip_writer is not None:
            self.clip_writer.release()
            self.clip_writer = None

    def _write_metadata(self) -> None:
        if self.frame_dir is None:
            return
        metadata = {
            "track_id": self.track_id,
            "source_video": self.source_video,
            "accepted_frames": self.frame_count,
            "first_frame_index": self.first_frame_index,
            "last_frame_index": self.last_frame_index,
            "frame_dir": str(self.frame_dir),
            "clip_path": str(self.clip_path) if self.clip_path else None,
        }
        (self.frame_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


class TrackExporter:
    def __init__(self, config: TrackExportConfig | None = None, *, source_video: str = "") -> None:
        self.config = config or TrackExportConfig()
        self.source_video = source_video
        self.output_root = Path(self.config.output_root)
        _prepare_output_root(self.output_root)
        self._writers: dict[int, _TrackWriter] = {}

    def observe_frame(
        self,
        frame_index: int,
        frame_bgr: np.ndarray,
        detections: Sequence[Detection],
    ) -> None:
        for detection in detections:
            if detection.track_id is None:
                continue
            if not _should_export(detection, self.config):
                continue
            crop_bgr = _extract_crop(frame_bgr, detection.bbox_xyxy, self.config.track_padding)
            if crop_bgr.size == 0:
                continue
            writer = self._writers.get(detection.track_id)
            if writer is None:
                writer = _TrackWriter(
                    track_id=detection.track_id,
                    base_dir=self.output_root,
                    export_frames=self.config.export_frames,
                    export_clips=self.config.export_clips,
                    clip_fps=self.config.clip_fps,
                    clip_size=self.config.clip_size,
                    source_video=self.source_video,
                )
                self._writers[detection.track_id] = writer
            writer.append(crop_bgr, frame_index)

    def finalize(self) -> list[TrackSummary]:
        summaries: list[TrackSummary] = []
        for track_id in sorted(self._writers):
            summaries.append(self._writers[track_id].finalize(self.config.min_track_length))
        kept_summaries = [summary for summary in summaries if summary.kept]
        _write_manifest(self.output_root / "manifest.tsv", kept_summaries)
        return kept_summaries


def _prepare_output_root(output_root: Path) -> None:
    tracks_dir = output_root / "tracks"
    clips_dir = output_root / "clips"
    if tracks_dir.exists():
        shutil.rmtree(tracks_dir, ignore_errors=True)
    if clips_dir.exists():
        shutil.rmtree(clips_dir, ignore_errors=True)
    tracks_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)


def _should_export(detection: Detection, config: TrackExportConfig) -> bool:
    x1, y1, x2, y2 = detection.bbox_xyxy
    width = x2 - x1
    height = y2 - y1
    return (
        detection.confidence >= config.min_confidence
        and width >= config.min_bbox_width
        and height >= config.min_bbox_height
    )


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


def fit_to_size(image_bgr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
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


def _write_manifest(path: Path, summaries: Sequence[TrackSummary]) -> None:
    lines = ["track_id\tframes\tfirst_frame\tlast_frame\tframe_dir\tclip_path\tsource_video"]
    for summary in summaries:
        lines.append(
            "\t".join(
                [
                    f"{summary.track_id:04d}",
                    str(summary.accepted_frames),
                    str(summary.first_frame_index if summary.first_frame_index is not None else ""),
                    str(summary.last_frame_index if summary.last_frame_index is not None else ""),
                    str(summary.frame_dir) if summary.frame_dir else "",
                    str(summary.clip_path) if summary.clip_path else "",
                    summary.source_video,
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


ExportConfig = TrackExportConfig
TrackExportSummary = TrackSummary
