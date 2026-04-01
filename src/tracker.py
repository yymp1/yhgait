from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from .models import Detection


@dataclass(slots=True)
class TrackerConfig:
    iou_threshold: float = 0.3
    max_age: int = 15
    min_hits: int = 2


@dataclass(slots=True)
class TrackState:
    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    hits: int = 1
    misses: int = 0
    age: int = 1
    confirmed: bool = False
    last_frame_index: int = 0


@dataclass(slots=True)
class TrackedDetection:
    detection: Detection
    track_id: int
    confirmed: bool
    hits: int
    age: int


class SimpleIoUTracker:
    """轻量单视频 tracker，只做 IoU 匹配，适合 gait 序列提取。"""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 15,
        min_hits: int = 2,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self._next_track_id = 1
        self._tracks: Dict[int, TrackState] = {}

    @property
    def active_tracks(self) -> list[TrackState]:
        return list(self._tracks.values())

    def reset(self) -> None:
        self._next_track_id = 1
        self._tracks.clear()

    def update(
        self,
        detections: Sequence[Detection],
        frame_index: int,
    ) -> list[TrackedDetection]:
        if not detections and not self._tracks:
            return []

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        candidates: list[tuple[float, int, int]] = []

        for track_id, track in self._tracks.items():
            for det_index, detection in enumerate(detections):
                score = bbox_iou(track.bbox_xyxy, detection.bbox_xyxy)
                if score >= self.iou_threshold:
                    candidates.append((score, track_id, det_index))

        candidates.sort(reverse=True, key=lambda item: item[0])
        assigned: list[TrackedDetection] = []

        for score, track_id, det_index in candidates:
            if track_id in matched_tracks or det_index in matched_detections:
                continue
            track = self._tracks[track_id]
            detection = detections[det_index].with_track_id(track_id)
            self._update_track(track, detection, frame_index)
            matched_tracks.add(track_id)
            matched_detections.add(det_index)
            assigned.append(
                TrackedDetection(
                    detection=detection,
                    track_id=track_id,
                    confirmed=track.confirmed,
                    hits=track.hits,
                    age=track.age,
                )
            )

        for track_id, track in list(self._tracks.items()):
            if track_id in matched_tracks:
                continue
            track.misses += 1
            track.age += 1
            if track.misses > self.max_age:
                self._tracks.pop(track_id, None)

        for det_index, detection in enumerate(detections):
            if det_index in matched_detections:
                continue
            track_id = self._next_track_id
            self._next_track_id += 1
            tracked_detection = detection.with_track_id(track_id)
            self._tracks[track_id] = TrackState(
                track_id=track_id,
                bbox_xyxy=tracked_detection.bbox_xyxy,
                confidence=tracked_detection.confidence,
                hits=1,
                misses=0,
                age=1,
                confirmed=self.min_hits <= 1,
                last_frame_index=frame_index,
            )
            assigned.append(
                TrackedDetection(
                    detection=tracked_detection,
                    track_id=track_id,
                    confirmed=self.min_hits <= 1,
                    hits=1,
                    age=1,
                )
            )

        assigned.sort(key=lambda item: item.track_id)
        return assigned

    def _update_track(self, track: TrackState, detection: Detection, frame_index: int) -> None:
        track.bbox_xyxy = detection.bbox_xyxy
        track.confidence = detection.confidence
        track.hits += 1
        track.misses = 0
        track.age += 1
        track.last_frame_index = frame_index
        track.confirmed = track.hits >= self.min_hits


IoUTracker = SimpleIoUTracker


def bbox_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union
