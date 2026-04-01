from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(slots=True)
class Detection:
    bbox_xyxy: Tuple[float, float, float, float]
    confidence: float
    class_id: int = 0
    class_name: str = "person"
    track_id: int | None = None

    def with_track_id(self, track_id: int | None) -> "Detection":
        return Detection(
            bbox_xyxy=self.bbox_xyxy,
            confidence=self.confidence,
            class_id=self.class_id,
            class_name=self.class_name,
            track_id=track_id,
        )


@dataclass(slots=True)
class Landmark:
    name: str
    x: float
    y: float
    z: float
    visibility: float
    presence: float = 0.0


@dataclass(slots=True)
class PoseSample:
    landmarks: List[Landmark]
    bbox_xyxy: Tuple[int, int, int, int]
    score: float = 0.0
    has_pose: bool = False


@dataclass(slots=True)
class PersonAnnotation:
    detection: Detection
    pose: PoseSample | None
    orientation: str
