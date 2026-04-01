from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import mediapipe as mp
import numpy as np

from .detector import clamp_bbox
from .models import Landmark, PoseSample


_POSE_LANDMARKS = mp.solutions.pose.PoseLandmark
_POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS


@dataclass(slots=True)
class PoseConfig:
    min_detection_confidence: float = 0.4
    model_complexity: int = 1
    visibility_threshold: float = 0.5
    crop_margin: float = 0.15


class PoseEstimator:
    def __init__(self, config: PoseConfig | None = None) -> None:
        self.config = config or PoseConfig()
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=self.config.model_complexity,
            smooth_landmarks=False,
            enable_segmentation=False,
            min_detection_confidence=self.config.min_detection_confidence,
        )

    def estimate(
        self,
        frame_bgr: np.ndarray,
        bbox_xyxy: tuple[float, float, float, float],
    ) -> PoseSample | None:
        frame_h, frame_w = frame_bgr.shape[:2]
        left, top, right, bottom = clamp_bbox(
            bbox_xyxy,
            width=frame_w,
            height=frame_h,
            margin=self.config.crop_margin,
        )
        if right <= left or bottom <= top:
            return None

        crop = frame_bgr[top : bottom + 1, left : right + 1]
        if crop.size == 0:
            return None

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = self.pose.process(crop_rgb)
        if not result.pose_landmarks:
            return PoseSample(landmarks=[], bbox_xyxy=(left, top, right, bottom), score=0.0, has_pose=False)

        crop_h, crop_w = crop.shape[:2]
        landmarks: List[Landmark] = []
        visible_scores: List[float] = []
        for index, lm in enumerate(result.pose_landmarks.landmark):
            name = _POSE_LANDMARKS(index).name
            x_abs = left + lm.x * crop_w
            y_abs = top + lm.y * crop_h
            visibility = float(getattr(lm, "visibility", 0.0))
            presence = float(getattr(lm, "presence", 0.0))
            landmarks.append(
                Landmark(
                    name=name,
                    x=float(x_abs),
                    y=float(y_abs),
                    z=float(lm.z),
                    visibility=visibility,
                    presence=presence,
                )
            )
            if visibility >= self.config.visibility_threshold:
                visible_scores.append(visibility)

        score = float(sum(visible_scores) / len(visible_scores)) if visible_scores else 0.0
        return PoseSample(
            landmarks=landmarks,
            bbox_xyxy=(left, top, right, bottom),
            score=score,
            has_pose=True,
        )

    def close(self) -> None:
        self.pose.close()
