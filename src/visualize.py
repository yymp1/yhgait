from __future__ import annotations

from typing import List

import cv2
import mediapipe as mp
import numpy as np

from .models import PersonAnnotation, PoseSample


POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS
FACE_LANDMARK_NAMES = {
    "NOSE",
    "LEFT_EYE_INNER",
    "LEFT_EYE",
    "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER",
    "RIGHT_EYE",
    "RIGHT_EYE_OUTER",
    "LEFT_EAR",
    "RIGHT_EAR",
    "MOUTH_LEFT",
    "MOUTH_RIGHT",
}

LABEL_COLORS = {
    "front": (46, 204, 113),
    "back": (52, 152, 219),
    "side": (241, 196, 15),
    "unknown": (189, 195, 199),
}


def annotate_frame(frame_bgr: np.ndarray, annotations: List[PersonAnnotation]) -> np.ndarray:
    canvas = frame_bgr.copy()
    for annotation in annotations:
        canvas = draw_person(canvas, annotation)
    return canvas


def draw_person(frame_bgr: np.ndarray, annotation: PersonAnnotation) -> np.ndarray:
    color = LABEL_COLORS.get(annotation.orientation, LABEL_COLORS["unknown"])
    x1, y1, x2, y2 = [int(round(v)) for v in annotation.detection.bbox_xyxy]
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
    text = _build_label(annotation)
    draw_label(frame_bgr, text, (x1, max(0, y1 - 8)), color)

    if annotation.pose and annotation.pose.has_pose and annotation.pose.landmarks:
        draw_skeleton(frame_bgr, annotation.pose, color)
    return frame_bgr


def draw_label(frame_bgr: np.ndarray, text: str, anchor: tuple[int, int], color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = anchor
    top_left = (x, max(0, y - text_h - baseline - 6))
    bottom_right = (x + text_w + 8, max(0, y + 2))
    cv2.rectangle(frame_bgr, top_left, bottom_right, color, thickness=-1)
    text_org = (x + 4, max(0, y - 4))
    cv2.putText(frame_bgr, text, text_org, font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def _build_label(annotation: PersonAnnotation) -> str:
    parts: list[str] = []
    if annotation.detection.track_id is not None:
        parts.append(f"track={annotation.detection.track_id}")
    parts.append(annotation.orientation)
    parts.append(f"{annotation.detection.confidence:.2f}")
    return " ".join(parts)


def draw_skeleton(frame_bgr: np.ndarray, pose: PoseSample, color: tuple[int, int, int]) -> None:
    landmark_map = {lm.name: lm for lm in pose.landmarks}
    for left_idx, right_idx in POSE_CONNECTIONS:
        left_name = mp.solutions.pose.PoseLandmark(left_idx).name
        right_name = mp.solutions.pose.PoseLandmark(right_idx).name
        if left_name in FACE_LANDMARK_NAMES or right_name in FACE_LANDMARK_NAMES:
            continue
        left = landmark_map.get(left_name)
        right = landmark_map.get(right_name)
        if left is None or right is None:
            continue
        if left.visibility < 0.2 or right.visibility < 0.2:
            continue
        p1 = (int(round(left.x)), int(round(left.y)))
        p2 = (int(round(right.x)), int(round(right.y)))
        cv2.line(frame_bgr, p1, p2, color, 2, cv2.LINE_AA)

    for landmark in pose.landmarks:
        if landmark.visibility < 0.2:
            continue
        if landmark.name in FACE_LANDMARK_NAMES:
            continue
        center = (int(round(landmark.x)), int(round(landmark.y)))
        cv2.circle(frame_bgr, center, 3, color, -1, cv2.LINE_AA)
