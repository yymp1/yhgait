from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import mediapipe as mp

from .models import Landmark, PoseSample


POSE_LANDMARK = mp.solutions.pose.PoseLandmark

FACE_POINTS = [
    POSE_LANDMARK.NOSE,
    POSE_LANDMARK.LEFT_EYE,
    POSE_LANDMARK.RIGHT_EYE,
    POSE_LANDMARK.LEFT_EAR,
    POSE_LANDMARK.RIGHT_EAR,
]
TORSO_POINTS = [
    POSE_LANDMARK.LEFT_SHOULDER,
    POSE_LANDMARK.RIGHT_SHOULDER,
    POSE_LANDMARK.LEFT_HIP,
    POSE_LANDMARK.RIGHT_HIP,
]


@dataclass(slots=True)
class OrientationConfig:
    visibility_threshold: float = 0.5
    min_pose_score: float = 0.15
    front_depth_margin: float = 0.12
    back_depth_margin: float = 0.12
    ear_depth_margin: float = 0.05
    side_span_ratio: float = 0.18


class OrientationClassifier:
    def __init__(self, config: OrientationConfig | None = None) -> None:
        self.config = config or OrientationConfig()

    def classify(self, pose: PoseSample | None) -> str:
        if pose is None or not pose.has_pose or not pose.landmarks:
            return "unknown"
        if pose.score < self.config.min_pose_score:
            return "unknown"

        landmark_map = {lm.name: lm for lm in pose.landmarks}
        left_shoulder = landmark_map.get("LEFT_SHOULDER")
        right_shoulder = landmark_map.get("RIGHT_SHOULDER")
        left_hip = landmark_map.get("LEFT_HIP")
        right_hip = landmark_map.get("RIGHT_HIP")
        nose = landmark_map.get("NOSE")
        left_eye = landmark_map.get("LEFT_EYE")
        right_eye = landmark_map.get("RIGHT_EYE")
        left_ear = landmark_map.get("LEFT_EAR")
        right_ear = landmark_map.get("RIGHT_EAR")

        shoulders_visible = self._both_visible(left_shoulder, right_shoulder)
        hips_visible = self._both_visible(left_hip, right_hip)
        torso_visible = self._visible_count(landmark_map, TORSO_POINTS)
        if torso_visible < 2:
            return "unknown"

        body_depth = self._mean_z([left_shoulder, right_shoulder, left_hip, right_hip])
        face_visible = self._visible_count(landmark_map, FACE_POINTS)
        nose_depth_offset = self._depth_offset(nose, body_depth)
        eye_depth_offset = self._depth_offset_from_value(self._mean_z([left_eye, right_eye]), body_depth)
        ear_depth_offset = self._depth_offset_from_value(self._mean_z([left_ear, right_ear]), body_depth)

        front_score = 0.0
        back_score = 0.0
        side_score = self._side_score(pose, landmark_map, shoulders_visible, hips_visible)

        if shoulders_visible:
            front_score += 0.5
            back_score += 0.5
        if hips_visible:
            front_score += 0.25
            back_score += 0.25

        # 当 face 点很少时，背影概率本来就更高。
        if face_visible <= 1 and shoulders_visible and hips_visible:
            back_score += 0.75

        # 关键修正：MediaPipe 在背影上也可能给出高 visibility 的鼻子/眼睛。
        # 直接看“有没有鼻子”不可靠，必须看脸相对躯干的深度。
        if nose_depth_offset is not None:
            if nose_depth_offset <= -self.config.front_depth_margin:
                front_score += 1.5
            elif nose_depth_offset >= self.config.back_depth_margin:
                back_score += 1.5

        if eye_depth_offset is not None:
            if eye_depth_offset <= -self.config.front_depth_margin * 0.75:
                front_score += 0.75
            elif eye_depth_offset >= self.config.back_depth_margin * 0.75:
                back_score += 0.75

        if nose_depth_offset is not None and ear_depth_offset is not None:
            nose_vs_ears = nose_depth_offset - ear_depth_offset
            if nose_vs_ears <= -self.config.ear_depth_margin:
                front_score += 0.75
            elif nose_vs_ears >= self.config.ear_depth_margin:
                back_score += 0.75

        if side_score >= 1.5 and side_score > front_score and side_score > back_score:
            return "side"
        if back_score >= 1.5 and back_score >= front_score + 0.4:
            return "back"
        if front_score >= 1.5 and front_score >= back_score + 0.4:
            return "front"
        if side_score >= 1.0 and max(front_score, back_score) < 1.5:
            return "side"
        return "unknown"

    def _visible_count(self, landmark_map: Dict[str, Landmark], points: Iterable[POSE_LANDMARK]) -> int:
        count = 0
        for point in points:
            if self._is_visible(landmark_map.get(point.name)):
                count += 1
        return count

    def _is_visible(self, landmark: Landmark | None) -> bool:
        return landmark is not None and landmark.visibility >= self.config.visibility_threshold

    def _both_visible(self, left: Landmark | None, right: Landmark | None) -> bool:
        return self._is_visible(left) and self._is_visible(right)

    def _side_score(
        self,
        pose: PoseSample,
        landmark_map: Dict[str, Landmark],
        shoulders_visible: bool,
        hips_visible: bool,
    ) -> float:
        score = 0.0
        if self._one_sided(landmark_map.get("LEFT_SHOULDER"), landmark_map.get("RIGHT_SHOULDER")):
            score += 1.0
        if self._one_sided(landmark_map.get("LEFT_HIP"), landmark_map.get("RIGHT_HIP")):
            score += 1.0
        if self._one_sided(landmark_map.get("LEFT_EAR"), landmark_map.get("RIGHT_EAR")):
            score += 0.5
        if self._one_sided(landmark_map.get("LEFT_EYE"), landmark_map.get("RIGHT_EYE")):
            score += 0.5
        if self._span_ratio(landmark_map.get("LEFT_SHOULDER"), landmark_map.get("RIGHT_SHOULDER"), pose.bbox_xyxy) < self.config.side_span_ratio:
            score += 0.75
        if self._span_ratio(landmark_map.get("LEFT_HIP"), landmark_map.get("RIGHT_HIP"), pose.bbox_xyxy) < self.config.side_span_ratio:
            score += 0.75
        if shoulders_visible and not hips_visible:
            score += 0.25
        if hips_visible and not shoulders_visible:
            score += 0.25
        return score

    def _one_sided(self, left: Landmark | None, right: Landmark | None) -> bool:
        return self._is_visible(left) != self._is_visible(right)

    def _mean_z(self, landmarks: Sequence[Landmark | None]) -> float | None:
        values = [landmark.z for landmark in landmarks if self._is_visible(landmark)]
        if not values:
            return None
        return sum(values) / len(values)

    def _depth_offset(self, landmark: Landmark | None, body_depth: float | None) -> float | None:
        if not self._is_visible(landmark) or body_depth is None:
            return None
        return landmark.z - body_depth

    def _depth_offset_from_value(self, value: float | None, body_depth: float | None) -> float | None:
        if value is None or body_depth is None:
            return None
        return value - body_depth

    def _span_ratio(
        self,
        left: Landmark | None,
        right: Landmark | None,
        bbox_xyxy: Sequence[int],
    ) -> float:
        if not self._both_visible(left, right):
            return 1.0
        bbox_width = max(1.0, float(bbox_xyxy[2] - bbox_xyxy[0]))
        return abs(left.x - right.x) / bbox_width
