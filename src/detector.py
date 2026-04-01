from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from ultralytics import YOLO

from .models import Detection


@dataclass(slots=True)
class DetectorConfig:
    model_path: str = "yolov8n.pt"
    conf: float = 0.35
    iou: float = 0.5
    imgsz: int = 640
    device: Optional[str] = None


class PersonDetector:
    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        self.model = YOLO(self.config.model_path)

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            source=frame_bgr,
            conf=self.config.conf,
            iou=self.config.iou,
            imgsz=self.config.imgsz,
            classes=[0],
            device=self.config.device,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(xyxy))
        clss = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros(len(xyxy))

        detections: List[Detection] = []
        for box, conf, cls_id in zip(xyxy, confs, clss):
            if int(cls_id) != 0:
                continue
            detections.append(
                Detection(
                    bbox_xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name="person",
                )
            )

        detections.sort(key=lambda det: det.confidence, reverse=True)
        return detections


def clamp_bbox(
    bbox_xyxy: tuple[float, float, float, float],
    width: int,
    height: int,
    margin: float = 0.15,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = box_w * margin
    pad_y = box_h * margin
    left = max(0, int(round(x1 - pad_x)))
    top = max(0, int(round(y1 - pad_y)))
    right = min(width - 1, int(round(x2 + pad_x)))
    bottom = min(height - 1, int(round(y2 + pad_y)))
    return left, top, right, bottom
