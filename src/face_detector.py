"""Face detection utilities leveraging OpenCV cascades and modern detectors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None  # type: ignore


@dataclass
class Detection:
    bbox: List[int]  # [xmin, ymin, xmax, ymax]
    score: float
    label: str = "face"


class HaarFaceDetector:
    """Wrapper around OpenCV Haar cascades (Assignment 4)."""

    def __init__(self, cascade_path: Optional[str | Path] = None):
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(str(cascade_path))

    def detect(self, image: np.ndarray, scaleFactor: float = 1.2, minNeighbors: int = 5) -> List[Detection]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
        detections = []
        for (x, y, w, h) in faces:
            detections.append(Detection([int(x), int(y), int(x + w), int(y + h)], 1.0))
        return detections


class YOLOFaceDetector:
    """YOLOv8/YOLOv10 based detector for stronger performance."""

    def __init__(self, model_name: str = "yolov8n.pt"):
        if YOLO is None:
            raise ImportError("ultralytics is required for YOLOFaceDetector")
        self.model = YOLO(model_name)

    def detect(self, image: np.ndarray, conf: float = 0.25) -> List[Detection]:
        results = self.model.predict(image, conf=conf, verbose=False)
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = float(box.conf[0].item())
                label_idx = int(box.cls[0].item())
                label = self.model.names.get(label_idx, "face")
                detections.append(Detection([int(x1), int(y1), int(x2), int(y2)], score, label))
        return detections


def visualize_detections(image: np.ndarray, detections: List[Detection]) -> np.ndarray:
    vis_img = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{det.label}:{det.score:.2f}"
        cv2.putText(vis_img, text, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return vis_img

__all__ = [
    "Detection",
    "HaarFaceDetector",
    "YOLOFaceDetector",
    "visualize_detections",
]
