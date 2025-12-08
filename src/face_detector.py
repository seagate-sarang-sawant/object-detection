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

    def __init__(self, model_ref = None, model_name: str = "yolov8n.pt"):
        if YOLO is None:
            raise ImportError("ultralytics is required for YOLOFaceDetector")
        if model_ref is not None:
            self.model = model_ref 
        else:   
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
                # Accept all detections (face detection model should only detect faces)
                # But also accept "person" class in case model detects that
                if label.lower() in ["face", "person", "0"] or label_idx == 0:
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


def delete_faces(
    image: np.ndarray, 
    detections: List[Detection],
    method: str = "blur",
    blur_strength: int = 50,
    pixelate_size: int = 10
) -> np.ndarray:
    """
    Delete/blur detected faces from image.
    
    Args:
        image: Input image (BGR format)
        detections: List of face detections to delete
        method: Deletion method - "blur", "pixelate", "black", or "inpaint"
        blur_strength: Blur kernel size (must be odd, for blur method)
        pixelate_size: Size for pixelation (for pixelate method)
    
    Returns:
        Image with faces deleted/blurred
    """
    result = image.copy()
    
    # Ensure blur_strength is odd
    if blur_strength % 2 == 0:
        blur_strength += 1
    
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        
        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            continue  # Skip invalid bounding boxes
        
        if method == "blur":
            # Gaussian blur
            face_roi = result[y1:y2, x1:x2]
            if face_roi.size > 0:
                blurred = cv2.GaussianBlur(face_roi, (blur_strength, blur_strength), 0)
                result[y1:y2, x1:x2] = blurred
                
        elif method == "pixelate":
            # Pixelate
            face_roi = result[y1:y2, x1:x2]
            if face_roi.size > 0:
                # Downscale
                small = cv2.resize(face_roi, (pixelate_size, pixelate_size), interpolation=cv2.INTER_LINEAR)
                # Upscale back
                pixelated = cv2.resize(small, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
                result[y1:y2, x1:x2] = pixelated
                
        elif method == "black":
            # Black rectangle
            result[y1:y2, x1:x2] = 0
            
        elif method == "inpaint":
            # Inpainting (requires mask)
            mask = np.zeros(result.shape[:2], dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255
            result = cv2.inpaint(result, mask, 3, cv2.INPAINT_TELEA)
            
        else:
            raise ValueError(f"Unknown deletion method: {method}. Use 'blur', 'pixelate', 'black', or 'inpaint'")
    
    return result


def delete_specific_person_faces(
    image: np.ndarray,
    face_recognizer,  # FaceRecognizer instance
    yolo_detector,   # YOLOFaceDetector instance
    target_identity: str,
    deletion_method: str = "blur",
    recognition_threshold: float = 0.25,
    blur_strength: int = 50
) -> tuple[np.ndarray, List[Detection]]:
    """
    Delete faces of a specific person from an image.
    
    Args:
        image: Input image (BGR format)
        face_recognizer: FaceRecognizer instance with registered identities
        yolo_detector: YOLOFaceDetector instance for face detection
        target_identity: Name of the identity whose faces should be deleted
        deletion_method: Method to use for deletion ("blur", "pixelate", "black", "inpaint")
        recognition_threshold: Threshold for face recognition matching
        blur_strength: Blur kernel size (for blur method)
    
    Returns:
        Tuple of (processed_image, deleted_detections)
    """
    # Detect all faces
    all_detections = yolo_detector.detect(image)
    
    if not all_detections:
        return image, []
    
    # Filter detections to only target identity
    target_detections = []
    
    for det in all_detections:
        x1, y1, x2, y2 = det.bbox
        # Crop face region
        crop = image[y1:y2, x1:x2]
        
        # Match against registered identities
        matches = face_recognizer.match(crop, threshold=recognition_threshold)
        
        # Check if this face matches the target identity
        for match in matches:
            if match["name"] == target_identity:
                target_detections.append(det)
                break
    
    # Delete only the target identity's faces
    if target_detections:
        result = delete_faces(image, target_detections, method=deletion_method, blur_strength=blur_strength)
        return result, target_detections
    
    return image, []


__all__ = [
    "Detection",
    "HaarFaceDetector",
    "YOLOFaceDetector",
    "visualize_detections",
    "delete_faces",
    "delete_specific_person_faces",
]
