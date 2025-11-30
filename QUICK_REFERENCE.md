# Quick Reference Guide

## Common Code Snippets

### Initialize Face Recognition
```python
from src.face_recognizer import FaceRecognizer
recognizer = FaceRecognizer(providers=['CPUExecutionProvider'])
```

### Add Reference Images
```python
import glob
for img_path in glob.glob('data/reference/*.jpg'):
    recognizer.add_reference_image('Tom Cruise', img_path)
```

### Detect and Recognize Faces
```python
from src.face_detector import YOLOFaceDetector
from ultralytics import YOLO

# Setup
detector = YOLOFaceDetector('results/yolo_face_tuned6/weights/best.pt')
image = cv2.imread('test.jpg')

# Detect
yolo_results = YOLO('results/yolo_face_tuned6/weights/best.pt').predict(image, verbose=False)

# Recognize
for result in yolo_results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = image[y1:y2, x1:x2]
        matches = recognizer.match(crop, threshold=0.25)
        if matches:
            print(f"Found: {matches[0]['name']}")
```

### Delete Specific Person's Faces
```python
from src.face_detector import delete_specific_person_faces

deleted_image, detections = delete_specific_person_faces(
    image, recognizer, detector,
    target_identity='Tom Cruise',
    deletion_method='blur'
)
```

## Threshold Guidelines

| Scenario | Recommended Threshold |
|----------|----------------------|
| High security | 0.15 |
| Default (balanced) | 0.25 |
| High variance data | 0.35-0.50 |
| Very lenient | 0.60+ |

## Model Selection

| Use Case | Recommended Model |
|----------|------------------|
| Best accuracy | buffalo_l |
| Balanced | buffalo_m |
| Fast inference | buffalo_s |
| Real-time | buffalo_s + YOLOv8n |

## Common Errors & Fixes

| Error | Fix |
|-------|-----|
| "No identities registered" | Add reference images first |
| "High variance in embeddings" | Filter reference images |
| "Model download failed" | Run `python download_insightface_model.py buffalo_l` |
| "CUDAExecutionProvider error" | Use `CPUExecutionProvider` |

## File Paths

| Purpose | Path |
|---------|------|
| Reference images | `data/reference/` |
| Training images | `data/raw/train/` |
| Validation images | `data/raw/val/` |
| Trained models | `results/yolo_face_tuned*/weights/best.pt` |
| YOLO config | `data/face_yolo.yaml` |

