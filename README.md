# Face Detection Pipeline

End-to-end pipeline to detect and identify a specific person's face, reusing Assignment 3 and Assignment 4 components.

## Dataset
- **WIDER FACE** via TFDS for general face detection training.
- Custom Pascal VOC style folder (images + annotations) for the target person.
- Reference images placed in `data/reference/` for identity embeddings.

## Structure
```
face_detection_pipeline/
├── data/
├── models/
├── notebooks/
├── results/
├── src/
│   ├── data_loader.py
│   ├── dataset_wrapper.py
│   ├── face_detector.py
│   ├── face_recognizer.py
│   └── preprocessing.py
├── requirements.txt
└── README.md
```

## Steps
1. Install dependencies: `pip install -r requirements.txt`
2. Download WIDER FACE automatically via TFDS (handled in `data_loader.py`).
3. Place custom face dataset (Pascal VOC format) under `data/raw` and annotations under `data/annotations`.
4. Add reference images of the specific person into `data/reference` and register them via `FaceRecognizer`.
5. Run the notebook in `notebooks/` (to be created) to train detectors and run inference.

## Components
- **Data Loader**: TFDS ingestion, custom Pascal VOC loader.
- **Dataset Wrapper**: Mask R-CNN dataset adapter (from Assignment 3).
- **Face Detector**: Haar cascades (Assignment 4) + YOLO.
- **Face Recognizer**: InsightFace embeddings for identity matching.
- **Preprocessing**: Resizing and augmentations (Assignment 3 style).

# object-detection
