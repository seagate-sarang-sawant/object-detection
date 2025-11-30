# Face Detection & Recognition Pipeline - Technical Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Installation & Setup](#installation--setup)
4. [System Requirements](#system-requirements)
5. [Project Structure](#project-structure)
6. [Core Components](#core-components)
7. [API Reference](#api-reference)
8. [Configuration](#configuration)
9. [Usage Guide](#usage-guide)
10. [Algorithms & Models](#algorithms--models)
11. [Performance Considerations](#performance-considerations)
12. [Troubleshooting](#troubleshooting)

---

## Project Overview

The Face Detection & Recognition Pipeline is an end-to-end computer vision system designed to:
- **Detect faces** in images using multiple detection methods (Haar Cascades, YOLOv8)
- **Recognize specific individuals** using deep learning embeddings (InsightFace)
- **Delete/blur faces** of specific persons from images
- **Train custom face detection models** using YOLOv8 fine-tuning

### Key Features
- Multi-model face detection (Haar Cascades, YOLOv8)
- Deep face recognition using InsightFace embeddings
- Support for multiple reference images per identity
- Face deletion/blurring with multiple methods
- Custom model training on WIDER FACE and custom datasets
- Comprehensive validation and diagnostics

### Use Cases
- Privacy protection (face blurring/deletion)
- Access control systems
- Content moderation
- Media organization and tagging
- Security applications

---

## Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Image/Video                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Face Detection Module                           │
│  ┌──────────────┐         ┌──────────────┐                  │
│  │ Haar Cascade │    OR   │   YOLOv8     │                  │
│  │  Detector    │         │  Detector    │                  │
│  └──────────────┘         └──────────────┘                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│            Face Recognition Module                           │
│  ┌──────────────────────────────────────────┐               │
│  │      InsightFace Embedding Extraction     │               │
│  │  (buffalo_l, buffalo_m, or buffalo_s)    │               │
│  └──────────────────┬───────────────────────┘               │
│                     │                                         │
│  ┌──────────────────▼───────────────────────┐               │
│  │    Identity Matching (Cosine Distance)    │               │
│  │  - Multiple embeddings per identity      │               │
│  │  - Best match selection                  │               │
│  │  - Confidence scoring                    │               │
│  └──────────────────┬───────────────────────┘               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Output Processing                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ Visualization│  │ Face Deletion│  │   Results   │       │
│  │   & Labels   │  │   / Blurring │  │  Export     │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Training Phase**:
   - WIDER FACE dataset → Data Loader → YOLOv8 Training → Trained Model
   - Custom dataset → Label Generation → YOLO Training → Fine-tuned Model

2. **Inference Phase**:
   - Input Image → Face Detection → Face Cropping → Embedding Extraction → Identity Matching → Results

3. **Face Deletion Phase**:
   - Input Image → Face Detection → Face Recognition → Filter Target Identity → Apply Deletion Method → Output Image

---

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip package manager
- 8GB+ RAM recommended
- GPU optional but recommended for training

### Step 1: Clone/Setup Project
```bash
cd face_detection_pipeline
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download InsightFace Models
Models are downloaded automatically on first use. If download fails:
```bash
python download_insightface_model.py buffalo_l
```

### Step 5: Prepare Data Directories
```bash
mkdir -p data/{raw,annotations,reference,processed}
```

---

## System Requirements

### Minimum Requirements
- **CPU**: Multi-core processor (Intel i5 or equivalent)
- **RAM**: 8GB
- **Storage**: 10GB free space
- **OS**: macOS, Linux, or Windows 10+

### Recommended Requirements
- **CPU**: Intel i7/AMD Ryzen 7 or better
- **RAM**: 16GB+
- **GPU**: NVIDIA GPU with CUDA support (for training)
- **Storage**: 50GB+ SSD

### Software Dependencies
- Python 3.10+
- TensorFlow 2.13+
- PyTorch 2.0+
- OpenCV 4.8+
- ONNX Runtime (for InsightFace)

---

## Project Structure

```
face_detection_pipeline/
├── data/
│   ├── raw/                    # Raw images (train/val splits)
│   │   ├── train/              # Training images and labels
│   │   └── val/                # Validation images and labels
│   ├── annotations/            # Pascal VOC XML annotations
│   ├── reference/               # Reference images for face recognition
│   ├── processed/               # Processed/preprocessed data
│   ├── downloads/              # Downloaded datasets
│   └── face_yolo.yaml          # YOLO dataset configuration
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Dataset loading utilities
│   ├── dataset_wrapper.py       # Mask R-CNN dataset adapter
│   ├── face_detector.py         # Face detection (Haar, YOLO)
│   ├── face_recognizer.py       # Face recognition (InsightFace)
│   └── preprocessing.py        # Image preprocessing functions
├── notebooks/
│   └── face_detection_pipeline.ipynb  # Main notebook
├── results/                     # Training results and models
│   └── yolo_face_tuned*/       # Trained YOLO models
├── models/                      # Pre-trained models
├── generate_yolo_labels.py      # Label generation script
├── download_insightface_model.py # Model download helper
├── requirements.txt             # Python dependencies
├── README.md                    # Project overview
├── TECHNICAL_DOCUMENTATION.md   # This file
├── TROUBLESHOOTING.md          # Troubleshooting guide
└── CHANGES_SUMMARY.md          # Change log
```

---

## Core Components

### 1. Data Loader (`src/data_loader.py`)

**Purpose**: Load and preprocess datasets for training and inference.

**Key Functions**:
- `load_wider_face()`: Load WIDER FACE dataset via TFDS or direct file access
- `load_custom_pascal_voc()`: Load custom datasets in Pascal VOC format
- `save_dataset_statistics()`: Generate dataset statistics

**Features**:
- Automatic fallback to direct file loading if TFDS fails
- Support for multiple image formats
- Automatic preprocessing and normalization
- Batch processing with TensorFlow Data API

### 2. Face Detector (`src/face_detector.py`)

**Purpose**: Detect faces in images using multiple methods.

**Classes**:
- `HaarFaceDetector`: OpenCV Haar Cascade face detection
- `YOLOFaceDetector`: YOLOv8-based face detection

**Functions**:
- `visualize_detections()`: Draw bounding boxes on images
- `delete_faces()`: Delete/blur faces from images
- `delete_specific_person_faces()`: Delete only specific person's faces

**Deletion Methods**:
- `blur`: Gaussian blur
- `pixelate`: Pixelation effect
- `black`: Black rectangle overlay
- `inpaint`: Inpainting-based removal

### 3. Face Recognizer (`src/face_recognizer.py`)

**Purpose**: Recognize specific individuals using deep learning embeddings.

**Classes**:
- `IdentityProfile`: Stores multiple embeddings per identity
- `FaceRecognizer`: Main recognition engine

**Key Features**:
- Multiple reference images per identity
- Best-match selection across all embeddings
- Confidence level scoring
- Reference image validation

**Models Supported**:
- `buffalo_l`: Large model (best accuracy)
- `buffalo_m`: Medium model (balanced)
- `buffalo_s`: Small model (fastest)

### 4. Dataset Wrapper (`src/dataset_wrapper.py`)

**Purpose**: Adapter for Mask R-CNN compatibility (from Assignment 3).

**Classes**:
- `FaceDataset`: Mask R-CNN compatible dataset wrapper

### 5. Preprocessing (`src/preprocessing.py`)

**Purpose**: Image preprocessing utilities.

**Functions**:
- `resize_and_normalize()`: Resize and normalize images
- `augment_image()`: Apply data augmentation

---

## API Reference

### FaceRecognizer

#### `FaceRecognizer.__init__(providers=None, model_name=None)`

Initialize the face recognition system.

**Parameters**:
- `providers` (List[str], optional): ONNX execution providers. Default: `["CPUExecutionProvider"]`
- `model_name` (str, optional): Specific model to use. Options: `"buffalo_l"`, `"buffalo_m"`, `"buffalo_s"`. Default: Auto-select

**Returns**: `FaceRecognizer` instance

**Example**:
```python
recognizer = FaceRecognizer(providers=['CPUExecutionProvider'])
```

#### `add_reference_image(name: str, image_path: str | Path) -> None`

Add a reference image for an identity. Multiple images can be added for the same identity.

**Parameters**:
- `name` (str): Identity name (e.g., "Tom Cruise")
- `image_path` (str | Path): Path to reference image

**Raises**:
- `FileNotFoundError`: If image file doesn't exist
- `ValueError`: If no face detected in image

**Example**:
```python
recognizer.add_reference_image('Tom Cruise', 'data/reference/tom1.jpg')
recognizer.add_reference_image('Tom Cruise', 'data/reference/tom2.jpg')
```

#### `match(image: np.ndarray, threshold=0.25, confidence_levels=True, return_all_scores=False) -> List[Dict]`

Match faces in image against registered identities.

**Parameters**:
- `image` (np.ndarray): Input image in BGR format
- `threshold` (float): Maximum distance for a match (lower = stricter). Default: 0.25
- `confidence_levels` (bool): Add confidence levels to results. Default: True
- `return_all_scores` (bool): Return best match even if above threshold. Default: False

**Returns**: List of match dictionaries with:
- `name` (str): Matched identity name
- `score` (float): Distance score (lower is better)
- `bbox` (List[float]): Bounding box coordinates
- `confidence` (str, optional): "high", "medium", or "low"

**Example**:
```python
matches = recognizer.match(image, threshold=0.25)
for match in matches:
    print(f"Found {match['name']} with confidence {match['confidence']}")
```

#### `validate_reference_images(name: str) -> Dict[str, Any]`

Validate consistency of reference images for an identity.

**Parameters**:
- `name` (str): Identity name to validate

**Returns**: Dictionary with validation results:
- `valid` (bool): Whether validation passed
- `num_references` (int): Number of reference images
- `avg_distance` (float): Average embedding distance
- `max_distance` (float): Maximum embedding distance
- `std_distance` (float): Standard deviation of distances

**Example**:
```python
validation = recognizer.validate_reference_images('Tom Cruise')
if not validation['valid']:
    print(f"Warning: {validation['reason']}")
```

### Face Detector

#### `YOLOFaceDetector.__init__(model_name="yolov8n.pt")`

Initialize YOLO face detector.

**Parameters**:
- `model_name` (str): Path to YOLO model file

**Example**:
```python
detector = YOLOFaceDetector('results/yolo_face_tuned6/weights/best.pt')
```

#### `detect(image: np.ndarray, conf=0.25) -> List[Detection]`

Detect faces in image.

**Parameters**:
- `image` (np.ndarray): Input image in BGR format
- `conf` (float): Confidence threshold. Default: 0.25

**Returns**: List of `Detection` objects with:
- `bbox` (List[int]): [xmin, ymin, xmax, ymax]
- `score` (float): Detection confidence
- `label` (str): Detection label

**Example**:
```python
detections = detector.detect(image, conf=0.25)
for det in detections:
    x1, y1, x2, y2 = det.bbox
    print(f"Face at ({x1}, {y1}) to ({x2}, {y2}) with confidence {det.score}")
```

#### `delete_specific_person_faces(image, face_recognizer, yolo_detector, target_identity, deletion_method="blur", recognition_threshold=0.25, blur_strength=50) -> Tuple[np.ndarray, List[Detection]]`

Delete faces of a specific person from an image.

**Parameters**:
- `image` (np.ndarray): Input image in BGR format
- `face_recognizer` (FaceRecognizer): FaceRecognizer instance
- `yolo_detector` (YOLOFaceDetector): YOLOFaceDetector instance
- `target_identity` (str): Name of identity whose faces to delete
- `deletion_method` (str): "blur", "pixelate", "black", or "inpaint"
- `recognition_threshold` (float): Threshold for face recognition
- `blur_strength` (int): Blur kernel size (for blur method)

**Returns**: Tuple of (processed_image, deleted_detections)

**Example**:
```python
deleted_image, detections = delete_specific_person_faces(
    image,
    face_recognizer,
    yolo_detector,
    target_identity='Tom Cruise',
    deletion_method='blur'
)
```

---

## Configuration

### YOLO Dataset Configuration (`data/face_yolo.yaml`)

```yaml
train: raw/train/
val: raw/val/
yaml_version: 1.0
names:
  0: face
```

### Pipeline Configuration (Notebook)

```python
@dataclass
class PipelineConfig:
    project_root: Path = Path('..').resolve()
    data_dir: Path = project_root / 'data'
    raw_dir: Path = data_dir / 'raw'
    annotations_dir: Path = data_dir / 'annotations'
    reference_dir: Path = data_dir / 'reference'
    processed_dir: Path = data_dir / 'processed'
    wider_split: str = 'train'
    batch_size: int = 4
    image_size: tuple[int, int] = (512, 512)
    yolo_model: str = 'yolov8n.pt'
    yolo_epochs: int = 25
    yolo_data_cfg: Path = data_dir / 'face_yolo.yaml'
```

### Recognition Threshold Guidelines

| Threshold | Use Case | False Positives | False Negatives |
|-----------|----------|----------------|-----------------|
| 0.15 | Very strict (high security) | Very low | Higher |
| 0.25 | Default (balanced) | Low | Low |
| 0.35 | Moderate | Moderate | Very low |
| 0.50+ | Lenient (high variance data) | Higher | Very low |

**Recommendation**: Start with 0.25, adjust based on validation results.

---

## Usage Guide

### Basic Face Detection

```python
from src.face_detector import YOLOFaceDetector
import cv2

# Initialize detector
detector = YOLOFaceDetector('results/yolo_face_tuned6/weights/best.pt')

# Load image
image = cv2.imread('test_image.jpg')

# Detect faces
detections = detector.detect(image, conf=0.25)

# Visualize
from src.face_detector import visualize_detections
visualized = visualize_detections(image, detections)
cv2.imwrite('output.jpg', visualized)
```

### Face Recognition

```python
from src.face_recognizer import FaceRecognizer
import glob

# Initialize recognizer
recognizer = FaceRecognizer(providers=['CPUExecutionProvider'])

# Add reference images
reference_images = glob.glob('data/reference/*.jpg')
for img_path in reference_images:
    recognizer.add_reference_image('Tom Cruise', img_path)

# Validate
validation = recognizer.validate_reference_images('Tom Cruise')
print(f"Valid: {validation['valid']}, References: {validation['num_references']}")

# Match faces
import cv2
image = cv2.imread('test_image.jpg')
matches = recognizer.match(image, threshold=0.25)

for match in matches:
    print(f"Found {match['name']} with score {match['score']:.3f}")
```

### Face Deletion

```python
from src.face_detector import YOLOFaceDetector, delete_specific_person_faces
from src.face_recognizer import FaceRecognizer

# Setup
recognizer = FaceRecognizer()
# ... add reference images ...

detector = YOLOFaceDetector('results/yolo_face_tuned6/weights/best.pt')

# Delete specific person's faces
image = cv2.imread('input.jpg')
deleted_image, detections = delete_specific_person_faces(
    image,
    recognizer,
    detector,
    target_identity='Tom Cruise',
    deletion_method='blur',  # or 'pixelate', 'black', 'inpaint'
    blur_strength=51
)

cv2.imwrite('output.jpg', deleted_image)
```

### Training Custom YOLO Model

```python
from ultralytics import YOLO

# Load base model
model = YOLO('yolov8n.pt')

# Train on custom dataset
results = model.train(
    data='data/face_yolo.yaml',
    epochs=25,
    imgsz=512,
    project='results',
    name='yolo_face_tuned'
)
```

---

## Algorithms & Models

### Face Detection

#### Haar Cascade Classifier
- **Algorithm**: Viola-Jones algorithm
- **Type**: Traditional machine learning
- **Speed**: Fast (~30 FPS on CPU)
- **Accuracy**: Moderate
- **Use Case**: Real-time applications, quick prototyping

#### YOLOv8
- **Algorithm**: You Only Look Once (YOLO) v8
- **Type**: Deep learning (CNN)
- **Architecture**: CSPDarknet backbone with PANet neck
- **Speed**: Fast (~60 FPS on GPU)
- **Accuracy**: High (mAP50: ~0.92 on face detection)
- **Use Case**: Production applications, high accuracy requirements

### Face Recognition

#### InsightFace
- **Framework**: ArcFace (Additive Angular Margin Loss)
- **Model Variants**:
  - `buffalo_l`: Large model, 512-dim embeddings, best accuracy
  - `buffalo_m`: Medium model, balanced speed/accuracy
  - `buffalo_s`: Small model, fastest inference
- **Embedding Dimension**: 512
- **Distance Metric**: Euclidean distance (L2 norm)
- **Matching Strategy**: Best match across multiple embeddings

#### Recognition Pipeline
1. **Face Detection**: Detect and crop face regions
2. **Alignment**: Face alignment (handled by InsightFace)
3. **Embedding Extraction**: Generate 512-dimensional feature vector
4. **Normalization**: L2 normalization of embeddings
5. **Matching**: Compute Euclidean distance to reference embeddings
6. **Thresholding**: Apply distance threshold for match decision

### Distance Metrics

**Euclidean Distance**:
```
distance = ||embedding_query - embedding_reference||_2
```

**Confidence Levels**:
- **High**: distance < 0.15 (very confident match)
- **Medium**: 0.15 ≤ distance < 0.25 (confident match)
- **Low**: 0.25 ≤ distance < threshold (weak match)

---

## Performance Considerations

### Detection Performance

| Method | FPS (CPU) | FPS (GPU) | mAP50 | Model Size |
|--------|-----------|-----------|-------|------------|
| Haar Cascade | ~30 | N/A | ~0.70 | ~1 MB |
| YOLOv8n | ~15 | ~60 | ~0.92 | ~6 MB |
| YOLOv8s | ~10 | ~45 | ~0.94 | ~22 MB |

### Recognition Performance

| Model | Inference Time (CPU) | Embedding Dim | Accuracy |
|-------|---------------------|---------------|----------|
| buffalo_s | ~50ms | 512 | Good |
| buffalo_m | ~80ms | 512 | Better |
| buffalo_l | ~120ms | 512 | Best |

### Optimization Tips

1. **Use GPU**: Significant speedup for YOLO detection and training
2. **Batch Processing**: Process multiple images together
3. **Model Selection**: Choose smaller models for real-time applications
4. **Image Resolution**: Lower resolution = faster processing
5. **Reference Images**: Limit to 20-50 high-quality images per identity

### Memory Usage

- **YOLO Model**: ~200-500 MB
- **InsightFace Model**: ~100-300 MB
- **Reference Embeddings**: ~2 KB per image
- **Total**: ~500 MB - 1 GB typical

---

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed troubleshooting guide.

### Common Issues

1. **High Variance in Reference Images**
   - **Symptom**: Validation shows avg_distance > 0.5
   - **Solution**: Filter reference images to ensure consistency

2. **No Matches Found**
   - **Symptom**: "Face detected: unknown"
   - **Solution**: Check threshold, verify reference images, check image quality

3. **Model Download Failures**
   - **Symptom**: RuntimeError during model initialization
   - **Solution**: Use manual download script or check internet connection

4. **CUDA Errors**
   - **Symptom**: CUDAExecutionProvider errors
   - **Solution**: Use CPUExecutionProvider on macOS or systems without CUDA

---

## References & Citations

### Datasets
- **WIDER FACE**: Yang, S., Luo, P., Loy, C. C., & Tang, X. (2016). WIDER FACE: A Face Detection Benchmark. CVPR.

### Models
- **YOLOv8**: Ultralytics. (2023). YOLOv8 Documentation. https://docs.ultralytics.com
- **InsightFace**: Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). ArcFace: Additive Angular Margin Loss for Deep Face Recognition. CVPR.

### Libraries
- **OpenCV**: Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal.
- **TensorFlow**: Abadi, M., et al. (2016). TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems.
- **PyTorch**: Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library.

---

## License & Credits

This project is developed for educational purposes as part of a Computer Vision course.

### Acknowledgments
- Assignment 3 & 4 components reused
- WIDER FACE dataset providers
- Ultralytics for YOLOv8
- InsightFace team for face recognition models

---

## Version History

See [CHANGES_SUMMARY.md](CHANGES_SUMMARY.md) for detailed change log.

### Key Versions
- **v1.0**: Initial implementation with basic detection and recognition
- **v1.1**: Added multiple embeddings support, face deletion, validation

---

## Contact & Support

For issues, questions, or contributions, please refer to:
- Project repository
- Troubleshooting guide
- Technical documentation (this file)

---

*Last Updated: 2024*

