#!/usr/bin/env python3
"""
Generate YOLO format label files (.txt) by automatically detecting faces in images.

This script:
1. Detects faces in images using OpenCV Haar Cascade or YOLO
2. Converts bounding boxes to YOLO format (normalized center_x, center_y, width, height)
3. Saves .txt label files alongside images
4. Can organize images into train/val structure
"""

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


def bbox_to_yolo_format(bbox: Tuple[int, int, int, int], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """
    Convert bounding box from (xmin, ymin, xmax, ymax) to YOLO format.
    
    YOLO format: (center_x, center_y, width, height) - all normalized [0, 1]
    
    Args:
        bbox: (xmin, ymin, xmax, ymax) in pixels
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        (center_x, center_y, width, height) normalized [0, 1]
    """
    xmin, ymin, xmax, ymax = bbox
    
    # Calculate center and dimensions
    center_x = (xmin + xmax) / 2.0 / img_width
    center_y = (ymin + ymax) / 2.0 / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    
    # Clamp to [0, 1] range
    center_x = max(0.0, min(1.0, center_x))
    center_y = max(0.0, min(1.0, center_y))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    
    return center_x, center_y, width, height


def detect_faces_haar(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect faces using OpenCV Haar Cascade."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Convert (x, y, w, h) to (xmin, ymin, xmax, ymax)
    bboxes = []
    for (x, y, w, h) in faces:
        bboxes.append((int(x), int(y), int(x + w), int(y + h)))
    
    return bboxes


def detect_faces_yolo(image: np.ndarray, model_path: str = "yolov8n.pt", conf_threshold: float = 0.25) -> List[Tuple[int, int, int, int]]:
    """Detect faces using YOLO model."""
    if not YOLO_AVAILABLE:
        raise ImportError("ultralytics is required for YOLO detection")
    
    model = YOLO(model_path)
    results = model.predict(image, conf=conf_threshold, verbose=False)
    
    bboxes = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            # Get class ID - check if it's a person/face class
            class_id = int(box.cls[0].item())
            class_name = model.names.get(class_id, "")
            
            # YOLO models typically have 'person' class (0), but for face detection
            # you might want to use a face-specific model or filter by class
            # For now, we'll accept all detections (you can filter by class_name if needed)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bboxes.append((int(x1), int(y1), int(x2), int(y2)))
    
    return bboxes


def generate_label_file(image_path: Path, output_label_path: Path, bboxes: List[Tuple[int, int, int, int]], class_id: int = 0):
    """
    Generate YOLO format label file.
    
    Args:
        image_path: Path to the image file
        output_label_path: Path where to save the .txt label file
        bboxes: List of bounding boxes in (xmin, ymin, xmax, ymax) format
        class_id: Class ID (0 for face in this case)
    """
    # Read image to get dimensions
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: Could not read image {image_path}")
        return False
    
    img_height, img_width = image.shape[:2]
    
    # Convert all bboxes to YOLO format
    yolo_lines = []
    for bbox in bboxes:
        center_x, center_y, width, height = bbox_to_yolo_format(bbox, img_width, img_height)
        yolo_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
    
    # Write label file
    output_label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_label_path, 'w') as f:
        f.writelines(yolo_lines)
    
    return True


def process_images(
    input_dir: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
    method: str = "haar",
    yolo_model: str = "yolov8n.pt",
    train_split: float = 0.8,
    copy_images: bool = True
):
    """
    Process images and generate YOLO labels.
    
    Args:
        input_dir: Directory containing input images (can have subdirectories)
        output_images_dir: Directory to save images (will create train/val subdirs)
        output_labels_dir: Directory to save labels (will create train/val subdirs)
        method: Detection method ("haar" or "yolo")
        yolo_model: Path to YOLO model if using yolo method
        train_split: Fraction of images to use for training (rest for validation)
        copy_images: If True, copy images to output dir; if False, just generate labels
    """
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(input_dir.rglob(f"*{ext}"))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Shuffle for train/val split
    import random
    random.seed(42)
    random.shuffle(image_files)
    
    split_idx = int(len(image_files) * train_split)
    train_images = image_files[:split_idx]
    val_images = image_files[split_idx:]
    
    print(f"Train: {len(train_images)} images, Val: {len(val_images)} images")
    
    # Process train images
    print("\nProcessing training images...")
    for img_path in tqdm(train_images):
        process_single_image(
            img_path, output_images_dir / "train", output_labels_dir / "train",
            method, yolo_model, copy_images
        )
    
    # Process val images
    print("\nProcessing validation images...")
    for img_path in tqdm(val_images):
        process_single_image(
            img_path, output_images_dir / "val", output_labels_dir / "val",
            method, yolo_model, copy_images
        )
    
    print(f"\nDone! Labels saved to {output_labels_dir}")
    print(f"Images saved to {output_images_dir}")


def process_single_image(
    image_path: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
    method: str,
    yolo_model: str,
    copy_images: bool
):
    """Process a single image and generate its label file."""
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        return
    
    # Detect faces
    if method == "haar":
        bboxes = detect_faces_haar(image)
    elif method == "yolo":
        bboxes = detect_faces_yolo(image, yolo_model)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Skip if no faces detected
    if not bboxes:
        return
    
    # Generate output paths
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy or reference image
    output_image_path = output_images_dir / image_path.name
    if copy_images:
        shutil.copy2(image_path, output_image_path)
    
    # Generate label file
    label_filename = image_path.stem + ".txt"
    output_label_path = output_labels_dir / label_filename
    
    generate_label_file(image_path, output_label_path, bboxes, class_id=0)


def main():
    parser = argparse.ArgumentParser(
        description="Generate YOLO format labels by detecting faces in images"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/raw",
        help="Input directory containing images (default: data/raw)"
    )
    parser.add_argument(
        "--output-images", "-oi",
        type=str,
        default="data/raw",
        help="Output directory for images (default: data/raw, will create train/val subdirs)"
    )
    parser.add_argument(
        "--output-labels", "-ol",
        type=str,
        default="data/raw",
        help="Output directory for labels (default: data/raw, will create train/val subdirs)"
    )
    parser.add_argument(
        "--method", "-m",
        type=str,
        choices=["haar", "yolo"],
        default="haar",
        help="Face detection method: haar (OpenCV) or yolo (default: haar)"
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="yolov8n.pt",
        help="Path to YOLO model if using yolo method (default: yolov8n.pt)"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of images for training (default: 0.8)"
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Don't copy images, just generate labels (images must already be in output dir)"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_images_dir = Path(args.output_images)
    output_labels_dir = Path(args.output_labels)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    process_images(
        input_dir,
        output_images_dir,
        output_labels_dir,
        method=args.method,
        yolo_model=args.yolo_model,
        train_split=args.train_split,
        copy_images=not args.no_copy
    )


if __name__ == "__main__":
    main()

