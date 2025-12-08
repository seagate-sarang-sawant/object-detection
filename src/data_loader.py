"""Data loading utilities for face detection pipeline.
Reuses tfds ingestion patterns from Assignment 3, extending to face datasets.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import tensorflow as tf
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.AUTOTUNE


def _load_wider_face_direct(
    split: str,
    data_dir: Optional[str | Path],
    image_size: Tuple[int, int],
) -> tf.data.Dataset:
    """Load WIDER FACE dataset directly from extracted files (fallback when TFDS fails).
    
    This function reads directly from the extracted WIDER FACE directories and annotation files.
    It's used as a fallback when TFDS has download issues.
    
    Args:
        split: Dataset split ("train" or "validation")
        data_dir: Base data directory
        image_size: Target image size
        
    Returns:
        A tf.data.Dataset with the same format as TFDS
    """
    data_dir = Path(data_dir) if data_dir else Path("data")
    
    # Map split names
    split_map = {
        "train": "WIDER_train",
        "validation": "WIDER_val",
        "val": "WIDER_val",
    }
    wider_split = split_map.get(split, "WIDER_train")
    
    # Find extracted directories
    extracted_dir = data_dir / "downloads" / "extracted"
    if not extracted_dir.exists():
        raise ValueError(f"Extracted files not found in {extracted_dir}")
    
    # First, try to find WIDER_* directories directly in extracted/
    wider_dir = extracted_dir / wider_split
    if not wider_dir.exists():
        # If not found directly, look inside ZIP extraction folders
        wider_dirs = list(extracted_dir.glob(f"*/{wider_split}"))
        if not wider_dirs:
            raise ValueError(
                f"Could not find {wider_split} in {extracted_dir}.\n"
                f"Expected either:\n"
                f"  - {extracted_dir / wider_split}\n"
                f"  - {extracted_dir / 'ZIP.*' / wider_split}"
            )
        wider_dir = wider_dirs[0]
    
    images_dir = wider_dir / "images"
    
    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")
    
    # Collect all image files
    image_files = []
    for category_dir in sorted(images_dir.iterdir()):
        if category_dir.is_dir():
            image_files.extend(category_dir.glob("*.jpg"))
    
    if not image_files:
        raise ValueError(f"No images found in {images_dir}")
    
    # Create a generator function
    def generator() -> Iterable[Dict[str, tf.Tensor]]:
        for img_path in image_files:
            try:
                # Read and process image
                image_bytes = tf.io.read_file(str(img_path))
                image = tf.io.decode_jpeg(image_bytes, channels=3)
                original_height = tf.cast(tf.shape(image)[0], tf.float32)
                original_width = tf.cast(tf.shape(image)[1], tf.float32)
                
                # Resize image
                image = tf.image.resize(image, image_size)
                image = tf.image.convert_image_dtype(image, tf.float32)
                
                # For now, create empty bboxes (can be enhanced with annotation parsing)
                # WIDER FACE annotations are complex, so we'll return empty boxes for now
                # Users can add annotation parsing if needed
                boxes = tf.zeros((0, 4), dtype=tf.float32)  # Empty boxes
                labels = tf.zeros((0,), dtype=tf.int32)  # Empty labels
                
                yield {
                    "image": image,
                    "bboxes": boxes,
                    "labels": labels,
                    "image_id": img_path.stem,
                }
            except Exception as e:
                warnings.warn(f"Skipping {img_path}: {e}", UserWarning)
                continue
    
    # Define output signature
    output_signature = {
        "image": tf.TensorSpec((*image_size, 3), tf.float32),
        "bboxes": tf.TensorSpec((None, 4), tf.float32),
        "labels": tf.TensorSpec((None,), tf.int32),
        "image_id": tf.TensorSpec((), tf.string),
    }
    
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    return dataset


def load_wider_face(
    split: str = "train",
    data_dir: Optional[str | Path] = None,
    shuffle: bool = True,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (640, 640),
    download_mode: Optional[str] = None,
) -> tf.data.Dataset:
    """Load the WIDER FACE dataset via TFDS.

    Args:
        split: TFDS split ("train", "validation", "test").
        data_dir: Optional override for tfds data location.
        shuffle: Whether to shuffle the dataset.
        batch_size: Batch size for training/eval pipelines.
        image_size: Target size for resizing images.
        download_mode: TFDS download mode. Use "reuse_cache_if_exists" to skip downloads.

    Returns:
        A `tf.data.Dataset` of dictionaries with fields `image`, `bboxes`, `labels`.
    """
    
    # Set download mode to reuse dataset if not specified
    if download_mode is None:
        download_mode = "reuse_dataset_if_exists"
    
    # Handle "skip_download" mode - try to load without any download/preparation
    if download_mode == "skip_download":
        try:
            dataset = tfds.load(
                "wider_face",
                split=split,
                shuffle_files=shuffle,
                data_dir=str(data_dir) if data_dir else None,
                with_info=False,
                as_supervised=False,
                download=False,  # Skip download and preparation entirely
            )
        except (ValueError, RuntimeError, OSError) as e:
            raise ValueError(
                f"Dataset not found. It needs to be downloaded and prepared first.\n"
                f"Original error: {e}\n\n"
                f"Try running without download_mode='skip_download' to download the dataset."
            ) from e
        else:
            # Apply preprocessing and return
            def preprocess(record: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
                image = tf.image.resize(record["image"], image_size)
                image = tf.image.convert_image_dtype(image, tf.float32)

                boxes = record["faces"]["bbox"]  # [N, 4] normalized
                labels = tf.ones((tf.shape(boxes)[0],), dtype=tf.int32)  # single class (face)

                return {"image": image, "bboxes": boxes, "labels": labels}

            dataset = dataset.map(preprocess, num_parallel_calls=AUTOTUNE)

            if shuffle:
                dataset = dataset.shuffle(2048, reshuffle_each_iteration=True)

            dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
            return dataset
    
    # First, try to load without downloading if dataset might already be prepared
    # This is faster and avoids download issues
    elif download_mode in ("reuse_cache_if_exists", "reuse_dataset_if_exists"):
        try:
            dataset = tfds.load(
                "wider_face",
                split=split,
                shuffle_files=shuffle,
                data_dir=str(data_dir) if data_dir else None,
                with_info=False,
                as_supervised=False,
                download=False,  # Skip download and preparation if dataset exists
            )
            # If we get here, the dataset was already prepared, so we can return it
            # Apply preprocessing below
        except (ValueError, RuntimeError, OSError, AssertionError) as e:
            # Dataset not prepared yet, continue with download mode
            # The AssertionError occurs when dataset isn't prepared
            if isinstance(e, AssertionError) and "could not find data" in str(e):
                # Dataset needs to be prepared, but we'll try with reuse mode
                # which should use existing downloaded files
                warnings.warn(
                    "Dataset not prepared yet. Will attempt to prepare using existing downloads.",
                    UserWarning
                )
            pass
        else:
            # Dataset loaded successfully, skip to preprocessing
            def preprocess(record: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
                image = tf.image.resize(record["image"], image_size)
                image = tf.image.convert_image_dtype(image, tf.float32)

                boxes = record["faces"]["bbox"]  # [N, 4] normalized
                labels = tf.ones((tf.shape(boxes)[0],), dtype=tf.int32)  # single class (face)

                return {"image": image, "bboxes": boxes, "labels": labels}

            dataset = dataset.map(preprocess, num_parallel_calls=AUTOTUNE)

            if shuffle:
                dataset = dataset.shuffle(2048, reshuffle_each_iteration=True)

            dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
            return dataset
    
    # Convert string download_mode to TFDS GenerateMode enum
    if download_mode == "reuse_cache_if_exists":
        mode = tfds.GenerateMode.REUSE_CACHE_IF_EXISTS
    elif download_mode == "reuse_dataset_if_exists":
        mode = tfds.GenerateMode.REUSE_DATASET_IF_EXISTS
    elif download_mode == "force_redownload":
        mode = tfds.GenerateMode.FORCE_REDOWNLOAD
    else:
        mode = tfds.GenerateMode.REUSE_DATASET_IF_EXISTS
    
    # Create DownloadConfig with the specified mode
    try:
        download_config = tfds.download.DownloadConfig(download_mode=mode)
        download_kwargs = {"download_config": download_config}
    except (AttributeError, TypeError):
        # Fallback for older TFDS versions that might not support DownloadConfig
        # In this case, we'll skip the download mode specification
        warnings.warn(
            "DownloadConfig not available in this TFDS version. "
            "Skipping download_mode specification.",
            UserWarning
        )
        download_kwargs = {}
    
    # Check if dataset folder exists but is incomplete (missing dataset_info.json)
    # This handles the case where preparation was started but not completed
    data_dir_path = Path(data_dir) if data_dir else Path.home() / "tensorflow_datasets"
    dataset_dir = data_dir_path / "wider_face" / "0.1.0"
    dataset_info_path = dataset_dir / "dataset_info.json"
    
    dataset = None  # Track if we've already loaded the dataset
    
    if dataset_dir.exists() and not dataset_info_path.exists():
        # Incomplete preparation detected - need to complete it
        if download_mode in ("reuse_cache_if_exists", "reuse_dataset_if_exists", None):
            warnings.warn(
                f"Found incomplete dataset preparation in {dataset_dir}.\n"
                f"Completing preparation using existing downloads...",
                UserWarning
            )
            # Force completion of preparation using existing downloads
            try:
                mode = tfds.GenerateMode.REUSE_CACHE_IF_EXISTS
                download_config = tfds.download.DownloadConfig(download_mode=mode)
                # Load with preparation to complete the incomplete dataset
                dataset = tfds.load(
                    "wider_face",
                    split=split,
                    shuffle_files=shuffle,
                    data_dir=str(data_dir) if data_dir else None,
                    with_info=False,
                    as_supervised=False,
                    download_and_prepare_kwargs={"download_config": download_config},
                )
                # Successfully completed preparation, skip to preprocessing below
            except Exception as prep_error:
                warnings.warn(
                    f"Failed to complete preparation: {prep_error}\n"
                    f"Attempting to clean incomplete preparation and retry...",
                    UserWarning
                )
                # Try to remove incomplete folder and retry
                try:
                    import shutil
                    if dataset_dir.exists():
                        shutil.rmtree(dataset_dir)
                    warnings.warn("Removed incomplete dataset folder. Retrying...", UserWarning)
                except Exception as cleanup_error:
                    warnings.warn(
                        f"Could not clean incomplete folder: {cleanup_error}\n"
                        f"Continuing with normal load attempt...",
                        UserWarning
                    )
    
    # Build tfds.load arguments
    load_kwargs = {
        "split": split,
        "shuffle_files": shuffle,
        "data_dir": str(data_dir) if data_dir else None,
        "with_info": False,
        "as_supervised": False,
    }
    if download_kwargs:
        load_kwargs["download_and_prepare_kwargs"] = download_kwargs
    
    # Only try to load if we haven't already loaded it above
    if dataset is None:
        try:
            dataset = tfds.load("wider_face", **load_kwargs)
        except (ValueError, TypeError, AssertionError) as e:
            error_msg = str(e)
            
            # Handle AssertionError when dataset is not prepared
            if isinstance(e, AssertionError) and "could not find data" in error_msg:
                # Try to prepare the dataset using existing downloads
                if download_mode in ("reuse_cache_if_exists", "reuse_dataset_if_exists"):
                    warnings.warn(
                    "Dataset not prepared. Attempting to prepare using REUSE_CACHE_IF_EXISTS mode.\n"
                    "This will use existing downloaded files if available.",
                    UserWarning
                )
                # Use REUSE_CACHE_IF_EXISTS to prepare from existing downloads
                try:
                    mode = tfds.GenerateMode.REUSE_CACHE_IF_EXISTS
                    download_config = tfds.download.DownloadConfig(download_mode=mode)
                    dataset = tfds.load(
                        "wider_face",
                        split=split,
                        shuffle_files=shuffle,
                        data_dir=str(data_dir) if data_dir else None,
                        with_info=False,
                        as_supervised=False,
                        download_and_prepare_kwargs={"download_config": download_config},
                    )
                    # Successfully prepared, continue to preprocessing below
                except Exception as prep_error:
                    raise ValueError(
                        f"Failed to prepare dataset from existing downloads.\n"
                        f"Original error: {error_msg}\n"
                        f"Preparation error: {prep_error}\n\n"
                        f"SOLUTIONS:\n"
                        f"1. The dataset needs to be prepared. Try running:\n"
                        f"   import tensorflow_datasets as tfds\n"
                        f"   builder = tfds.builder('wider_face', data_dir='{data_dir or 'default'}')\n"
                        f"   builder.download_and_prepare()\n\n"
                        f"2. Or use download_mode='force_redownload' to re-download and prepare.\n"
                        f"3. Make sure the downloaded files are in the correct location."
                    ) from prep_error
                else:
                    raise ValueError(
                    f"Dataset not prepared. It needs to be downloaded and prepared first.\n"
                    f"Original error: {error_msg}\n\n"
                    f"SOLUTIONS:\n"
                    f"1. Try with download_mode='reuse_cache_if_exists' to prepare from existing downloads:\n"
                    f"   wider_ds = load_wider_face(..., download_mode='reuse_cache_if_exists')\n\n"
                    f"2. Or prepare manually:\n"
                    f"   import tensorflow_datasets as tfds\n"
                    f"   builder = tfds.builder('wider_face', data_dir='{data_dir or 'default'}')\n"
                    f"   builder.download_and_prepare()\n"
                    ) from e
                # If we successfully prepared, continue to preprocessing (skip the rest of error handling)
            elif "Failed to obtain confirmation link for GDrive URL" in error_msg:
                # Extract the Google Drive URL from the error message
                url_match = re.search(r'https://drive\.google\.com/[^\s]+', error_msg)
                if url_match:
                    gdrive_url = url_match.group(0)
                    
                    # Extract file ID for gdown
                    file_id_match = re.search(r'id=([a-zA-Z0-9_-]+)', gdrive_url)
                    if file_id_match:
                        file_id = file_id_match.group(1)
                        gdrive_direct_url = f"https://drive.google.com/uc?id={file_id}"
                    else:
                        gdrive_direct_url = gdrive_url
                    
                    # Determine download directory
                    if data_dir:
                        download_dir = Path(data_dir) / "wider_face"
                    else:
                        download_dir = Path.home() / "tensorflow_datasets" / "wider_face"
                    
                    # Check if dataset might already be prepared
                    try:
                        builder = tfds.builder("wider_face", data_dir=str(data_dir) if data_dir else None)
                        if builder.info.version and Path(builder.data_dir).exists():
                            error_instructions = (
                                f"\n{'='*80}\n"
                                f"Google Drive download failed, but dataset may already be prepared.\n"
                                f"Try loading with download=False:\n"
                                f"  wider_ds = load_wider_face(..., download_mode='skip_download')\n\n"
                                f"If that doesn't work, you need to download the files manually:\n"
                                f"URL: {gdrive_url}\n\n"
                                f"SOLUTIONS:\n"
                                f"1. Try with download=False first:\n"
                                f"   wider_ds = load_wider_face(..., download_mode='skip_download')\n\n"
                                f"2. Install gdown and download manually:\n"
                                f"   pip install gdown\n"
                                f"   gdown {gdrive_direct_url}\n\n"
                                f"3. Or download manually from: {gdrive_url}\n"
                                f"   Then place the file in: {download_dir}\n"
                                f"{'='*80}\n"
                            )
                        else:
                            raise ValueError("Dataset not prepared")
                    except Exception:
                        error_instructions = (
                            f"\n{'='*80}\n"
                            f"Google Drive download failed. This is a known issue with large files.\n"
                            f"URL: {gdrive_url}\n\n"
                            f"SOLUTIONS:\n"
                            f"1. Install gdown and download manually:\n"
                            f"   pip install gdown\n"
                            f"   gdown {gdrive_direct_url}\n\n"
                            f"2. Or download manually from: {gdrive_url}\n"
                            f"   Then place the file in: {download_dir}\n\n"
                            f"3. After downloading, retry with:\n"
                            f"   load_wider_face(..., download_mode='reuse_dataset_if_exists')\n"
                            f"{'='*80}\n"
                        )
                    
                    # Try fallback to direct loader from extracted files first
                    warnings.warn(
                        "TFDS download failed. Attempting to load directly from extracted files...",
                        UserWarning
                    )
                    try:
                        dataset = _load_wider_face_direct(split, data_dir, image_size)
                        # Successfully loaded from extracted files, continue to preprocessing
                        warnings.warn(
                            "Successfully loaded dataset from extracted files (without annotations).\n"
                            "Note: Bounding boxes are empty. To load with annotations, ensure annotation files are available.",
                            UserWarning
                        )
                    except Exception as fallback_error:
                        # Fallback failed, try gdown if available
                        try:
                            import gdown  # type: ignore
                            warnings.warn(
                                f"Direct loader failed: {fallback_error}\n"
                                f"Attempting to download with gdown...\n"
                                f"This may take a while for large files.",
                                UserWarning
                            )
                            # Note: gdown can't directly fix TFDS cache, but we provide the info
                            raise ValueError(error_instructions) from e
                        except ImportError:
                            raise ValueError(
                                error_instructions +
                                f"\nTo use gdown, install it first: pip install gdown"
                            ) from e
                else:
                    # No URL match found, try fallback to direct loader
                    warnings.warn(
                        "TFDS download failed. Attempting to load directly from extracted files...",
                        UserWarning
                    )
                    try:
                        dataset = _load_wider_face_direct(split, data_dir, image_size)
                        warnings.warn(
                            "Successfully loaded dataset from extracted files (without annotations).\n"
                            "Note: Bounding boxes are empty. To load with annotations, ensure annotation files are available.",
                            UserWarning
                        )
                    except Exception as fallback_error:
                        raise ValueError(
                            f"Google Drive download error occurred.\n"
                            f"Original error: {error_msg}\n\n"
                            f"Fallback loader also failed: {fallback_error}\n\n"
                            f"Try installing gdown: pip install gdown\n"
                            f"Then manually download the required files."
                        ) from e
            else:
                # Re-raise if it's a different ValueError or TypeError  
                raise

    def preprocess(record: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        image = tf.image.resize(record["image"], image_size)
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Handle both TFDS format (record["faces"]["bbox"]) and direct loader format (record["bboxes"])
        if "faces" in record:
            boxes = record["faces"]["bbox"]  # [N, 4] normalized (TFDS format)
        else:
            boxes = record["bboxes"]  # [N, 4] normalized (direct loader format)
        
        labels = tf.ones((tf.shape(boxes)[0],), dtype=tf.int32)  # single class (face)

        return {"image": image, "bboxes": boxes, "labels": labels}

    dataset = dataset.map(preprocess, num_parallel_calls=AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(2048, reshuffle_each_iteration=True)

    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset


def load_custom_pascal_voc(
    images_dir: str | Path,
    annotations_dir: str | Path,
    class_map: Optional[Dict[str, int]] = None,
    image_size: Tuple[int, int] = (640, 640),
) -> tf.data.Dataset:
    """Load a folder of images + Pascal VOC (XML) annotations using tf.data."""

    images_dir = Path(images_dir)
    annotations_dir = Path(annotations_dir)

    # Collect all JPG files - check both directly in images_dir and in subdirectories
    img_files = []
    
    # First, check for JPG files directly in images_dir
    img_files.extend(images_dir.glob("*.jpg"))
    img_files.extend(images_dir.glob("*.JPG"))
    
    # Then, check in subdirectories
    for subdir in images_dir.iterdir():
        if subdir.is_dir():
            img_files.extend(subdir.glob("*.jpg"))
            img_files.extend(subdir.glob("*.JPG"))
    
    img_files = sorted(img_files)
    def generator() -> Iterable[Dict[str, tf.Tensor]]:
        import xml.etree.ElementTree as ET
        import numpy as np  # Lazy import for generator only

        for img_path in img_files:
            xml_path = annotations_dir / (img_path.stem + ".xml")
            if not xml_path.exists():
                continue

            image = tf.io.read_file(str(img_path))
            image = tf.io.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, image_size)
            image = tf.image.convert_image_dtype(image, tf.float32)

            tree = ET.parse(str(xml_path))
            root = tree.getroot()

            boxes = []
            labels = []
            for obj in root.findall("object"):
                label = obj.find("name").text
                bbox = obj.find("bndbox")
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)

                size = root.find("size")
                width = float(size.find("width").text)
                height = float(size.find("height").text)

                boxes.append([
                    ymin / height,
                    xmin / width,
                    ymax / height,
                    xmax / width,
                ])

                if class_map and label in class_map:
                    labels.append(class_map[label])
                else:
                    labels.append(1)

            yield {
                "image": image,
                "bboxes": tf.constant(boxes, dtype=tf.float32),
                "labels": tf.constant(labels, dtype=tf.int32),
                "image_id": img_path.stem,
            }

    output_signature = {
        "image": tf.TensorSpec((*image_size, 3), tf.float32),
        "bboxes": tf.TensorSpec((None, 4), tf.float32),
        "labels": tf.TensorSpec((None,), tf.int32),
        "image_id": tf.TensorSpec((), tf.string),
    }

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    return dataset


def load_yolo_dataset(
    train_dir: str | Path,
    val_dir: str | Path,
    image_size: Tuple[int, int] = (640, 640),
    split: str = "train",
) -> tf.data.Dataset:
    """Load images and YOLO format annotations from train/val folders.
    
    Args:
        train_dir: Directory containing train images and .txt annotations
        val_dir: Directory containing val images and .txt annotations
        image_size: Target image size for resizing
        split: Which split to load ("train" or "val")
        
    Returns:
        A tf.data.Dataset with fields: image, bboxes, labels, image_id
    """
    data_dir = Path(train_dir if split == "train" else val_dir)
    
    if not data_dir.exists():
        raise ValueError(f"Directory not found: {data_dir}")
    
    # Collect all JPG files
    img_files = sorted(list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.JPG")))
    
    if not img_files:
        raise ValueError(f"No images found in {data_dir}")
    
    def generator() -> Iterable[Dict[str, tf.Tensor]]:
        for img_path in img_files:
            # YOLO annotation file has same name as image but .txt extension
            txt_path = data_dir / (img_path.stem + ".txt")
            
            if not txt_path.exists():
                # Skip images without annotations
                continue
            
            # Read and process image
            image = tf.io.read_file(str(img_path))
            image = tf.io.decode_jpeg(image, channels=3)
            original_height = tf.cast(tf.shape(image)[0], tf.float32)
            original_width = tf.cast(tf.shape(image)[1], tf.float32)
            
            # Resize image
            image = tf.image.resize(image, image_size)
            image = tf.image.convert_image_dtype(image, tf.float32)
            
            # Read YOLO format annotations
            # Format: class_id x_center y_center width height (all normalized 0-1)
            boxes = []
            labels = []
            
            try:
                annotation_text = txt_path.read_text().strip()
                if annotation_text:  # Only process if file is not empty
                    for line in annotation_text.split('\n'):
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            # Convert YOLO format (center, width, height) to (ymin, xmin, ymax, xmax)
                            xmin = x_center - width / 2.0
                            ymin = y_center - height / 2.0
                            xmax = x_center + width / 2.0
                            ymax = y_center + height / 2.0
                            
                            boxes.append([ymin, xmin, ymax, xmax])
                            labels.append(class_id)
            except Exception as e:
                warnings.warn(f"Error reading annotation {txt_path}: {e}", UserWarning)
                continue
            
            if not boxes:  # Skip if no valid boxes found
                continue
            
            yield {
                "image": image,
                "bboxes": tf.constant(boxes, dtype=tf.float32),
                "labels": tf.constant(labels, dtype=tf.int32),
                "image_id": img_path.stem,
            }
    
    output_signature = {
        "image": tf.TensorSpec((*image_size, 3), tf.float32),
        "bboxes": tf.TensorSpec((None, 4), tf.float32),
        "labels": tf.TensorSpec((None,), tf.int32),
        "image_id": tf.TensorSpec((), tf.string),
    }
    
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    return dataset


def save_dataset_statistics(dataset: tf.data.Dataset, output_path: str | Path, sample_size: int = 100) -> None:
    """Compute simple dataset stats and write to JSON."""

    output_path = Path(output_path)
    stats = {
        "num_samples": 0,
        "avg_num_faces": 0.0,
    }

    total_faces = 0
    for idx, sample in enumerate(dataset.take(sample_size)):
        stats["num_samples"] += 1
        total_faces += int(sample["labels"].shape[0])

    if stats["num_samples"]:
        stats["avg_num_faces"] = total_faces / stats["num_samples"]

    output_path.write_text(json.dumps(stats, indent=2))


def get_tfds_dataloader(
    dataset_name: str,
    split: str,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (640, 640),
) -> tf.data.Dataset:
    """Generic TFDS loader for future reuse.

    Mirrors the Assignment 3 approach but keeps interface generic.
    """

    dataset = tfds.load(dataset_name, split=split, as_supervised=False)

    def preprocess(record):
        image = tf.image.resize(record["image"], image_size)
        image = tf.image.convert_image_dtype(image, tf.float32)
        labels = record.get("label", tf.constant([0], tf.int32))
        return image, labels

    dataset = dataset.map(preprocess, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset

__all__ = [
    "load_wider_face",
    "load_custom_pascal_voc",
    "load_yolo_dataset",
    "save_dataset_statistics",
    "get_tfds_dataloader",
]
