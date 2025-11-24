"""Dataset wrapper adapted from Assignment 3's RaccoonDataset for faces."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from mrcnn.utils import Dataset
from xml.etree import ElementTree


class FaceDataset(Dataset):
    """Mask R-CNN compatible dataset wrapper for face detection."""

    def load_dataset(self, dataset_dir: str | Path, class_name: str = "person", is_train: bool = True) -> None:
        dataset_dir = Path(dataset_dir)
        images_dir = dataset_dir / "images"
        annotations_dir = dataset_dir / "annotations"

        self.add_class("face_dataset", 1, class_name)

        file_list = sorted([f for f in images_dir.glob("*.jpg")])
        split_idx = int(0.8 * len(file_list))

        for idx, img_path in enumerate(file_list):
            if is_train and idx >= split_idx:
                continue
            if not is_train and idx < split_idx:
                continue

            image_id = img_path.stem
            ann_path = annotations_dir / f"{image_id}.xml"
            if not ann_path.exists():
                continue

            self.add_image(
                "face_dataset",
                image_id=image_id,
                path=str(img_path),
                annotation=str(ann_path),
            )

    def extract_boxes(self, filename: str) -> Tuple[List[List[int]], int, int]:
        tree = ElementTree.parse(filename)
        root = tree.getroot()

        boxes = []
        for box in root.findall(".//bndbox"):
            xmin = int(box.find("xmin").text)
            ymin = int(box.find("ymin").text)
            xmax = int(box.find("xmax").text)
            ymax = int(box.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])

        width = int(root.find(".//size/width").text)
        height = int(root.find(".//size/height").text)
        return boxes, width, height

    def load_mask(self, image_id: int):
        info = self.image_info[image_id]
        boxes, width, height = self.extract_boxes(info["annotation"])

        masks = np.zeros([height, width, len(boxes)], dtype=np.uint8)
        for i, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = box
            masks[ymin:ymax, xmin:xmax, i] = 1

        class_ids = np.array([1 for _ in boxes], dtype=np.int32)
        return masks, class_ids

    def image_reference(self, image_id: int):
        return self.image_info[image_id]["path"]

__all__ = ["FaceDataset"]
