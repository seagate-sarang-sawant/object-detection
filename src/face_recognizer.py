"""Face recognition utilities using FaceNet/InsightFace embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

try:
    from insightface.app import FaceAnalysis
except ImportError:  # pragma: no cover
    FaceAnalysis = None  # type: ignore


@dataclass
class IdentityProfile:
    name: str
    embedding: np.ndarray


class FaceRecognizer:
    """Compute embeddings and match against reference identities."""

    def __init__(self, providers: Optional[List[str]] = None):
        if FaceAnalysis is None:
            raise ImportError("insightface is required for FaceRecognizer")
        self.app = FaceAnalysis(name="buffalo_m", providers=providers or ["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.identities: Dict[str, IdentityProfile] = {}

    def add_reference_image(self, name: str, image_path: str | Path) -> None:
        image = cv2.imread(str(image_path))
        embedding = self._extract_embedding(image)
        if embedding is None:
            raise ValueError(f"No face detected in reference image {image_path}")
        self.identities[name] = IdentityProfile(name, embedding)

    def _extract_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        faces = self.app.get(image)
        if not faces:
            return None
        return faces[0].normed_embedding

    def match(self, image: np.ndarray, threshold: float = 0.35) -> List[Dict[str, float]]:
        faces = self.app.get(image)
        matches = []
        for face in faces:
            embedding = face.normed_embedding
            best_name, best_score = None, float("inf")
            for profile in self.identities.values():
                score = float(np.linalg.norm(embedding - profile.embedding))
                if score < best_score:
                    best_name, best_score = profile.name, score
            if best_name is not None and best_score <= threshold:
                matches.append({"name": best_name, "score": best_score, "bbox": face.bbox.tolist()})
        return matches

__all__ = ["FaceRecognizer", "IdentityProfile"]
