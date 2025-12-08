"""Face recognition utilities using FaceNet/InsightFace embeddings."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import cv2
import numpy as np

try:
    from insightface.app import FaceAnalysis
except ImportError:  # pragma: no cover
    FaceAnalysis = None  # type: ignore


@dataclass
class IdentityProfile:
    name: str
    embeddings: List[np.ndarray] = field(default_factory=list)
    
    def get_average_embedding(self) -> np.ndarray:
        """Return average embedding for better matching."""
        if not self.embeddings:
            raise ValueError(f"No embeddings found for {self.name}")
        return np.mean(self.embeddings, axis=0)
    
    def get_best_match(self, query_embedding: np.ndarray) -> float:
        """Return best (minimum) distance from all embeddings."""
        if not self.embeddings:
            return float("inf")
        distances = [float(np.linalg.norm(query_embedding - emb)) for emb in self.embeddings]
        return min(distances)
    
    def get_all_distances(self, query_embedding: np.ndarray) -> List[float]:
        """Return all distances for analysis."""
        return [float(np.linalg.norm(query_embedding - emb)) for emb in self.embeddings]


class FaceRecognizer:
    """Compute embeddings and match against reference identities."""

    def __init__(self, providers: Optional[List[str]] = None, model_name: Optional[str] = None, 
                 yolo_detector=None):
        """
        Initialize FaceRecognizer.
        
        Args:
            providers: ONNX runtime providers (default: ["CPUExecutionProvider"])
            model_name: InsightFace model name (default: tries buffalo_l, buffalo_m, buffalo_s)
            yolo_detector: Optional YOLOFaceDetector instance to use for face detection.
                          If provided, will use YOLO for detection instead of InsightFace's detector.
                          InsightFace will only be used for embedding extraction from detected faces.
        """
        if FaceAnalysis is None:
            raise ImportError("insightface is required for FaceRecognizer")
        
        # Store YOLO detector if provided
        self.yolo_detector = yolo_detector
        
        # Try multiple models in order of preference
        # Note: buffalo models are more reliable than antelopev2
        model_names = model_name and [model_name] or ["buffalo_l", "buffalo_m", "buffalo_s"]
        providers = providers or ["CPUExecutionProvider"]
        
        self.app = None
        last_error = None
        
        for model in model_names:
            try:
                self.app = FaceAnalysis(name=model, providers=providers)
                self.app.prepare(ctx_id=0, det_size=(640, 640))
                print(f"✓ Successfully loaded model: {model}")
                if self.yolo_detector is not None:
                    print(f"✓ Using YOLO detector for face detection")
                break
            except (AssertionError, RuntimeError, Exception) as e:
                last_error = e
                error_str = str(e).lower()
                error_type = type(e).__name__
                
                # Try next model if download fails, model not found, or incomplete model
                if any(keyword in error_str for keyword in ["failed downloading", "404", "not found", "download", "assertion"]):
                    print(f"✗ Failed to load {model} ({error_type}): {str(e)[:100]}...")
                    print(f"  Trying next model...")
                    continue
                elif error_type == "AssertionError":
                    # Model is incomplete (missing detection module, etc.)
                    print(f"✗ Model {model} is incomplete or corrupted ({error_type})")
                    print(f"  Trying next model...")
                    continue
                else:
                    # Re-raise if it's a different error (e.g., import error)
                    print(f"✗ Unexpected error with {model}: {error_type}: {str(e)[:100]}")
                    continue
        
        if self.app is None:
            error_msg = (
                f"Failed to initialize any insightface model. Tried: {', '.join(model_names)}.\n"
                f"Last error: {last_error}\n\n"
                f"Possible solutions:\n"
                f"1. Check your internet connection and try again\n"
                f"2. Manually download the model using:\n"
                f"   python download_insightface_model.py buffalo_l\n"
                f"3. Try specifying a different model:\n"
                f"   FaceRecognizer(model_name='buffalo_s')\n"
                f"4. Check https://github.com/deepinsight/insightface/releases for available models"
            )
            raise RuntimeError(error_msg)
        
        self.identities: Dict[str, IdentityProfile] = {}

    def add_reference_image(self, name: str, image_path: str | Path, skip_on_error: bool = False) -> bool:
        """
        Add a reference image for an identity.
        
        Args:
            name: Identity name
            image_path: Path to reference image
            skip_on_error: If True, return False instead of raising ValueError when no face is detected
            
        Returns:
            True if successful, False if skipped (only when skip_on_error=True)
        """
        image_path = Path(image_path)
        if not image_path.exists():
            if skip_on_error:
                print(f"⚠️  Skipping {image_path.name}: File not found")
                return False
            raise FileNotFoundError(f"Reference image not found: {image_path}")
        
        # Try to read image, including alpha channel if present
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            if skip_on_error:
                print(f"⚠️  Skipping {image_path.name}: Failed to load image")
                return False
            raise ValueError(f"Failed to load image: {image_path}. Check if it's a valid image file.")
        
        # Handle images with alpha channel (RGBA) - convert to RGB
        if len(image.shape) == 3 and image.shape[2] == 4:
            # Convert BGRA to BGR by compositing on white background
            alpha = image[:, :, 3:4] / 255.0
            bgr = image[:, :, :3]
            white_bg = np.ones_like(bgr) * 255
            image = (bgr * alpha + white_bg * (1 - alpha)).astype(np.uint8)
            print(f"✓ Converted RGBA to RGB for {image_path.name}")
        elif len(image.shape) == 2:
            # Grayscale image - convert to BGR
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            print(f"✓ Converted grayscale to BGR for {image_path.name}")
        
        # Extract embedding - use YOLO if available, otherwise InsightFace detector
        embedding = self._extract_embedding(image)
        
        # If no face detected and using YOLO, try with InsightFace as fallback
        if embedding is None and self.yolo_detector is not None:
            # Fallback to InsightFace detector
            faces = self.app.get(image)
            if faces:
                embedding = faces[0].normed_embedding
                if embedding is not None:
                    print(f"✓ Face detected in {image_path.name} using InsightFace fallback")
        
        # If still no face detected, try with different detection sizes (only if not using YOLO)
        if embedding is None and self.yolo_detector is None:
            # Try with larger detection size for better detection of small faces
            try:
                self.app.prepare(ctx_id=0, det_size=(1280, 1280))
                embedding = self._extract_embedding(image)
                if embedding is not None:
                    print(f"✓ Face detected in {image_path.name} with larger detection size")
            except Exception as e:
                # If that fails, restore to default and continue
                try:
                    self.app.prepare(ctx_id=0, det_size=(640, 640))
                except Exception:
                    pass
        
        if embedding is None:
            # Try resizing the image if it's very small or very large
            h, w = image.shape[:2]
            if h < 100 or w < 100:
                # Image is too small, try upscaling
                scale = max(200 / h, 200 / w)
                new_h, new_w = int(h * scale), int(w * scale)
                image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                embedding = self._extract_embedding(image_resized)
                if embedding is not None:
                    print(f"✓ Face detected in {image_path.name} after upscaling")
            elif h > 2000 or w > 2000:
                # Image is too large, try downscaling
                scale = min(1000 / h, 1000 / w)
                new_h, new_w = int(h * scale), int(w * scale)
                image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                embedding = self._extract_embedding(image_resized)
                if embedding is not None:
                    print(f"✓ Face detected in {image_path.name} after downscaling")
        
        if embedding is None:
            if skip_on_error:
                print(f"⚠️  Skipping {image_path.name}: No face detected (dimensions: {image.shape[1]}x{image.shape[0]})")
                return False
            raise ValueError(
                f"No face detected in reference image {image_path.name}\n"
                f"Image dimensions: {image.shape[1]}x{image.shape[0]}\n"
                f"Please ensure the image contains a clear, frontal face."
            )
        
        # Initialize identity if it doesn't exist
        if name not in self.identities:
            self.identities[name] = IdentityProfile(name, [])
        
        # Add embedding to the list (supports multiple reference images per identity)
        self.identities[name].embeddings.append(embedding)
        print(f"✓ Added reference {len(self.identities[name].embeddings)} for '{name}' from {image_path.name}")
        return True

    def _extract_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from image.
        Uses YOLO detector if available, otherwise InsightFace detector.
        """
        if self.yolo_detector is not None:
            # Use YOLO for detection - try multiple confidence thresholds
            detections = self.yolo_detector.detect(image, conf=0.1)
            if not detections:
                detections = self.yolo_detector.detect(image, conf=0.05)
            if not detections:
                detections = self.yolo_detector.detect(image, conf=0.01)
            if not detections:
                return None
            
            # Use the first (highest confidence) detection
            det = detections[0]
            x1, y1, x2, y2 = det.bbox
            
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            # Crop face region
            face_crop = image[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                return None
            
            # Use InsightFace to extract embedding from the cropped face
            # InsightFace's get() method can work on face crops directly
            faces = self.app.get(face_crop)
            if not faces:
                return None
            
            return faces[0].normed_embedding
        else:
            # Use InsightFace's built-in detector
            faces = self.app.get(image)
            if not faces:
                return None
            return faces[0].normed_embedding

    def match(self, image: np.ndarray, threshold: float = 0.25, confidence_levels: bool = True, 
              return_all_scores: bool = False) -> List[Dict[str, float]]:
        """
        Match faces in image against registered identities.
        
        Args:
            image: Input image (BGR format)
            threshold: Maximum distance for a match (lower = stricter). 
                      Default 0.25. For high variance reference images (avg_distance > 0.4), 
                      consider using 0.5-0.6. Typical good embeddings have avg_distance < 0.3.
            confidence_levels: If True, add confidence level to results
            return_all_scores: If True, include best_score even if above threshold (for diagnostics)
        
        Returns:
            List of matches with name, score, bbox, and optionally confidence level
        """
        if not self.identities:
            raise ValueError("No identities registered. Add reference images first using add_reference_image().")
        
        # Detect faces using YOLO if available, otherwise InsightFace
        if self.yolo_detector is not None:
            # Use YOLO for detection - try lower confidence threshold first
            detections = self.yolo_detector.detect(image, conf=0.1)
            # If no detections with low threshold, try even lower
            if not detections:
                detections = self.yolo_detector.detect(image, conf=0.05)
            # If still no detections, try with very low threshold
            if not detections:
                detections = self.yolo_detector.detect(image, conf=0.01)
            # Debug: Check what YOLO model is detecting
            if not detections:
                # Try direct YOLO predict to see what's happening
                from ultralytics import YOLO
                import numpy as np
                # Get the model from detector
                yolo_model = self.yolo_detector.model
                results = yolo_model.predict(image, conf=0.01, verbose=False)
                print(f"DEBUG: YOLO direct predict - {len(results)} results")
                for i, result in enumerate(results):
                    if result.boxes is not None:
                        print(f"  Result {i}: {len(result.boxes)} boxes detected")
                        for j, box in enumerate(result.boxes):
                            label_idx = int(box.cls[0].item())
                            label = yolo_model.names.get(label_idx, "unknown")
                            score = float(box.conf[0].item())
                            print(f"    Box {j}: label={label} (idx={label_idx}), score={score:.3f}")
                    else:
                        print(f"  Result {i}: No boxes")
                print(f"DEBUG: Image shape: {image.shape}, dtype: {image.dtype}")
            # If still no detections, fall back to InsightFace
            if not detections:
                # Fallback to InsightFace detector
                faces = self.app.get(image)
                if not faces:
                    return []  # No faces detected by either method
                # Continue with InsightFace path (will be handled below)
                use_yolo = False
            else:
                use_yolo = True
            
            if use_yolo:
                matches = []
                for det in detections:
                    x1, y1, x2, y2 = det.bbox
                    
                    # Ensure coordinates are within image bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(image.shape[1], x2)
                    y2 = min(image.shape[0], y2)
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    # Crop face region
                    face_crop = image[y1:y2, x1:x2]
                    
                    if face_crop.size == 0:
                        continue
                    
                    # Extract embedding using InsightFace
                    faces = self.app.get(face_crop)
                    if not faces:
                        continue
                    
                    embedding = faces[0].normed_embedding
                    best_name, best_score = None, float("inf")
                    
                    # Find best match across all identities and their embeddings
                    for profile in self.identities.values():
                        if not profile.embeddings:
                            continue  # Skip identities with no embeddings
                        # Use best match from all embeddings for this identity
                        score = profile.get_best_match(embedding)
                        if score < best_score:
                            best_name, best_score = profile.name, score
                    
                    # Check if we found a match within threshold
                    if best_name is not None and best_score <= threshold:
                        match_dict = {
                            "name": best_name, 
                            "score": best_score, 
                            "bbox": [x1, y1, x2, y2]  # Use YOLO bbox
                        }
                        
                        if confidence_levels:
                            # Add confidence level
                            if best_score < 0.15:
                                match_dict["confidence"] = "high"
                            elif best_score < 0.25:
                                match_dict["confidence"] = "medium"
                            else:
                                match_dict["confidence"] = "low"
                        
                        matches.append(match_dict)
                    elif return_all_scores and best_name is not None:
                        # Return best match even if above threshold (for diagnostics)
                        match_dict = {
                            "name": best_name,
                            "score": best_score,
                            "bbox": [x1, y1, x2, y2],  # Use YOLO bbox
                            "above_threshold": True
                        }
                        if confidence_levels:
                            match_dict["confidence"] = "very_low"
                        matches.append(match_dict)
                
                return matches
            else:
                # Fallback to InsightFace detector (YOLO didn't detect anything)
                faces = self.app.get(image)
                if not faces:
                    return []  # No faces detected
                
                matches = []
                for face in faces:
                    embedding = face.normed_embedding
                    best_name, best_score = None, float("inf")
                    
                    # Find best match across all identities and their embeddings
                    for profile in self.identities.values():
                        if not profile.embeddings:
                            continue  # Skip identities with no embeddings
                        # Use best match from all embeddings for this identity
                        score = profile.get_best_match(embedding)
                        if score < best_score:
                            best_name, best_score = profile.name, score
                    
                    # Check if we found a match within threshold
                    if best_name is not None and best_score <= threshold:
                        match_dict = {
                            "name": best_name, 
                            "score": best_score, 
                            "bbox": face.bbox.tolist()
                        }
                        
                        if confidence_levels:
                            # Add confidence level
                            if best_score < 0.15:
                                match_dict["confidence"] = "high"
                            elif best_score < 0.25:
                                match_dict["confidence"] = "medium"
                            else:
                                match_dict["confidence"] = "low"
                        
                        matches.append(match_dict)
                    elif return_all_scores and best_name is not None:
                        # Return best match even if above threshold (for diagnostics)
                        match_dict = {
                            "name": best_name,
                            "score": best_score,
                            "bbox": face.bbox.tolist(),
                            "above_threshold": True
                        }
                        if confidence_levels:
                            match_dict["confidence"] = "very_low"
                        matches.append(match_dict)
                
                return matches
        else:
            # Use InsightFace's built-in detector
            faces = self.app.get(image)
            if not faces:
                return []  # No faces detected
            
            matches = []
            for face in faces:
                embedding = face.normed_embedding
                best_name, best_score = None, float("inf")
                
                # Find best match across all identities and their embeddings
                for profile in self.identities.values():
                    if not profile.embeddings:
                        continue  # Skip identities with no embeddings
                    # Use best match from all embeddings for this identity
                    score = profile.get_best_match(embedding)
                    if score < best_score:
                        best_name, best_score = profile.name, score
                
                # Check if we found a match within threshold
                if best_name is not None and best_score <= threshold:
                    match_dict = {
                        "name": best_name, 
                        "score": best_score, 
                        "bbox": face.bbox.tolist()
                    }
                    
                    if confidence_levels:
                        # Add confidence level
                        if best_score < 0.15:
                            match_dict["confidence"] = "high"
                        elif best_score < 0.25:
                            match_dict["confidence"] = "medium"
                        else:
                            match_dict["confidence"] = "low"
                    
                    matches.append(match_dict)
                elif return_all_scores and best_name is not None:
                    # Return best match even if above threshold (for diagnostics)
                    match_dict = {
                        "name": best_name,
                        "score": best_score,
                        "bbox": face.bbox.tolist(),
                        "above_threshold": True
                    }
                    if confidence_levels:
                        match_dict["confidence"] = "very_low"
                    matches.append(match_dict)
            
            return matches
    
    def validate_reference_images(self, name: str) -> Dict[str, Any]:
        """
        Validate that all reference images for an identity are consistent.
        
        Returns:
            Dictionary with validation results
        """
        if name not in self.identities:
            return {"valid": False, "reason": "Identity not found"}
        
        profile = self.identities[name]
        if len(profile.embeddings) < 2:
            return {
                "valid": True, 
                "reason": "Only one reference image (cannot validate consistency)",
                "num_references": 1
            }
        
        # Check consistency between embeddings
        distances = []
        for i in range(len(profile.embeddings)):
            for j in range(i+1, len(profile.embeddings)):
                dist = np.linalg.norm(profile.embeddings[i] - profile.embeddings[j])
                distances.append(dist)
        
        avg_distance = np.mean(distances)
        max_distance = np.max(distances)
        std_distance = np.std(distances)
        
        # If average distance is too high, images might be of different people
        # Typical good embeddings have avg_distance < 0.3
        # 0.3-0.5 is acceptable but may have some variation
        # > 0.5 suggests different people or very poor quality images
        if avg_distance > 0.5:
            return {
                "valid": False, 
                "reason": f"Very high variance in embeddings (avg: {avg_distance:.3f})",
                "suggestion": "Reference images likely contain different people. Please ensure all reference images are of the same person.",
                "avg_distance": avg_distance,
                "max_distance": max_distance,
                "std_distance": std_distance,
                "num_references": len(profile.embeddings),
                "recommendation": "Filter reference images to only include the target person"
            }
        elif avg_distance > 0.4:
            return {
                "valid": True,  # Still valid but with warning
                "reason": f"Moderate variance in embeddings (avg: {avg_distance:.3f})",
                "warning": "Reference images may have some inconsistency",
                "suggestion": "Consider reviewing reference images to ensure they're all of the same person",
                "avg_distance": avg_distance,
                "max_distance": max_distance,
                "std_distance": std_distance,
                "num_references": len(profile.embeddings)
            }
        
        return {
            "valid": True,
            "avg_distance": avg_distance,
            "max_distance": max_distance,
            "std_distance": std_distance,
            "num_references": len(profile.embeddings)
        }
    
    def filter_outlier_embeddings(self, name: str, max_distance_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Filter out outlier embeddings that are too different from the majority.
        
        Args:
            name: Identity name
            max_distance_threshold: Maximum average distance to other embeddings before being considered outlier
        
        Returns:
            Dictionary with filtering results
        """
        if name not in self.identities:
            return {"success": False, "reason": "Identity not found"}
        
        profile = self.identities[name]
        if len(profile.embeddings) < 3:
            return {
                "success": False,
                "reason": "Need at least 3 reference images to filter outliers",
                "num_references": len(profile.embeddings)
            }
        
        # Calculate average distance from each embedding to all others
        embedding_distances = []
        for i, emb in enumerate(profile.embeddings):
            distances = []
            for j, other_emb in enumerate(profile.embeddings):
                if i != j:
                    dist = np.linalg.norm(emb - other_emb)
                    distances.append(dist)
            avg_dist = np.mean(distances)
            embedding_distances.append((i, avg_dist))
        
        # Sort by average distance (highest first = most likely outliers)
        embedding_distances.sort(key=lambda x: x[1], reverse=True)
        
        # Filter out embeddings with average distance > threshold
        outliers_removed = []
        filtered_embeddings = []
        filtered_indices = []
        
        for idx, avg_dist in embedding_distances:
            if avg_dist > max_distance_threshold:
                outliers_removed.append((idx, avg_dist))
            else:
                filtered_embeddings.append(profile.embeddings[idx])
                filtered_indices.append(idx)
        
        if outliers_removed:
            # Update profile with filtered embeddings
            profile.embeddings = filtered_embeddings
            return {
                "success": True,
                "outliers_removed": len(outliers_removed),
                "remaining": len(filtered_embeddings),
                "outlier_indices": [idx for idx, _ in outliers_removed],
                "outlier_distances": [dist for _, dist in outliers_removed]
            }
        else:
            return {
                "success": True,
                "outliers_removed": 0,
                "remaining": len(profile.embeddings),
                "message": "No outliers found above threshold"
            }
    
    def get_identity_info(self, name: str) -> Dict[str, Any]:
        """Get information about a registered identity."""
        if name not in self.identities:
            return {"exists": False}
        
        profile = self.identities[name]
        return {
            "exists": True,
            "name": name,
            "num_references": len(profile.embeddings),
            "validation": self.validate_reference_images(name)
        }

__all__ = ["FaceRecognizer", "IdentityProfile"]
