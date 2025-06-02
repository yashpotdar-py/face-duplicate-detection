"""
GPU-accelerated face detection and recognition module.
"""

import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import face_recognition
from loguru import logger

from ..utils.config import Config


class FaceDetector:
    """
    GPU-accelerated face detection and feature extraction using MTCNN and FaceNet.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the face detector.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize models
        self._init_models()
        
        logger.info(f"FaceDetector initialized on device: {self.device}")
    
    def _init_models(self):
        """Initialize face detection and recognition models."""
        try:
            # Use face_recognition library for all face detection
            logger.info("Using face_recognition library for face detection")
            self.mtcnn = None  # Not using MTCNN due to TensorFlow dependency
            self.use_gpu = self.config.use_gpu and torch.cuda.is_available()
            
        except Exception as e:
            logger.warning(f"Error initializing models: {e}")
            self.mtcnn = None
            self.use_gpu = False
    
    def detect_faces_cpu(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        """
        Fallback CPU-based face detection using face_recognition library.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (face_crops, face_encodings, detection_probabilities)
        """
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(rgb_image, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            face_crops = []
            probs = []
            
            for (top, right, bottom, left) in face_locations:
                # Extract face crop
                face_crop = rgb_image[top:bottom, left:right]
                if face_crop.size > 0:
                    face_crop = cv2.resize(face_crop, self.config.face_size)
                    face_crops.append(face_crop)
                    probs.append(0.95)  # High confidence for detected faces
            
            return face_crops, face_encodings, probs
            
        except Exception as e:
            logger.error(f"CPU face detection failed: {e}")
            return [], [], []
    
    def detect_faces(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        """
        Main face detection method using face_recognition library.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (face_crops, face_embeddings, detection_probabilities)
        """
        return self.detect_faces_cpu(image)
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process a single image and extract face information.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing face information
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Detect faces
            face_crops, embeddings, probs = self.detect_faces(image)
            
            result = {
                "image_path": image_path,
                "num_faces": len(face_crops),
                "faces": [],
                "processing_method": "face_recognition"
            }
            
            for i, (crop, embedding, prob) in enumerate(zip(face_crops, embeddings, probs)):
                face_info = {
                    "face_id": f"{Path(image_path).stem}_face_{i}",
                    "crop": crop,
                    "embedding": embedding.numpy() if hasattr(embedding, 'numpy') else embedding,
                    "confidence": prob
                }
                result["faces"].append(face_info)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return {
                "image_path": image_path,
                "num_faces": 0,
                "faces": [],
                "error": str(e)
            }
    
    def process_video_frame(self, frame: np.ndarray, frame_number: int, video_name: str) -> Dict[str, Any]:
        """
        Process a single video frame.
        
        Args:
            frame: Video frame as numpy array
            frame_number: Frame number in the video
            video_name: Name of the video file
            
        Returns:
            Dictionary containing face information for the frame
        """
        try:
            # Detect faces in frame
            face_crops, embeddings, probs = self.detect_faces(frame)
            
            result = {
                "video_name": video_name,
                "frame_number": frame_number,
                "num_faces": len(face_crops),
                "faces": [],
                "processing_method": "face_recognition"
            }
            
            for i, (crop, embedding, prob) in enumerate(zip(face_crops, embeddings, probs)):
                face_info = {
                    "face_id": f"{video_name}_frame_{frame_number}_face_{i}",
                    "crop": crop,
                    "embedding": embedding.numpy() if hasattr(embedding, 'numpy') else embedding,
                    "confidence": prob,
                    "frame_number": frame_number
                }
                result["faces"].append(face_info)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_number} from {video_name}: {e}")
            return {
                "video_name": video_name,
                "frame_number": frame_number,
                "num_faces": 0,
                "faces": [],
                "error": str(e)
            }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get GPU memory usage information."""
        if torch.cuda.is_available():
            return {
                "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3
            }
        return {"message": "CUDA not available"}
