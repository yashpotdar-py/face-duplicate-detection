"""
Face detection module using GPU-accelerated models.
"""
import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional
from pathlib import Path
from facenet_pytorch import MTCNN, InceptionResnetV1
from loguru import logger
import face_recognition
from PIL import Image

from .models import BoundingBox, DetectionConfig


class FaceDetector:
    """GPU-accelerated face detection and embedding extraction."""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize MTCNN for face detection (GPU accelerated)
        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self.device,
            keep_all=True
        )
        
        # Initialize FaceNet for embeddings (GPU accelerated)
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        logger.info("Face detection models loaded successfully")
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[BoundingBox, float]]:
        """
        Detect faces in a single frame.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            
        Returns:
            List of tuples containing (BoundingBox, confidence_score)
        """
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Detect faces using MTCNN
            boxes, probs = self.mtcnn.detect(pil_image)
            
            detections = []
            if boxes is not None and probs is not None:
                for box, prob in zip(boxes, probs):
                    if prob >= self.config.min_confidence:
                        x, y, x2, y2 = box.astype(int)
                        bbox = BoundingBox(
                            x=max(0, x),
                            y=max(0, y),
                            width=max(1, x2 - x),
                            height=max(1, y2 - y)
                        )
                        detections.append((bbox, float(prob)))
            
            return detections
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    def extract_embeddings(self, frame: np.ndarray, faces: List[BoundingBox]) -> List[np.ndarray]:
        """
        Extract face embeddings for detected faces.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            faces: List of face bounding boxes
            
        Returns:
            List of 512-dimensional embeddings
        """
        embeddings = []
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            face_tensors = []
            valid_faces = []
            
            # Extract and preprocess face regions
            for bbox in faces:
                try:
                    # Crop face region with some padding
                    x1 = max(0, bbox.x - 10)
                    y1 = max(0, bbox.y - 10)
                    x2 = min(pil_image.width, bbox.x + bbox.width + 10)
                    y2 = min(pil_image.height, bbox.y + bbox.height + 10)
                    
                    face_crop = pil_image.crop((x1, y1, x2, y2))
                    
                    # Resize to 160x160 for FaceNet
                    face_resized = face_crop.resize((160, 160), Image.LANCZOS)
                    
                    # Convert to tensor and normalize
                    face_tensor = torch.tensor(np.array(face_resized)).permute(2, 0, 1).float()
                    face_tensor = (face_tensor - 127.5) / 128.0
                    
                    face_tensors.append(face_tensor)
                    valid_faces.append(bbox)
                    
                except Exception as e:
                    logger.warning(f"Failed to process face at {bbox}: {e}")
                    continue
            
            if face_tensors:
                # Batch process embeddings on GPU
                batch_tensor = torch.stack(face_tensors).to(self.device)
                
                with torch.no_grad():
                    batch_embeddings = self.facenet(batch_tensor)
                    batch_embeddings = batch_embeddings.cpu().numpy()
                
                embeddings = [emb for emb in batch_embeddings]
            
        except Exception as e:
            logger.error(f"Error extracting embeddings: {e}")
        
        return embeddings
    
    def detect_and_extract(self, frame: np.ndarray) -> List[Tuple[BoundingBox, float, np.ndarray]]:
        """
        Detect faces and extract embeddings in one step.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            
        Returns:
            List of tuples containing (BoundingBox, confidence, embedding)
        """
        # Detect faces
        face_detections = self.detect_faces(frame)
        
        if not face_detections:
            return []
        
        # Extract bounding boxes
        bboxes = [bbox for bbox, _ in face_detections]
        confidences = [conf for _, conf in face_detections]
        
        # Extract embeddings
        embeddings = self.extract_embeddings(frame, bboxes)
        
        # Combine results
        results = []
        for i, (bbox, conf) in enumerate(face_detections):
            if i < len(embeddings):
                results.append((bbox, conf, embeddings[i]))
        
        return results
