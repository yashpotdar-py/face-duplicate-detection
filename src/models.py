"""
Configuration and data models for the face duplicate detection system.
"""
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from pydantic import BaseModel
import json


@dataclass
class DetectionConfig:
    """Configuration for face detection and matching."""
    min_confidence: float = 0.5
    match_threshold: float = 0.6
    batch_size: int = 16
    log_level: str = "INFO"
    device: str = "cuda"  # cuda or cpu
    face_model: str = "facenet"  # facenet or dlib


class BoundingBox(BaseModel):
    """Bounding box coordinates."""
    x: int
    y: int
    width: int
    height: int
    
    def to_list(self) -> List[int]:
        return [self.x, self.y, self.width, self.height]


class FaceDetection(BaseModel):
    """Single face detection result."""
    face_id: str
    video_filename: str
    timestamp: str  # format: hh:mm:ss.ms
    bounding_box: BoundingBox
    confidence: float
    embedding: Optional[List[float]] = None
    
    class Config:
        arbitrary_types_allowed = True


class DuplicateReport(BaseModel):
    """Report containing all duplicate face detections."""
    total_faces: int
    unique_faces: int
    duplicate_groups: int
    detections: List[FaceDetection]
    processing_time: float
    config: Dict[str, Any]
    
    def to_json(self, output_path: Path) -> None:
        """Save report to JSON file."""
        data = self.dict()
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def to_csv(self, output_path: Path) -> None:
        """Save report to CSV file."""
        import pandas as pd
        
        rows = []
        for detection in self.detections:
            rows.append({
                'face_id': detection.face_id,
                'video_filename': detection.video_filename,
                'timestamp': detection.timestamp,
                'bounding_box': detection.bounding_box.to_list(),
                'confidence': detection.confidence
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
