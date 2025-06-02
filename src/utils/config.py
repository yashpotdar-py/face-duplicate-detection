"""
Configuration module for face duplicate detection system.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import torch


@dataclass
class Config:
    """Configuration class for the face detection system."""
    
    # Paths
    data_dir: str = "data"
    thumbnails_dir: str = "data/Thumbnails"
    videos_dir: str = "data/Videos"
    results_dir: str = "results"
    
    # Face detection parameters
    face_detection_model: str = "mtcnn"  # or "dlib", "opencv"
    face_recognition_model: str = "facenet"  # or "dlib"
    similarity_threshold: float = 0.6
    face_size: tuple = (160, 160)
    
    # GPU settings
    use_gpu: bool = torch.cuda.is_available()
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    
    # Video processing
    video_frame_skip: int = 30  # Process every Nth frame
    max_video_duration: int = 300  # seconds
    
    # Performance
    num_workers: int = 4
    max_faces_per_image: int = 50
    
    # Output
    save_face_crops: bool = True
    save_duplicate_pairs: bool = True
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Validate and setup configuration after initialization."""
        # Create directories if they don't exist
        for dir_path in [self.results_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Validate paths
        if not Path(self.data_dir).exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create Config instance from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Config instance to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get information about the computing device."""
        device_info = {
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        if torch.cuda.is_available():
            device_info.update({
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(),
                "memory_allocated": torch.cuda.memory_allocated(),
                "memory_reserved": torch.cuda.memory_reserved()
            })
        
        return device_info
