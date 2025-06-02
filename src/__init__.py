"""
Package initialization for face duplicate detection system.
"""
from .models import DetectionConfig, FaceDetection, DuplicateReport, BoundingBox
from .face_detector import FaceDetector
from .video_processor import VideoProcessor
from .face_clusterer import FaceClusterer
from .detector import DuplicateFaceDetector
from .image_processor import ImageProcessor, ThumbnailDuplicateDetector

__version__ = "1.0.0"
__author__ = "Face Duplicate Detection System"

__all__ = [
    "DetectionConfig",
    "FaceDetection", 
    "DuplicateReport",
    "BoundingBox",
    "FaceDetector",
    "VideoProcessor",
    "FaceClusterer",
    "DuplicateFaceDetector",
    "ImageProcessor",
    "ThumbnailDuplicateDetector"
]
