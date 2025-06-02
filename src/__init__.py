"""
GPU-Accelerated Face Duplicate Detection System
=============================================

A production-grade Python application for detecting and identifying 
duplicate faces in both images and videos using GPU acceleration.

Author: Face Detection Team
Version: 1.0.0
"""

from .core.face_detector import FaceDetector
from .core.duplicate_finder import DuplicateFinder
from .utils.config import Config

__version__ = "1.0.0"
__all__ = ["FaceDetector", "DuplicateFinder", "Config"]
