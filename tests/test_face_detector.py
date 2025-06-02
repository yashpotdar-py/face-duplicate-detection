"""
Tests for face detection module.
"""
import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch
from pathlib import Path

from src.face_detector import FaceDetector
from src.models import DetectionConfig, BoundingBox


@pytest.fixture
def config():
    """Create test configuration."""
    return DetectionConfig(
        min_confidence=0.5,
        match_threshold=0.6,
        batch_size=4,
        device="cpu"  # Use CPU for tests
    )


@pytest.fixture
def face_detector(config):
    """Create face detector instance."""
    with patch('src.face_detector.MTCNN'), patch('src.face_detector.InceptionResnetV1'):
        detector = FaceDetector(config)
        return detector


@pytest.fixture
def sample_frame():
    """Create a sample video frame."""
    # Create a 640x480 RGB image
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return frame


class TestFaceDetector:
    """Test suite for FaceDetector class."""
    
    def test_initialization(self, config):
        """Test face detector initialization."""
        with patch('src.face_detector.MTCNN') as mock_mtcnn, \
             patch('src.face_detector.InceptionResnetV1') as mock_facenet:
            
            detector = FaceDetector(config)
            
            # Check that models were initialized
            mock_mtcnn.assert_called_once()
            mock_facenet.assert_called_once()
            
            assert detector.config == config
            assert detector.device.type == "cpu"
    
    def test_detect_faces_success(self, face_detector, sample_frame):
        """Test successful face detection."""
        # Mock MTCNN detection results
        mock_boxes = np.array([[100, 150, 200, 250]])  # x1, y1, x2, y2
        mock_probs = np.array([0.95])
        
        face_detector.mtcnn.detect = Mock(return_value=(mock_boxes, mock_probs))
        
        detections = face_detector.detect_faces(sample_frame)
        
        assert len(detections) == 1
        bbox, confidence = detections[0]
        
        assert isinstance(bbox, BoundingBox)
        assert bbox.x == 100
        assert bbox.y == 150
        assert bbox.width == 100  # x2 - x1
        assert bbox.height == 100  # y2 - y1
        assert confidence == 0.95
    
    def test_detect_faces_no_faces(self, face_detector, sample_frame):
        """Test face detection when no faces are found."""
        face_detector.mtcnn.detect = Mock(return_value=(None, None))
        
        detections = face_detector.detect_faces(sample_frame)
        
        assert len(detections) == 0
    
    def test_detect_faces_low_confidence(self, face_detector, sample_frame):
        """Test face detection with low confidence faces."""
        mock_boxes = np.array([[100, 150, 200, 250]])
        mock_probs = np.array([0.3])  # Below threshold
        
        face_detector.mtcnn.detect = Mock(return_value=(mock_boxes, mock_probs))
        
        detections = face_detector.detect_faces(sample_frame)
        
        assert len(detections) == 0  # Should be filtered out
    
    def test_extract_embeddings_success(self, face_detector, sample_frame):
        """Test successful embedding extraction."""
        # Create sample bounding boxes
        bboxes = [
            BoundingBox(x=100, y=150, width=100, height=100),
            BoundingBox(x=300, y=200, width=80, height=80)
        ]
        
        # Mock FaceNet output
        mock_embeddings = np.random.randn(2, 512).astype(np.float32)
        face_detector.facenet = Mock(return_value=mock_embeddings)
        
        embeddings = face_detector.extract_embeddings(sample_frame, bboxes)
        
        assert len(embeddings) == 2
        assert all(emb.shape == (512,) for emb in embeddings)
    
    def test_extract_embeddings_empty_faces(self, face_detector, sample_frame):
        """Test embedding extraction with no faces."""
        embeddings = face_detector.extract_embeddings(sample_frame, [])
        
        assert len(embeddings) == 0
    
    def test_detect_and_extract_integration(self, face_detector, sample_frame):
        """Test integrated detection and embedding extraction."""
        # Mock MTCNN detection
        mock_boxes = np.array([[100, 150, 200, 250]])
        mock_probs = np.array([0.95])
        face_detector.mtcnn.detect = Mock(return_value=(mock_boxes, mock_probs))
        
        # Mock FaceNet embedding
        mock_embeddings = np.random.randn(1, 512).astype(np.float32)
        face_detector.facenet = Mock(return_value=mock_embeddings)
        
        results = face_detector.detect_and_extract(sample_frame)
        
        assert len(results) == 1
        bbox, confidence, embedding = results[0]
        
        assert isinstance(bbox, BoundingBox)
        assert confidence == 0.95
        assert embedding.shape == (512,)
    
    def test_error_handling(self, face_detector, sample_frame):
        """Test error handling in face detection."""
        # Mock MTCNN to raise an exception
        face_detector.mtcnn.detect = Mock(side_effect=RuntimeError("GPU error"))
        
        detections = face_detector.detect_faces(sample_frame)
        
        # Should return empty list on error
        assert len(detections) == 0
