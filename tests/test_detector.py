"""
Tests for the main detector module.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json

from src.detector import DuplicateFaceDetector
from src.models import DetectionConfig, FaceDetection, DuplicateReport, BoundingBox


@pytest.fixture
def config():
    """Create test configuration."""
    return DetectionConfig(
        min_confidence=0.5,
        match_threshold=0.6,
        batch_size=4,
        device="cpu"
    )


@pytest.fixture
def detector(config):
    """Create detector instance with mocked components."""
    with patch('src.detector.VideoProcessor'), \
         patch('src.detector.FaceClusterer'):
        detector = DuplicateFaceDetector(config)
        return detector


@pytest.fixture
def sample_detections():
    """Create sample face detections."""
    detections = []
    for i in range(5):
        detection = FaceDetection(
            face_id=f"face_{i}",
            video_filename=f"video_{i}.mp4",
            timestamp=f"00:00:{i:02d}.000",
            bounding_box=BoundingBox(x=100+i*10, y=150+i*10, width=80, height=100),
            confidence=0.95,
            embedding=np.random.randn(512).tolist()
        )
        detections.append(detection)
    
    return detections


class TestDuplicateFaceDetector:
    """Test suite for DuplicateFaceDetector class."""
    
    def test_initialization(self, config):
        """Test detector initialization."""
        with patch('src.detector.VideoProcessor') as mock_vp, \
             patch('src.detector.FaceClusterer') as mock_fc:
            
            detector = DuplicateFaceDetector(config)
            
            mock_vp.assert_called_once_with(config)
            mock_fc.assert_called_once_with(config)
            assert detector.config == config
    
    def test_process_videos_success(self, detector):
        """Test successful video processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            videos_dir = Path(temp_dir)
            
            # Create mock video files
            (videos_dir / "video1.mp4").touch()
            (videos_dir / "video2.avi").touch()
            
            # Mock video processor
            mock_detections = [Mock(), Mock(), Mock()]
            detector.video_processor.process_videos = Mock(return_value=mock_detections)
            
            result = detector.process_videos(videos_dir)
            
            assert result == mock_detections
            detector.video_processor.process_videos.assert_called_once_with(videos_dir, 30)
    
    def test_process_videos_directory_not_exists(self, detector):
        """Test processing non-existent directory."""
        non_existent_dir = Path("/non/existent/directory")
        
        with pytest.raises(FileNotFoundError):
            detector.process_videos(non_existent_dir)
    
    def test_process_videos_no_detections(self, detector):
        """Test processing when no faces are detected."""
        with tempfile.TemporaryDirectory() as temp_dir:
            videos_dir = Path(temp_dir)
            (videos_dir / "video1.mp4").touch()
            
            # Mock empty detections
            detector.video_processor.process_videos = Mock(return_value=[])
            
            result = detector.process_videos(videos_dir)
            
            assert result == []
    
    def test_find_duplicates_success(self, detector, sample_detections):
        """Test successful duplicate finding."""
        # Mock face clusterer
        clustered_detections = sample_detections.copy()
        # Simulate clustering by giving some faces the same ID
        clustered_detections[1].face_id = clustered_detections[0].face_id
        clustered_detections[3].face_id = clustered_detections[2].face_id
        
        detector.face_clusterer.find_duplicate_faces = Mock(return_value=clustered_detections)
        detector.face_clusterer.validate_clusters = Mock(return_value={'accuracy': 0.95})
        
        result = detector.find_duplicates(sample_detections)
        
        assert result == clustered_detections
        detector.face_clusterer.find_duplicate_faces.assert_called_once_with(sample_detections)
        detector.face_clusterer.validate_clusters.assert_called_once_with(clustered_detections)
    
    def test_find_duplicates_empty_input(self, detector):
        """Test duplicate finding with empty input."""
        result = detector.find_duplicates([])
        
        assert result == []
    
    def test_generate_report_success(self, detector, sample_detections):
        """Test successful report generation."""
        # Modify detections to have duplicates
        sample_detections[1].face_id = sample_detections[0].face_id  # Create duplicate
        processing_time = 123.45
        
        report = detector.generate_report(sample_detections, processing_time)
        
        assert isinstance(report, DuplicateReport)
        assert report.total_faces == 5
        assert report.unique_faces == 4  # 5 detections, 4 unique faces
        assert report.duplicate_groups == 1  # 1 group with duplicates
        assert report.processing_time == processing_time
        assert len(report.detections) == 5
        assert report.config['min_confidence'] == detector.config.min_confidence
    
    def test_generate_report_no_duplicates(self, detector, sample_detections):
        """Test report generation with no duplicates."""
        processing_time = 100.0
        
        report = detector.generate_report(sample_detections, processing_time)
        
        assert report.total_faces == 5
        assert report.unique_faces == 5  # All unique
        assert report.duplicate_groups == 0  # No duplicates
    
    @patch('time.time')
    def test_run_detection_full_pipeline(self, mock_time, detector):
        """Test the complete detection pipeline."""
        # Mock time progression
        mock_time.side_effect = [0.0, 100.0]  # Start and end times
        
        with tempfile.TemporaryDirectory() as temp_dir:
            videos_dir = Path(temp_dir)
            output_file = Path(temp_dir) / "results.json"
            
            # Create mock video files
            (videos_dir / "video1.mp4").touch()
            
            # Mock each pipeline step
            mock_detections = [Mock(face_id=f"face_{i}") for i in range(3)]
            detector.process_videos = Mock(return_value=mock_detections)
            
            clustered_detections = mock_detections.copy()
            detector.find_duplicates = Mock(return_value=clustered_detections)
            
            mock_report = Mock()
            mock_report.to_json = Mock()
            detector.generate_report = Mock(return_value=mock_report)
            
            result = detector.run_detection(videos_dir, output_file)
            
            # Verify pipeline execution
            detector.process_videos.assert_called_once_with(videos_dir, None, 30)
            detector.find_duplicates.assert_called_once_with(mock_detections)
            detector.generate_report.assert_called_once_with(clustered_detections, 100.0)
            mock_report.to_json.assert_called_once_with(output_file)
            
            assert result == mock_report
    
    @patch('time.time')
    def test_run_detection_csv_output(self, mock_time, detector):
        """Test pipeline with CSV output."""
        mock_time.side_effect = [0.0, 50.0]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            videos_dir = Path(temp_dir)
            output_file = Path(temp_dir) / "results.csv"
            
            (videos_dir / "video1.mp4").touch()
            
            # Mock pipeline
            detector.process_videos = Mock(return_value=[Mock()])
            detector.find_duplicates = Mock(return_value=[Mock()])
            
            mock_report = Mock()
            mock_report.to_csv = Mock()
            detector.generate_report = Mock(return_value=mock_report)
            
            detector.run_detection(videos_dir, output_file)
            
            mock_report.to_csv.assert_called_once_with(output_file)
    
    @patch('time.time')
    def test_run_detection_no_faces_found(self, mock_time, detector):
        """Test pipeline when no faces are detected."""
        mock_time.side_effect = [0.0, 10.0]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            videos_dir = Path(temp_dir)
            output_file = Path(temp_dir) / "results.json"
            
            (videos_dir / "video1.mp4").touch()
            
            # Mock empty detection
            detector.process_videos = Mock(return_value=[])
            
            result = detector.run_detection(videos_dir, output_file)
            
            # Should still generate report
            assert isinstance(result, DuplicateReport)
            assert result.total_faces == 0
    
    def test_run_detection_pipeline_failure(self, detector):
        """Test pipeline failure handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            videos_dir = Path(temp_dir)
            output_file = Path(temp_dir) / "results.json"
            
            # Mock process_videos to raise exception
            detector.process_videos = Mock(side_effect=RuntimeError("Processing failed"))
            
            with pytest.raises(RuntimeError):
                detector.run_detection(videos_dir, output_file)
