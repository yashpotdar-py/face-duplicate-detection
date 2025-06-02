"""
Tests for video processing module.
"""
import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

from src.video_processor import VideoProcessor
from src.models import DetectionConfig, FaceDetection, BoundingBox


@pytest.fixture
def config():
    """Create test configuration."""
    return DetectionConfig(
        min_confidence=0.5,
        match_threshold=0.6,
        batch_size=2,
        device="cpu"
    )


@pytest.fixture
def video_processor(config):
    """Create video processor instance."""
    with patch('src.video_processor.FaceDetector'):
        processor = VideoProcessor(config)
        return processor


@pytest.fixture
def mock_video_file():
    """Create a mock video file."""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        video_path = Path(f.name)
    
    yield video_path
    
    # Cleanup
    if video_path.exists():
        os.unlink(video_path)


class TestVideoProcessor:
    """Test suite for VideoProcessor class."""
    
    def test_initialization(self, config):
        """Test video processor initialization."""
        with patch('src.video_processor.FaceDetector') as mock_detector:
            processor = VideoProcessor(config)
            
            mock_detector.assert_called_once_with(config)
            assert processor.config == config
    
    @patch('cv2.VideoCapture')
    def test_get_video_info_success(self, mock_cap_class, video_processor, mock_video_file):
        """Test successful video info extraction."""
        # Mock VideoCapture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 1000.0,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_WIDTH: 1920.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080.0
        }.get(prop, 0.0)
        mock_cap_class.return_value = mock_cap
        
        total_frames, fps, (width, height) = video_processor.get_video_info(mock_video_file)
        
        assert total_frames == 1000
        assert fps == 30.0
        assert width == 1920
        assert height == 1080
        mock_cap.release.assert_called_once()
    
    @patch('cv2.VideoCapture')
    def test_get_video_info_cannot_open(self, mock_cap_class, video_processor, mock_video_file):
        """Test video info extraction when video cannot be opened."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_cap_class.return_value = mock_cap
        
        with pytest.raises(ValueError, match="Cannot open video"):
            video_processor.get_video_info(mock_video_file)
    
    @patch('cv2.VideoCapture')
    def test_frame_generator_success(self, mock_cap_class, video_processor, mock_video_file):
        """Test successful frame generation."""
        # Mock VideoCapture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        
        # Mock frame reading
        frames = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (True, np.ones((480, 640, 3), dtype=np.uint8)),
            (True, np.full((480, 640, 3), 255, dtype=np.uint8)),
            (False, None)  # End of video
        ]
        mock_cap.read.side_effect = frames
        mock_cap_class.return_value = mock_cap
        
        # Generate frames
        generated_frames = list(video_processor.frame_generator(mock_video_file, skip_frames=1))
        
        assert len(generated_frames) == 3
        assert all(frame_num == i for i, (frame_num, _) in enumerate(generated_frames))
        mock_cap.release.assert_called_once()
    
    @patch('cv2.VideoCapture')
    def test_frame_generator_skip_frames(self, mock_cap_class, video_processor, mock_video_file):
        """Test frame generation with frame skipping."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        
        # Mock 6 frames
        frames = [(True, np.zeros((480, 640, 3), dtype=np.uint8))] * 6 + [(False, None)]
        mock_cap.read.side_effect = frames
        mock_cap_class.return_value = mock_cap
        
        # Skip every other frame
        generated_frames = list(video_processor.frame_generator(mock_video_file, skip_frames=2))
        
        # Should get frames 0, 2, 4
        assert len(generated_frames) == 3
        frame_numbers = [frame_num for frame_num, _ in generated_frames]
        assert frame_numbers == [0, 2, 4]
    
    def test_timestamp_from_frame(self, video_processor):
        """Test timestamp calculation from frame number."""
        fps = 30.0
        
        # Test various frame numbers
        assert video_processor.timestamp_from_frame(0, fps) == "00:00:00.000"
        assert video_processor.timestamp_from_frame(30, fps) == "00:00:01.000"
        assert video_processor.timestamp_from_frame(1800, fps) == "00:01:00.000"
        assert video_processor.timestamp_from_frame(3630, fps) == "00:02:01.000"
        assert video_processor.timestamp_from_frame(108000, fps) == "01:00:00.000"
    
    @patch('src.video_processor.VideoProcessor.get_video_info')
    @patch('src.video_processor.VideoProcessor.frame_generator')
    def test_process_video_success(self, mock_frame_gen, mock_video_info, video_processor, mock_video_file):
        """Test successful video processing."""
        # Mock video info
        mock_video_info.return_value = (100, 30.0, (640, 480))
        
        # Mock frame generator
        sample_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_frame_gen.return_value = [(0, sample_frame), (30, sample_frame)]
        
        # Mock face detector
        mock_bbox = BoundingBox(x=100, y=150, width=80, height=100)
        mock_embedding = np.random.randn(512)
        video_processor.face_detector.detect_and_extract = Mock(
            return_value=[(mock_bbox, 0.95, mock_embedding)]
        )
        
        detections = video_processor.process_video(mock_video_file, skip_frames=30)
        
        assert len(detections) == 2  # 2 frames processed
        
        # Check first detection
        detection = detections[0]
        assert detection.video_filename == mock_video_file.name
        assert detection.timestamp == "00:00:00.000"
        assert detection.bounding_box == mock_bbox
        assert detection.confidence == 0.95
        assert len(detection.embedding) == 512
    
    @patch('src.video_processor.VideoProcessor.get_video_info')
    def test_process_video_error_handling(self, mock_video_info, video_processor, mock_video_file):
        """Test video processing error handling."""
        # Mock video info to raise exception
        mock_video_info.side_effect = RuntimeError("Video error")
        
        detections = video_processor.process_video(mock_video_file)
        
        # Should return empty list on error
        assert len(detections) == 0
    
    def test_process_frame_batch(self, video_processor):
        """Test batch processing of frames."""
        # Create sample frames and frame numbers
        frames = [
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.ones((480, 640, 3), dtype=np.uint8)
        ]
        frame_numbers = [0, 30]
        fps = 30.0
        video_filename = "test_video.mp4"
        
        # Mock face detector
        mock_bbox = BoundingBox(x=100, y=150, width=80, height=100)
        mock_embedding = np.random.randn(512)
        video_processor.face_detector.detect_and_extract = Mock(
            return_value=[(mock_bbox, 0.95, mock_embedding)]
        )
        
        detections = video_processor._process_frame_batch(
            frames, frame_numbers, fps, video_filename
        )
        
        assert len(detections) == 2  # One detection per frame
        
        # Check timestamps
        assert detections[0].timestamp == "00:00:00.000"
        assert detections[1].timestamp == "00:00:01.000"
    
    @patch('src.video_processor.VideoProcessor.process_video')
    def test_process_videos_success(self, mock_process_video, video_processor):
        """Test processing multiple videos in a directory."""
        # Create temporary directory with mock video files
        with tempfile.TemporaryDirectory() as temp_dir:
            video_dir = Path(temp_dir)
            
            # Create mock video files
            video1 = video_dir / "video1.mp4"
            video2 = video_dir / "video2.avi"
            non_video = video_dir / "readme.txt"
            
            video1.touch()
            video2.touch()
            non_video.touch()
            
            # Mock process_video to return detections
            mock_detection = FaceDetection(
                face_id="test_face",
                video_filename="test.mp4",
                timestamp="00:00:01.000",
                bounding_box=BoundingBox(x=100, y=150, width=80, height=100),
                confidence=0.95
            )
            mock_process_video.return_value = [mock_detection]
            
            all_detections = video_processor.process_videos(video_dir)
            
            # Should process 2 video files
            assert mock_process_video.call_count == 2
            assert len(all_detections) == 2
    
    def test_process_videos_no_videos(self, video_processor):
        """Test processing directory with no video files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            video_dir = Path(temp_dir)
            
            # Create only non-video files
            (video_dir / "readme.txt").touch()
            (video_dir / "image.jpg").touch()
            
            all_detections = video_processor.process_videos(video_dir)
            
            assert len(all_detections) == 0
