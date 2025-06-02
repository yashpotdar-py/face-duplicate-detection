"""
Tests for the image processor module.
"""
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import cv2
from PIL import Image

from src.models import DetectionConfig, FaceDetection, BoundingBox
from src.image_processor import ImageProcessor, ThumbnailDuplicateDetector


class TestImageProcessor(unittest.TestCase):
    """Test cases for ImageProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = DetectionConfig(
            min_confidence=0.5,
            match_threshold=0.6,
            batch_size=4,
            device="cpu"
        )
        
        # Create temporary directory for test images
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create sample test images
        self.test_images = []
        for i in range(3):
            img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
            img_path = self.temp_dir / f"test_image_{i}.jpg"
            cv2.imwrite(str(img_path), img)
            self.test_images.append(img_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test ImageProcessor initialization."""
        processor = ImageProcessor(self.config)
        
        self.assertEqual(processor.config, self.config)
        self.assertIsNotNone(processor.face_detector)
        self.assertIsNotNone(processor.face_clusterer)
        self.assertIn('.jpg', processor.image_extensions)
        self.assertIn('.png', processor.image_extensions)
    
    def test_find_image_files(self):
        """Test finding image files in directory."""
        processor = ImageProcessor(self.config)
        
        # Test with existing directory
        image_files = processor.find_image_files(self.temp_dir)
        self.assertEqual(len(image_files), 3)
        
        # Test with non-existent directory
        fake_dir = Path("non_existent_dir")
        image_files = processor.find_image_files(fake_dir)
        self.assertEqual(len(image_files), 0)
    
    def test_load_and_preprocess_image(self):
        """Test image loading and preprocessing."""
        processor = ImageProcessor(self.config)
        
        # Test with valid image
        image = processor.load_and_preprocess_image(self.test_images[0])
        self.assertIsNotNone(image)
        self.assertEqual(len(image.shape), 3)  # Height, width, channels
        
        # Test with non-existent image
        fake_path = Path("non_existent_image.jpg")
        image = processor.load_and_preprocess_image(fake_path)
        self.assertIsNone(image)
    
    @patch('src.image_processor.ImageProcessor.process_single_image')
    def test_process_images_batch(self, mock_process_single):
        """Test batch processing of images."""
        processor = ImageProcessor(self.config)
        
        # Mock the single image processing
        mock_detection = FaceDetection(
            face_id="test_face_1",
            video_filename="test.jpg",
            timestamp="00:00:00.000",
            bounding_box=BoundingBox(x=10, y=10, width=50, height=60),
            confidence=0.8,
            embedding=[0.1] * 512
        )
        mock_process_single.return_value = [mock_detection]
        
        # Test batch processing
        batch_results = processor.process_images_batch(self.test_images)
        
        self.assertEqual(len(batch_results), 3)  # 3 images, 1 detection each
        self.assertEqual(mock_process_single.call_count, 3)
    
    @patch('src.image_processor.FaceDetector')
    @patch('src.image_processor.FaceClusterer')
    def test_detect_duplicate_faces_in_images(self, mock_clusterer_class, mock_detector_class):
        """Test complete duplicate detection pipeline."""
        # Setup mocks
        mock_detector = Mock()
        mock_clusterer = Mock()
        mock_detector_class.return_value = mock_detector
        mock_clusterer_class.return_value = mock_clusterer
        
        # Mock face detection
        mock_bbox = BoundingBox(x=10, y=10, width=50, height=60)
        mock_detector.detect_faces.return_value = [(mock_bbox, 0.8)]
        mock_detector.extract_embeddings.return_value = [np.array([0.1] * 512)]
        
        # Mock clustering
        mock_detection = FaceDetection(
            face_id="clustered_face_1",
            video_filename="test.jpg",
            timestamp="00:00:00.000",
            bounding_box=mock_bbox,
            confidence=0.8,
            embedding=[0.1] * 512
        )
        mock_clusterer.find_duplicate_faces.return_value = [mock_detection]
        
        processor = ImageProcessor(self.config)
        
        # Run detection
        report = processor.detect_duplicate_faces_in_images(self.temp_dir)
        
        # Verify results
        self.assertEqual(report.total_faces, 1)
        self.assertEqual(report.unique_faces, 1)
        self.assertEqual(report.duplicate_groups, 0)
        self.assertGreater(report.processing_time, 0)
    
    def test_find_duplicates_between_videos_and_images(self):
        """Test cross-medium duplicate detection."""
        processor = ImageProcessor(self.config)
        
        # Create sample detections
        video_detection = FaceDetection(
            face_id="temp_video_1",
            video_filename="video.mp4",
            timestamp="00:00:05.000",
            bounding_box=BoundingBox(x=10, y=10, width=50, height=60),
            confidence=0.8,
            embedding=[0.1] * 512
        )
        
        image_detection = FaceDetection(
            face_id="temp_image_1",
            video_filename="image.jpg",
            timestamp="00:00:00.000",
            bounding_box=BoundingBox(x=15, y=15, width=55, height=65),
            confidence=0.7,
            embedding=[0.1] * 512
        )
        
        # Mock the clustering to return same face_id (indicating duplicates)
        with patch.object(processor.face_clusterer, 'find_duplicate_faces') as mock_cluster:
            # Make both detections have the same face_id after clustering
            video_detection.face_id = "person_1"
            image_detection.face_id = "person_1"
            mock_cluster.return_value = [video_detection, image_detection]
            
            cross_duplicates = processor.find_duplicates_between_videos_and_images(
                video_detections=[video_detection],
                image_detections=[image_detection]
            )
            
            # Should find one cross-medium duplicate group
            self.assertEqual(len(cross_duplicates), 1)
            self.assertIn("person_1", cross_duplicates)
            self.assertEqual(len(cross_duplicates["person_1"]), 2)
    
    def test_generate_image_similarity_matrix(self):
        """Test similarity matrix generation."""
        processor = ImageProcessor(self.config)
        
        # Create test detections with embeddings
        detections = [
            FaceDetection(
                face_id="face_1",
                video_filename="img1.jpg",
                timestamp="00:00:00.000",
                bounding_box=BoundingBox(x=10, y=10, width=50, height=60),
                confidence=0.8,
                embedding=[0.1] * 512
            ),
            FaceDetection(
                face_id="face_2",
                video_filename="img2.jpg",
                timestamp="00:00:00.000",
                bounding_box=BoundingBox(x=20, y=20, width=50, height=60),
                confidence=0.7,
                embedding=[0.2] * 512
            )
        ]
        
        # Generate similarity matrix
        similarity_matrix = processor.generate_image_similarity_matrix(detections)
        
        # Verify matrix properties
        self.assertEqual(similarity_matrix.shape, (2, 2))
        self.assertEqual(similarity_matrix[0, 0], 1.0)  # Self-similarity should be 1
        self.assertEqual(similarity_matrix[1, 1], 1.0)
        self.assertEqual(similarity_matrix[0, 1], similarity_matrix[1, 0])  # Symmetric
    
    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        processor = ImageProcessor(self.config)
        
        # Test empty directory
        empty_dir = self.temp_dir / "empty"
        empty_dir.mkdir()
        
        report = processor.detect_duplicate_faces_in_images(empty_dir)
        self.assertEqual(report.total_faces, 0)
        self.assertEqual(report.unique_faces, 0)
        self.assertEqual(report.duplicate_groups, 0)
        
        # Test empty similarity matrix
        similarity_matrix = processor.generate_image_similarity_matrix([])
        self.assertEqual(similarity_matrix.size, 0)


class TestThumbnailDuplicateDetector(unittest.TestCase):
    """Test cases for ThumbnailDuplicateDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = DetectionConfig(device="cpu")
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create test image
        img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        self.test_image = self.temp_dir / "test.jpg"
        cv2.imwrite(str(self.test_image), img)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test ThumbnailDuplicateDetector initialization."""
        detector = ThumbnailDuplicateDetector(self.config)
        
        self.assertEqual(detector.config, self.config)
        self.assertIsNotNone(detector.image_processor)
    
    @patch('src.image_processor.ImageProcessor.detect_duplicate_faces_in_images')
    def test_run_thumbnail_detection(self, mock_detect):
        """Test running thumbnail detection."""
        detector = ThumbnailDuplicateDetector(self.config)
        
        # Mock the detection process
        from src.models import DuplicateReport
        mock_report = DuplicateReport(
            total_faces=2,
            unique_faces=2,
            duplicate_groups=0,
            detections=[],
            processing_time=1.5,
            config=self.config.__dict__
        )
        mock_detect.return_value = mock_report
        
        # Test without video comparison
        output_file = self.temp_dir / "results.json"
        report = detector.run_thumbnail_detection(
            thumbnails_dir=self.temp_dir,
            output_file=output_file,
            compare_with_videos=False
        )
        
        self.assertEqual(report.total_faces, 2)
        self.assertEqual(report.unique_faces, 2)
        mock_detect.assert_called_once()


if __name__ == '__main__':
    unittest.main()