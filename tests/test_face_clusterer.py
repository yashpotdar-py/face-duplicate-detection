"""
Tests for face clustering module.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch
from collections import defaultdict

from src.face_clusterer import FaceClusterer
from src.models import DetectionConfig, FaceDetection, BoundingBox


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
def face_clusterer(config):
    """Create face clusterer instance."""
    return FaceClusterer(config)


@pytest.fixture
def sample_detections():
    """Create sample face detections with embeddings."""
    detections = []
    
    # Create 3 groups of similar faces
    for group_id in range(3):
        for face_id in range(2):  # 2 faces per group
            # Create similar embeddings within each group
            base_embedding = np.random.randn(512)
            if group_id == 0:
                embedding = base_embedding + np.random.randn(512) * 0.1  # Very similar
            elif group_id == 1:
                embedding = base_embedding + np.random.randn(512) * 0.2  # Somewhat similar
            else:
                embedding = np.random.randn(512)  # Random/unique
            
            detection = FaceDetection(
                face_id=f"temp_face_{group_id}_{face_id}",
                video_filename=f"video_{group_id}.mp4",
                timestamp=f"00:00:{face_id:02d}.000",
                bounding_box=BoundingBox(x=100, y=150, width=80, height=100),
                confidence=0.95,
                embedding=embedding.tolist()
            )
            detections.append(detection)
    
    return detections


@pytest.fixture
def detections_no_embeddings():
    """Create sample detections without embeddings."""
    detections = []
    for i in range(3):
        detection = FaceDetection(
            face_id=f"temp_face_{i}",
            video_filename=f"video_{i}.mp4",
            timestamp=f"00:00:{i:02d}.000",
            bounding_box=BoundingBox(x=100, y=150, width=80, height=100),
            confidence=0.95,
            embedding=None
        )
        detections.append(detection)
    
    return detections


class TestFaceClusterer:
    """Test suite for FaceClusterer class."""
    
    def test_initialization(self, config):
        """Test face clusterer initialization."""
        clusterer = FaceClusterer(config)
        
        assert clusterer.config == config
        assert clusterer.match_threshold == config.match_threshold
    
    def test_compute_embedding_distance(self, face_clusterer):
        """Test embedding distance computation."""
        # Create test embeddings
        emb1 = np.array([1.0, 0.0, 0.0, 0.0])  # Normalized for testing
        emb2 = np.array([1.0, 0.0, 0.0, 0.0])  # Identical
        emb3 = np.array([0.0, 1.0, 0.0, 0.0])  # Orthogonal
        emb4 = np.array([-1.0, 0.0, 0.0, 0.0])  # Opposite
        
        # Test identical embeddings
        distance = face_clusterer.compute_embedding_distance(emb1, emb2)
        assert abs(distance) < 1e-6  # Should be ~0
        
        # Test orthogonal embeddings
        distance = face_clusterer.compute_embedding_distance(emb1, emb3)
        assert abs(distance - 1.0) < 1e-6  # Should be ~1
        
        # Test opposite embeddings  
        distance = face_clusterer.compute_embedding_distance(emb1, emb4)
        assert abs(distance - 2.0) < 1e-6  # Should be ~2
    
    @patch('sklearn.cluster.DBSCAN')
    def test_find_duplicate_faces_success(self, mock_dbscan_class, face_clusterer, sample_detections):
        """Test successful face clustering."""
        # Mock DBSCAN
        mock_dbscan = Mock()
        # Simulate 2 clusters: [0, 0, 1, 1, -1, -1] (2 faces each in clusters 0,1, 2 noise)
        mock_dbscan.fit_predict.return_value = np.array([0, 0, 1, 1, -1, -1])
        mock_dbscan_class.return_value = mock_dbscan
        
        clustered_detections = face_clusterer.find_duplicate_faces(sample_detections)
        
        assert len(clustered_detections) == len(sample_detections)
        
        # Check that DBSCAN was called
        mock_dbscan_class.assert_called_once()
        mock_dbscan.fit_predict.assert_called_once()
        
        # Check face ID assignment
        face_ids = [d.face_id for d in clustered_detections]
        
        # First two should have same face_id (cluster 0)
        assert face_ids[0] == face_ids[1]
        
        # Next two should have same face_id (cluster 1)  
        assert face_ids[2] == face_ids[3]
        
        # Last two should have different face_ids (noise)
        assert face_ids[4] != face_ids[5]
        assert face_ids[0] != face_ids[2]  # Different clusters
    
    def test_find_duplicate_faces_no_embeddings(self, face_clusterer, detections_no_embeddings):
        """Test clustering when no embeddings are available."""
        clustered_detections = face_clusterer.find_duplicate_faces(detections_no_embeddings)
        
        # Should return input detections with updated face IDs
        assert len(clustered_detections) == len(detections_no_embeddings)
        
        # All should have unique face IDs starting with "unknown_"
        face_ids = [d.face_id for d in clustered_detections]
        assert all(fid.startswith("unknown_") for fid in face_ids)
        assert len(set(face_ids)) == len(face_ids)  # All unique
    
    def test_find_duplicate_faces_insufficient_data(self, face_clusterer):
        """Test clustering with insufficient data."""
        # Only one detection
        single_detection = FaceDetection(
            face_id="temp_face",
            video_filename="video.mp4",
            timestamp="00:00:01.000",
            bounding_box=BoundingBox(x=100, y=150, width=80, height=100),
            confidence=0.95,
            embedding=np.random.randn(512).tolist()
        )
        
        result = face_clusterer.find_duplicate_faces([single_detection])
        
        # Should return original detection
        assert len(result) == 1
        assert result[0] == single_detection
    
    def test_assign_face_ids(self, face_clusterer, sample_detections):
        """Test face ID assignment based on cluster labels."""
        # Mock cluster labels: 2 clusters + noise
        cluster_labels = np.array([0, 0, 1, 1, -1, -1])
        
        # Take subset of detections
        detections = sample_detections[:6]
        
        updated_detections = face_clusterer._assign_face_ids(detections, cluster_labels)
        
        assert len(updated_detections) == 6
        
        # Check cluster assignments
        face_ids = [d.face_id for d in updated_detections]
        
        # Cluster 0 members should have same ID
        assert face_ids[0] == face_ids[1]
        assert face_ids[0].startswith("face_")
        
        # Cluster 1 members should have same ID
        assert face_ids[2] == face_ids[3]
        assert face_ids[2].startswith("face_")
        
        # Noise points should have unique IDs
        assert face_ids[4] != face_ids[5]
        assert face_ids[4].startswith("unique_")
        assert face_ids[5].startswith("unique_")
        
        # Different clusters should have different IDs
        assert face_ids[0] != face_ids[2]
    
    @patch('src.face_clusterer.logger')
    def test_log_cluster_summary(self, mock_logger, face_clusterer):
        """Test cluster summary logging."""
        # Create detections with known face IDs
        detections = []
        
        # Face 1 appears 3 times
        for i in range(3):
            detections.append(FaceDetection(
                face_id="face_duplicate_1",
                video_filename=f"video_{i}.mp4",
                timestamp=f"00:00:{i:02d}.000",
                bounding_box=BoundingBox(x=100, y=150, width=80, height=100),
                confidence=0.95
            ))
        
        # Face 2 appears 2 times
        for i in range(2):
            detections.append(FaceDetection(
                face_id="face_duplicate_2", 
                video_filename=f"video_{i+3}.mp4",
                timestamp=f"00:00:{i+3:02d}.000",
                bounding_box=BoundingBox(x=200, y=250, width=80, height=100),
                confidence=0.95
            ))
        
        # Unique face
        detections.append(FaceDetection(
            face_id="face_unique_1",
            video_filename="video_5.mp4",
            timestamp="00:00:05.000",
            bounding_box=BoundingBox(x=300, y=350, width=80, height=100),
            confidence=0.95
        ))
        
        face_clusterer._log_cluster_summary(detections)
        
        # Check that logging was called
        assert mock_logger.info.call_count >= 3  # At least summary info
        
        # Check log content
        log_calls = [call.args[0] for call in mock_logger.info.call_calls]
        summary_logs = [log for log in log_calls if "Total detections:" in log]
        assert len(summary_logs) > 0
    
    def test_validate_clusters_success(self, face_clusterer):
        """Test cluster validation."""
        # Create detections with known clusters
        detections = []
        
        # Cluster 1: Similar embeddings
        base_emb1 = np.random.randn(512)
        for i in range(3):
            embedding = base_emb1 + np.random.randn(512) * 0.1  # Small noise
            detections.append(FaceDetection(
                face_id="face_cluster_1",
                video_filename=f"video_{i}.mp4",
                timestamp=f"00:00:{i:02d}.000",
                bounding_box=BoundingBox(x=100, y=150, width=80, height=100),
                confidence=0.95,
                embedding=embedding.tolist()
            ))
        
        # Cluster 2: Different embeddings
        base_emb2 = np.random.randn(512) * 2  # Larger scale
        for i in range(2):
            embedding = base_emb2 + np.random.randn(512) * 0.1
            detections.append(FaceDetection(
                face_id="face_cluster_2",
                video_filename=f"video_{i+3}.mp4",
                timestamp=f"00:00:{i+3:02d}.000",
                bounding_box=BoundingBox(x=200, y=250, width=80, height=100),
                confidence=0.95,
                embedding=embedding.tolist()
            ))
        
        metrics = face_clusterer.validate_clusters(detections, sample_size=10)
        
        # Should compute basic metrics
        assert 'avg_intra_cluster_distance' in metrics
        assert 'avg_inter_cluster_distance' in metrics
        
        # Intra-cluster distance should be smaller than inter-cluster
        if 'avg_intra_cluster_distance' in metrics and 'avg_inter_cluster_distance' in metrics:
            assert metrics['avg_intra_cluster_distance'] < metrics['avg_inter_cluster_distance']
    
    def test_validate_clusters_no_embeddings(self, face_clusterer, detections_no_embeddings):
        """Test cluster validation with no embeddings."""
        metrics = face_clusterer.validate_clusters(detections_no_embeddings)
        
        # Should return empty or minimal metrics
        assert isinstance(metrics, dict)
