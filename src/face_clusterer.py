"""
Face clustering module to identify duplicate faces across videos.
"""
import numpy as np
from typing import List, Dict, Set, Tuple
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger
import uuid
from collections import defaultdict

from .models import FaceDetection, DetectionConfig


class FaceClusterer:
    """Cluster faces to identify duplicates using embedding similarity."""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.match_threshold = config.match_threshold
    
    def compute_embedding_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine distance between two face embeddings.
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            
        Returns:
            Cosine distance (0 = identical, 2 = opposite)
        """
        # Reshape to 2D arrays for sklearn
        emb1 = emb1.reshape(1, -1)
        emb2 = emb2.reshape(1, -1)
        
        # Compute cosine similarity and convert to distance
        similarity = cosine_similarity(emb1, emb2)[0, 0]
        distance = 1 - similarity
        
        return distance
    
    def find_duplicate_faces(self, detections: List[FaceDetection]) -> List[FaceDetection]:
        """
        Cluster faces to identify duplicates and assign consistent face IDs.
        
        Args:
            detections: List of face detections with embeddings
            
        Returns:
            List of detections with updated face IDs for duplicates
        """
        logger.info(f"Clustering {len(detections)} face detections")
        
        # Filter detections that have embeddings
        valid_detections = [d for d in detections if d.embedding is not None]
        
        if len(valid_detections) < 2:
            logger.warning("Not enough valid detections for clustering")
            return detections
        
        logger.info(f"Using {len(valid_detections)} detections with embeddings")
        
        # Extract embeddings matrix
        embeddings_matrix = np.array([d.embedding for d in valid_detections])
        
        # Use DBSCAN clustering with cosine distance
        # Convert match_threshold to eps parameter for DBSCAN
        eps = self.match_threshold
        min_samples = 2  # Minimum 2 faces to form a cluster
        
        clusterer = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric='cosine'
        )
        
        cluster_labels = clusterer.fit_predict(embeddings_matrix)
        
        # Generate cluster statistics
        unique_clusters = set(cluster_labels)
        n_clusters = len(unique_clusters) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        logger.info(f"Found {n_clusters} face clusters, {n_noise} unique faces")
        
        # Assign face IDs based on clusters
        updated_detections = self._assign_face_ids(valid_detections, cluster_labels)
        
        # Add back detections without embeddings (if any)
        invalid_detections = [d for d in detections if d.embedding is None]
        for detection in invalid_detections:
            detection.face_id = f"unknown_{uuid.uuid4().hex[:8]}"
        
        all_detections = updated_detections + invalid_detections
        
        # Log cluster summary
        self._log_cluster_summary(all_detections)
        
        return all_detections
    
    def _assign_face_ids(
        self,
        detections: List[FaceDetection],
        cluster_labels: np.ndarray
    ) -> List[FaceDetection]:
        """
        Assign consistent face IDs based on cluster labels.
        
        Args:
            detections: List of face detections
            cluster_labels: Cluster labels from DBSCAN
            
        Returns:
            List of detections with updated face IDs
        """
        # Create mapping from cluster label to face ID
        cluster_to_face_id = {}
        
        for label in set(cluster_labels):
            if label == -1:
                # Noise points (unique faces) get individual IDs
                continue
            else:
                # Cluster gets a shared face ID
                cluster_to_face_id[label] = f"face_{uuid.uuid4().hex[:8]}"
        
        # Update face IDs
        for i, detection in enumerate(detections):
            cluster_label = cluster_labels[i]
            
            if cluster_label == -1:
                # Unique face
                detection.face_id = f"unique_{uuid.uuid4().hex[:8]}"
            else:
                # Duplicate face
                detection.face_id = cluster_to_face_id[cluster_label]
        
        return detections
    
    def _log_cluster_summary(self, detections: List[FaceDetection]) -> None:
        """Log summary of clustering results."""
        # Count faces per ID
        face_counts = defaultdict(int)
        video_counts = defaultdict(set)
        
        for detection in detections:
            face_counts[detection.face_id] += 1
            video_counts[detection.face_id].add(detection.video_filename)
        
        # Identify duplicate groups
        duplicate_groups = {
            face_id: count for face_id, count in face_counts.items()
            if count > 1
        }
        
        unique_faces = len(face_counts)
        total_detections = len(detections)
        duplicate_groups_count = len(duplicate_groups)
        
        logger.info(f"Clustering Summary:")
        logger.info(f"  Total detections: {total_detections}")
        logger.info(f"  Unique faces: {unique_faces}")
        logger.info(f"  Duplicate groups: {duplicate_groups_count}")
        
        # Log top duplicate groups
        if duplicate_groups:
            sorted_duplicates = sorted(
                duplicate_groups.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            logger.info("Top duplicate groups:")
            for face_id, count in sorted_duplicates[:5]:
                videos = list(video_counts[face_id])
                logger.info(f"  {face_id}: {count} detections across {len(videos)} videos")
                if len(videos) <= 3:
                    logger.info(f"    Videos: {', '.join(videos)}")
    
    def validate_clusters(
        self,
        detections: List[FaceDetection],
        sample_size: int = 100
    ) -> Dict[str, float]:
        """
        Validate clustering quality by sampling pairs within clusters.
        
        Args:
            detections: Clustered face detections
            sample_size: Number of pairs to sample for validation
            
        Returns:
            Dictionary with validation metrics
        """
        logger.info("Validating cluster quality")
        
        # Group detections by face ID
        clusters = defaultdict(list)
        for detection in detections:
            if detection.embedding is not None:
                clusters[detection.face_id].append(detection)
        
        # Sample pairs within clusters
        intra_cluster_distances = []
        inter_cluster_distances = []
        
        face_ids = list(clusters.keys())
        samples_collected = 0
        
        # Intra-cluster distances (should be small)
        for face_id, cluster_detections in clusters.items():
            if len(cluster_detections) < 2:
                continue
            
            for i in range(min(10, len(cluster_detections) - 1)):
                for j in range(i + 1, min(i + 6, len(cluster_detections))):
                    emb1 = np.array(cluster_detections[i].embedding)
                    emb2 = np.array(cluster_detections[j].embedding)
                    distance = self.compute_embedding_distance(emb1, emb2)
                    intra_cluster_distances.append(distance)
                    
                    samples_collected += 1
                    if samples_collected >= sample_size // 2:
                        break
                if samples_collected >= sample_size // 2:
                    break
            if samples_collected >= sample_size // 2:
                break
        
        # Inter-cluster distances (should be large)
        samples_collected = 0
        for i in range(min(20, len(face_ids) - 1)):
            for j in range(i + 1, min(i + 6, len(face_ids))):
                cluster1 = clusters[face_ids[i]]
                cluster2 = clusters[face_ids[j]]
                
                if cluster1 and cluster2:
                    # Sample one detection from each cluster
                    emb1 = np.array(cluster1[0].embedding)
                    emb2 = np.array(cluster2[0].embedding)
                    distance = self.compute_embedding_distance(emb1, emb2)
                    inter_cluster_distances.append(distance)
                    
                    samples_collected += 1
                    if samples_collected >= sample_size // 2:
                        break
            if samples_collected >= sample_size // 2:
                break
        
        # Compute metrics
        metrics = {}
        
        if intra_cluster_distances:
            metrics['avg_intra_cluster_distance'] = np.mean(intra_cluster_distances)
            metrics['max_intra_cluster_distance'] = np.max(intra_cluster_distances)
        
        if inter_cluster_distances:
            metrics['avg_inter_cluster_distance'] = np.mean(inter_cluster_distances)
            metrics['min_inter_cluster_distance'] = np.min(inter_cluster_distances)
        
        if intra_cluster_distances and inter_cluster_distances:
            # Separation ratio (higher is better)
            metrics['separation_ratio'] = (
                np.mean(inter_cluster_distances) / np.mean(intra_cluster_distances)
            )
        
        # Estimate precision based on distance threshold
        if intra_cluster_distances:
            correct_matches = sum(1 for d in intra_cluster_distances if d <= self.match_threshold)
            metrics['estimated_precision'] = correct_matches / len(intra_cluster_distances)
        
        logger.info(f"Validation metrics: {metrics}")
        return metrics
