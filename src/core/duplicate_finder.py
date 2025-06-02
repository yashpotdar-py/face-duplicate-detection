"""
Duplicate face detection and matching module.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import json
from loguru import logger
from collections import defaultdict

from ..utils.config import Config


class DuplicateFinder:
    """
    Find duplicate faces across images and videos using face embeddings.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the duplicate finder.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.face_database = []  # Store all face embeddings and metadata
        self.similarity_matrix = None
        self.duplicate_clusters = []
        
        logger.info("DuplicateFinder initialized")
    
    def add_faces(self, face_data: Dict[str, Any]) -> None:
        """
        Add faces from processed image/video to the database.
        
        Args:
            face_data: Face data from FaceDetector
        """
        for face in face_data.get("faces", []):
            face_entry = {
                "face_id": face["face_id"],
                "embedding": face["embedding"],
                "confidence": face["confidence"],
                "source_path": face_data.get("image_path") or face_data.get("video_name"),
                "source_type": "image" if "image_path" in face_data else "video",
                "frame_number": face.get("frame_number"),
                "crop": face["crop"]
            }
            self.face_database.append(face_entry)
    
    def compute_similarity_matrix(self) -> np.ndarray:
        """
        Compute pairwise similarity matrix for all faces.
        
        Returns:
            Similarity matrix (n_faces x n_faces)
        """
        if not self.face_database:
            logger.warning("No faces in database to compute similarities")
            return np.array([])
        
        # Extract embeddings
        embeddings = []
        for face in self.face_database:
            embedding = face["embedding"]
            # Normalize embedding
            if isinstance(embedding, np.ndarray):
                if embedding.ndim > 1:
                    embedding = embedding.flatten()
                # Normalize to unit vector
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                else:
                    logger.warning("Zero-norm embedding detected, using random unit vector")
                    embedding = np.random.randn(len(embedding))
                    embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
        
        # Compute cosine similarity
        similarity_matrix = cosine_similarity(embeddings)
        
        # Clamp to valid range to handle floating-point precision errors
        similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)
        
        # Ensure diagonal is exactly 1.0
        np.fill_diagonal(similarity_matrix, 1.0)
        
        # Ensure symmetry
        self.similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
        
        logger.info(f"Computed similarity matrix for {len(embeddings)} faces (range: [{self.similarity_matrix.min():.6f}, {self.similarity_matrix.max():.6f}])")
        return self.similarity_matrix
    
    def find_duplicates_threshold(self, threshold: float = None) -> List[List[int]]:
        """
        Find duplicate faces using similarity threshold.
        
        Args:
            threshold: Similarity threshold (uses config default if None)
            
        Returns:
            List of lists, where each inner list contains indices of duplicate faces
        """
        if threshold is None:
            threshold = self.config.similarity_threshold
        
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()
        
        if self.similarity_matrix is None or self.similarity_matrix.size == 0:
            return []
        
        # Find pairs above threshold
        duplicate_pairs = []
        n_faces = self.similarity_matrix.shape[0]
        
        for i in range(n_faces):
            for j in range(i + 1, n_faces):
                if self.similarity_matrix[i, j] > threshold:
                    duplicate_pairs.append((i, j))
        
        # Group connected components
        if not duplicate_pairs:
            return []
        
        # Build adjacency list
        adjacency = defaultdict(set)
        for i, j in duplicate_pairs:
            adjacency[i].add(j)
            adjacency[j].add(i)
        
        # Find connected components
        visited = set()
        duplicate_groups = []
        
        for node in adjacency:
            if node not in visited:
                group = []
                stack = [node]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        group.append(current)
                        stack.extend(adjacency[current] - visited)
                
                if len(group) > 1:
                    duplicate_groups.append(sorted(group))
        
        self.duplicate_clusters = duplicate_groups
        logger.info(f"Found {len(duplicate_groups)} duplicate groups with threshold {threshold}")
        
        return duplicate_groups
    
    def find_duplicates_clustering(self, eps: float = 0.4, min_samples: int = 2) -> List[List[int]]:
        """
        Find duplicate faces using DBSCAN clustering.
        
        Args:
            eps: Maximum distance for clustering
            min_samples: Minimum samples in a cluster
            
        Returns:
            List of lists, where each inner list contains indices of duplicate faces
        """
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()
        
        if self.similarity_matrix is None or self.similarity_matrix.size == 0:
            return []
        
        # Convert similarity to distance, handling floating-point precision issues
        # Clamp similarity values to [0, 1] to avoid negative distances
        similarity_clamped = np.clip(self.similarity_matrix, 0.0, 1.0)
        distance_matrix = 1 - similarity_clamped
        
        # Ensure non-negative distances and make symmetric
        distance_matrix = np.maximum(distance_matrix, 0.0)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        
        # Set diagonal to 0 (distance from a point to itself)
        np.fill_diagonal(distance_matrix, 0.0)
        
        logger.debug(f"Distance matrix range: [{distance_matrix.min():.6f}, {distance_matrix.max():.6f}]")
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        # Group faces by cluster
        clusters = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            if label != -1:  # Ignore noise points
                clusters[label].append(idx)
        
        # Filter clusters with more than one face
        duplicate_groups = [group for group in clusters.values() if len(group) > 1]
        
        self.duplicate_clusters = duplicate_groups
        logger.info(f"Found {len(duplicate_groups)} duplicate groups using clustering")
        
        return duplicate_groups
    
    def get_duplicate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive duplicate report.
        
        Returns:
            Dictionary containing duplicate analysis results
        """
        if not self.duplicate_clusters:
            logger.warning("No duplicate analysis performed yet")
            return {}
        
        report = {
            "summary": {
                "total_faces": len(self.face_database),
                "duplicate_groups": len(self.duplicate_clusters),
                "total_duplicates": sum(len(group) for group in self.duplicate_clusters),
                "unique_faces": len(self.face_database) - sum(len(group) - 1 for group in self.duplicate_clusters)
            },
            "duplicate_groups": [],
            "statistics": {}
        }
        
        # Process each duplicate group
        for group_idx, face_indices in enumerate(self.duplicate_clusters):
            group_info = {
                "group_id": group_idx,
                "num_faces": len(face_indices),
                "faces": []
            }
            
            similarities = []
            for i, face_idx in enumerate(face_indices):
                face = self.face_database[face_idx]
                
                # Calculate average similarity within group
                group_similarities = []
                for other_idx in face_indices:
                    if other_idx != face_idx:
                        sim = self.similarity_matrix[face_idx, other_idx]
                        group_similarities.append(sim)
                
                avg_similarity = np.mean(group_similarities) if group_similarities else 0.0
                similarities.append(avg_similarity)
                
                face_info = {
                    "face_id": face["face_id"],
                    "source_path": face["source_path"],
                    "source_type": face["source_type"],
                    "frame_number": face["frame_number"],
                    "confidence": face["confidence"],
                    "avg_similarity": avg_similarity
                }
                group_info["faces"].append(face_info)
            
            group_info["avg_group_similarity"] = np.mean(similarities)
            report["duplicate_groups"].append(group_info)
        
        # Calculate statistics
        all_similarities = []
        for group in self.duplicate_clusters:
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    sim = self.similarity_matrix[group[i], group[j]]
                    all_similarities.append(sim)
        
        if all_similarities:
            report["statistics"] = {
                "min_similarity": float(np.min(all_similarities)),
                "max_similarity": float(np.max(all_similarities)),
                "mean_similarity": float(np.mean(all_similarities)),
                "std_similarity": float(np.std(all_similarities))
            }
        
        return report
    
    def save_results(self, output_dir: str) -> Dict[str, str]:
        """
        Save duplicate detection results to files.
        
        Args:
            output_dir: Output directory path
            
        Returns:
            Dictionary mapping result type to file path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        try:
            # Save duplicate report
            report = self.get_duplicate_report()
            report_file = output_path / "duplicate_report.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            saved_files["report"] = str(report_file)
            
            # Save similarity matrix
            if self.similarity_matrix is not None:
                similarity_file = output_path / "similarity_matrix.npy"
                np.save(similarity_file, self.similarity_matrix)
                saved_files["similarity_matrix"] = str(similarity_file)
            
            # Save face database
            face_db_file = output_path / "face_database.json"
            face_db_clean = []
            for face in self.face_database:
                clean_face = face.copy()
                # Convert numpy arrays to lists for JSON serialization
                if isinstance(clean_face["embedding"], np.ndarray):
                    clean_face["embedding"] = clean_face["embedding"].tolist()
                # Remove crop data (too large for JSON)
                clean_face.pop("crop", None)
                face_db_clean.append(clean_face)
            
            with open(face_db_file, 'w') as f:
                json.dump(face_db_clean, f, indent=2, default=str)
            saved_files["face_database"] = str(face_db_file)
            
            # Create duplicate pairs CSV
            if self.duplicate_clusters:
                pairs_data = []
                for group_idx, face_indices in enumerate(self.duplicate_clusters):
                    for i in range(len(face_indices)):
                        for j in range(i + 1, len(face_indices)):
                            idx1, idx2 = face_indices[i], face_indices[j]
                            face1 = self.face_database[idx1]
                            face2 = self.face_database[idx2]
                            
                            pairs_data.append({
                                "group_id": group_idx,
                                "face1_id": face1["face_id"],
                                "face1_source": face1["source_path"],
                                "face2_id": face2["face_id"],
                                "face2_source": face2["source_path"],
                                "similarity": self.similarity_matrix[idx1, idx2]
                            })
                
                pairs_df = pd.DataFrame(pairs_data)
                pairs_file = output_path / "duplicate_pairs.csv"
                pairs_df.to_csv(pairs_file, index=False)
                saved_files["duplicate_pairs"] = str(pairs_file)
            
            logger.info(f"Results saved to {output_dir}")
            return saved_files
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return {}
    
    def clear_database(self) -> None:
        """Clear the face database and computed results."""
        self.face_database.clear()
        self.similarity_matrix = None
        self.duplicate_clusters.clear()
        logger.info("Face database cleared")
