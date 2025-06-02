"""
Image processor for detecting duplicate faces in thumbnail images.

This module provides functionality to process thumbnail images and detect
duplicate faces using the same face detection and clustering algorithms
used for video processing.
"""
import os
import time
from typing import List, Optional, Dict, Set, Tuple
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageOps
from loguru import logger

from .models import DetectionConfig, FaceDetection, BoundingBox, DuplicateReport
from .face_detector import FaceDetector
from .face_clusterer import FaceClusterer


class ImageProcessor:
    """
    Process thumbnail images to detect and cluster duplicate faces.
    
    This class handles:
    - Loading and preprocessing thumbnail images
    - Detecting faces in static images
    - Extracting face embeddings
    - Clustering faces to find duplicates across images
    - Generating comprehensive reports
    """
    
    def __init__(self, config: DetectionConfig):
        """Initialize the image processor with configuration."""
        self.config = config
        self.face_detector = FaceDetector(config)
        self.face_clusterer = FaceClusterer(config)
        
        # Supported image formats
        self.image_extensions = {
            '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'
        }
        
        logger.info(f"ImageProcessor initialized with device: {config.device}")
    
    def load_and_preprocess_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Load and preprocess an image for face detection.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image as numpy array or None if loading failed
        """
        try:
            # Load image using PIL for better format support
            with Image.open(image_path) as pil_image:
                # Convert to RGB if necessary
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                
                # Auto-orient based on EXIF data
                pil_image = ImageOps.exif_transpose(pil_image)
                
                # Convert to numpy array
                image = np.array(pil_image)
                
                # Convert RGB to BGR for OpenCV
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Resize if image is too large (for memory efficiency)
                height, width = image.shape[:2]
                max_dimension = 1920  # Max width or height
                
                if max(height, width) > max_dimension:
                    if width > height:
                        new_width = max_dimension
                        new_height = int(height * (max_dimension / width))
                    else:
                        new_height = max_dimension
                        new_width = int(width * (max_dimension / height))
                    
                    image = cv2.resize(image, (new_width, new_height), 
                                     interpolation=cv2.INTER_AREA)
                    logger.debug(f"Resized {image_path.name} from {width}x{height} to {new_width}x{new_height}")
                
                return image
                
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None
    
    def find_image_files(self, thumbnails_dir: Path) -> List[Path]:
        """
        Find all supported image files in the thumbnails directory.
        
        Args:
            thumbnails_dir: Directory containing thumbnail images
            
        Returns:
            List of paths to valid image files
        """
        image_files = []
        
        if not thumbnails_dir.exists():
            logger.warning(f"Thumbnails directory does not exist: {thumbnails_dir}")
            return image_files
        
        for file_path in thumbnails_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.image_extensions:
                image_files.append(file_path)
        
        logger.info(f"Found {len(image_files)} image files in {thumbnails_dir}")
        return sorted(image_files)
    
    def process_single_image(self, image_path: Path) -> List[FaceDetection]:
        """
        Process a single image to detect faces.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of face detections found in the image
        """
        # Load and preprocess image
        image = self.load_and_preprocess_image(image_path)
        if image is None:
            return []
        
        # Detect faces
        detections = self.face_detector.detect_faces(image)
        
        if not detections:
            logger.debug(f"No faces detected in {image_path.name}")
            return []
        
        # Extract embeddings
        bboxes = [bbox for bbox, _ in detections]
        embeddings = self.face_detector.extract_embeddings(image, bboxes)
        
        if len(embeddings) != len(detections):
            logger.warning(f"Embedding count mismatch in {image_path.name}")
            return []
        
        # Create FaceDetection objects
        face_detections = []
        for i, ((bbox, confidence), embedding) in enumerate(zip(detections, embeddings)):
            detection = FaceDetection(
                face_id=f"temp_face_{image_path.stem}_{i}",
                video_filename=image_path.name,  # Use image filename
                timestamp="00:00:00.000",  # Static timestamp for images
                bounding_box=bbox,
                confidence=confidence,
                embedding=embedding.tolist()
            )
            face_detections.append(detection)
        
        logger.debug(f"Detected {len(face_detections)} faces in {image_path.name}")
        return face_detections
    
    def process_images_batch(self, image_paths: List[Path]) -> List[FaceDetection]:
        """
        Process a batch of images efficiently.
        
        Args:
            image_paths: List of image file paths to process
            
        Returns:
            List of all face detections from the batch
        """
        all_detections = []
        
        for i, image_path in enumerate(image_paths):
            logger.debug(f"Processing image {i+1}/{len(image_paths)}: {image_path.name}")
            
            detections = self.process_single_image(image_path)
            all_detections.extend(detections)
        
        return all_detections
    
    def detect_duplicate_faces_in_images(
        self, 
        thumbnails_dir: Path,
        output_file: Optional[Path] = None
    ) -> DuplicateReport:
        """
        Detect duplicate faces across all images in the thumbnails directory.
        
        Args:
            thumbnails_dir: Directory containing thumbnail images
            output_file: Optional path to save the report
            
        Returns:
            DuplicateReport containing all detected faces and duplicates
        """
        start_time = time.time()
        
        logger.info("Starting duplicate face detection in thumbnail images")
        logger.info(f"Thumbnails directory: {thumbnails_dir}")
        
        # Find all image files
        image_files = self.find_image_files(thumbnails_dir)
        
        if not image_files:
            logger.warning("No image files found to process")
            return DuplicateReport(
                total_faces=0,
                unique_faces=0,
                duplicate_groups=0,
                detections=[],
                processing_time=0.0,
                config=self.config.__dict__
            )
        
        # Process images in batches
        all_detections = []
        batch_size = min(self.config.batch_size, len(image_files))
        
        for i in range(0, len(image_files), batch_size):
            batch = image_files[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(image_files) + batch_size - 1)//batch_size}")
            
            batch_detections = self.process_images_batch(batch)
            all_detections.extend(batch_detections)
        
        logger.info(f"Total faces detected: {len(all_detections)}")
        
        if not all_detections:
            logger.warning("No faces detected in any images")
            return DuplicateReport(
                total_faces=0,
                unique_faces=0,
                duplicate_groups=0,
                detections=[],
                processing_time=time.time() - start_time,
                config=self.config.__dict__
            )
        
        # Cluster faces to find duplicates
        logger.info("Clustering faces to identify duplicates...")
        clustered_detections = self.face_clusterer.find_duplicate_faces(all_detections)
        
        # Calculate statistics
        unique_faces = len(set(detection.face_id for detection in clustered_detections))
        face_counts = {}
        for detection in clustered_detections:
            face_counts[detection.face_id] = face_counts.get(detection.face_id, 0) + 1
        
        duplicate_groups = len([fid for fid, count in face_counts.items() if count > 1])
        
        processing_time = time.time() - start_time
        
        # Create report
        report = DuplicateReport(
            total_faces=len(clustered_detections),
            unique_faces=unique_faces,
            duplicate_groups=duplicate_groups,
            detections=clustered_detections,
            processing_time=processing_time,
            config=self.config.__dict__
        )
        
        # Save report if output file specified
        if output_file:
            self._save_report(report, output_file)
        
        self._log_summary(report, len(image_files))
        
        return report
    
    def find_duplicates_between_videos_and_images(
        self,
        video_detections: List[FaceDetection],
        image_detections: List[FaceDetection]
    ) -> Dict[str, List[FaceDetection]]:
        """
        Find duplicate faces between video detections and image detections.
        
        Args:
            video_detections: Face detections from videos
            image_detections: Face detections from images
            
        Returns:
            Dictionary mapping face IDs to lists of detections (cross-medium duplicates)
        """
        logger.info("Finding duplicates between videos and images...")
        
        # Combine all detections
        all_detections = video_detections + image_detections
        
        # Cluster all faces together
        clustered_detections = self.face_clusterer.find_duplicate_faces(all_detections)
        
        # Find cross-medium duplicates
        cross_duplicates = {}
        face_groups = {}
        
        # Group detections by face ID
        for detection in clustered_detections:
            if detection.face_id not in face_groups:
                face_groups[detection.face_id] = []
            face_groups[detection.face_id].append(detection)
        
        # Find groups that contain both video and image detections
        for face_id, detections in face_groups.items():
            has_video = any(det.video_filename.endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')) 
                          for det in detections)
            has_image = any(det.video_filename.endswith(tuple(self.image_extensions)) 
                          for det in detections)
            
            if has_video and has_image and len(detections) > 1:
                cross_duplicates[face_id] = detections
        
        logger.info(f"Found {len(cross_duplicates)} cross-medium duplicate groups")
        
        return cross_duplicates
    
    def generate_image_similarity_matrix(self, detections: List[FaceDetection]) -> np.ndarray:
        """
        Generate a similarity matrix for face embeddings.
        
        Args:
            detections: List of face detections with embeddings
            
        Returns:
            Similarity matrix as numpy array
        """
        if not detections:
            return np.array([])
        
        # Extract embeddings
        embeddings = []
        for detection in detections:
            if detection.embedding:
                embeddings.append(np.array(detection.embedding))
        
        if not embeddings:
            return np.array([])
        
        embeddings = np.array(embeddings)
        
        # Calculate cosine similarity matrix
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(embeddings)
        
        return similarity_matrix
    
    def _save_report(self, report: DuplicateReport, output_file: Path) -> None:
        """Save the duplicate report to file."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if output_file.suffix.lower() == '.csv':
            report.to_csv(output_file)
        else:
            report.to_json(output_file)
        
        logger.info(f"Report saved to: {output_file}")
    
    def _log_summary(self, report: DuplicateReport, total_images: int) -> None:
        """Log a summary of the processing results."""
        logger.info("="*60)
        logger.info("IMAGE PROCESSING COMPLETED")
        logger.info("="*60)
        logger.info(f"Images processed: {total_images}")
        logger.info(f"Total faces detected: {report.total_faces}")
        logger.info(f"Unique faces: {report.unique_faces}")
        logger.info(f"Duplicate groups: {report.duplicate_groups}")
        logger.info(f"Processing time: {report.processing_time:.2f} seconds")
        
        if total_images > 0:
            logger.info(f"Images per second: {total_images / report.processing_time:.2f}")
        
        if report.total_faces > 0:
            logger.info(f"Faces per second: {report.total_faces / report.processing_time:.2f}")
            duplicate_percentage = (report.duplicate_groups / report.unique_faces) * 100
            logger.info(f"Duplicate percentage: {duplicate_percentage:.1f}%")


class ThumbnailDuplicateDetector:
    """
    High-level interface for thumbnail duplicate detection.
    
    This class provides a simple interface for detecting duplicates in thumbnails
    and optionally comparing them with video detections.
    """
    
    def __init__(self, config: DetectionConfig):
        """Initialize the thumbnail duplicate detector."""
        self.config = config
        self.image_processor = ImageProcessor(config)
    
    def run_thumbnail_detection(
        self,
        thumbnails_dir: Path,
        output_file: Path,
        compare_with_videos: bool = False,
        video_detections: Optional[List[FaceDetection]] = None
    ) -> DuplicateReport:
        """
        Run duplicate detection on thumbnail images.
        
        Args:
            thumbnails_dir: Directory containing thumbnail images
            output_file: Path to save the detection report
            compare_with_videos: Whether to compare with video detections
            video_detections: Optional video detections for comparison
            
        Returns:
            DuplicateReport containing results
        """
        logger.info("Starting thumbnail duplicate detection")
        
        # Process thumbnail images
        report = self.image_processor.detect_duplicate_faces_in_images(
            thumbnails_dir=thumbnails_dir,
            output_file=output_file
        )
        
        # Optional: Compare with video detections
        if compare_with_videos and video_detections:
            cross_duplicates = self.image_processor.find_duplicates_between_videos_and_images(
                video_detections=video_detections,
                image_detections=report.detections
            )
            
            if cross_duplicates:
                logger.info(f"Found {len(cross_duplicates)} cross-medium duplicate groups")
                
                # Save cross-duplicate report
                cross_output = output_file.parent / f"cross_media_duplicates_{output_file.name}"
                self._save_cross_duplicates(cross_duplicates, cross_output)
        
        return report
    
    def _save_cross_duplicates(
        self, 
        cross_duplicates: Dict[str, List[FaceDetection]], 
        output_file: Path
    ) -> None:
        """Save cross-medium duplicate results."""
        data = {}
        for face_id, detections in cross_duplicates.items():
            data[face_id] = [det.dict() for det in detections]
        
        import json
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Cross-medium duplicates saved to: {output_file}")