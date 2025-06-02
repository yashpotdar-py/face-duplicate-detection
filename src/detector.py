"""
Main module for duplicate face detection in bicycle videos.
"""
import time
from pathlib import Path
from typing import List, Optional
from loguru import logger

from .models import FaceDetection, DuplicateReport, DetectionConfig
from .video_processor import VideoProcessor
from .face_clusterer import FaceClusterer


class DuplicateFaceDetector:
    """Main class for detecting duplicate faces across videos."""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.video_processor = VideoProcessor(config)
        self.face_clusterer = FaceClusterer(config)
    
    def process_videos(
        self,
        videos_dir: Path,
        thumbnails_dir: Optional[Path] = None,
        skip_frames: int = 30
    ) -> List[FaceDetection]:
        """
        Process all videos in the directory to extract face detections.
        
        Args:
            videos_dir: Directory containing video files
            thumbnails_dir: Optional directory with thumbnails (not used in current implementation)
            skip_frames: Process every nth frame (default: 30, ~1fps for 30fps video)
            
        Returns:
            List of face detections from all videos
        """
        logger.info("Starting video processing phase")
        
        if not videos_dir.exists():
            raise FileNotFoundError(f"Videos directory not found: {videos_dir}")
        
        # Process all videos to extract faces
        detections = self.video_processor.process_videos(videos_dir, skip_frames)
        
        if not detections:
            logger.warning("No faces detected in any videos")
            return []
        
        logger.info(f"Extracted {len(detections)} face detections from videos")
        return detections
    
    def find_duplicates(self, detections: List[FaceDetection]) -> List[FaceDetection]:
        """
        Cluster faces to identify duplicates.
        
        Args:
            detections: List of face detections
            
        Returns:
            List of detections with updated face IDs for duplicates
        """
        logger.info("Starting face clustering phase")
        
        if not detections:
            logger.warning("No detections to cluster")
            return []
        
        # Cluster faces to find duplicates
        clustered_detections = self.face_clusterer.find_duplicate_faces(detections)
        
        # Validate clustering quality
        validation_metrics = self.face_clusterer.validate_clusters(clustered_detections)
        
        return clustered_detections
    
    def generate_report(
        self,
        detections: List[FaceDetection],
        processing_time: float
    ) -> DuplicateReport:
        """
        Generate a comprehensive report of duplicate face detection results.
        
        Args:
            detections: List of clustered face detections
            processing_time: Total processing time in seconds
            
        Returns:
            DuplicateReport object
        """
        logger.info("Generating duplicate detection report")
        
        # Count unique faces and duplicate groups
        face_ids = set(d.face_id for d in detections)
        unique_faces = len(face_ids)
        
        # Count duplicate groups (faces appearing more than once)
        face_counts = {}
        for detection in detections:
            face_counts[detection.face_id] = face_counts.get(detection.face_id, 0) + 1
        
        duplicate_groups = sum(1 for count in face_counts.values() if count > 1)
        
        # Create report
        report = DuplicateReport(
            total_faces=len(detections),
            unique_faces=unique_faces,
            duplicate_groups=duplicate_groups,
            detections=detections,
            processing_time=processing_time,
            config={
                'min_confidence': self.config.min_confidence,
                'match_threshold': self.config.match_threshold,
                'batch_size': self.config.batch_size,
                'device': self.config.device
            }
        )
        
        logger.info(f"Report generated - Total: {len(detections)}, "
                   f"Unique: {unique_faces}, Duplicate groups: {duplicate_groups}")
        
        return report
    
    def run_detection(
        self,
        videos_dir: Path,
        output_file: Path,
        thumbnails_dir: Optional[Path] = None,
        skip_frames: int = 30
    ) -> DuplicateReport:
        """
        Run the complete duplicate face detection pipeline.
        
        Args:
            videos_dir: Directory containing video files
            output_file: Path to save the output report
            thumbnails_dir: Optional directory with thumbnails
            skip_frames: Process every nth frame
            
        Returns:
            DuplicateReport object
        """
        start_time = time.time()
        
        logger.info("Starting duplicate face detection pipeline")
        logger.info(f"Videos directory: {videos_dir}")
        logger.info(f"Output file: {output_file}")
        logger.info(f"Configuration: {self.config.__dict__}")
        
        try:
            # Step 1: Process videos to extract faces
            detections = self.process_videos(videos_dir, thumbnails_dir, skip_frames)
            
            if not detections:
                logger.error("No faces detected in videos")
                # Create empty report
                processing_time = time.time() - start_time
                report = self.generate_report([], processing_time)
                return report
            
            # Step 2: Cluster faces to find duplicates
            clustered_detections = self.find_duplicates(detections)
            
            # Step 3: Generate report
            processing_time = time.time() - start_time
            report = self.generate_report(clustered_detections, processing_time)
            
            # Step 4: Save report
            if output_file.suffix.lower() == '.json':
                report.to_json(output_file)
                logger.info(f"Report saved to JSON: {output_file}")
            elif output_file.suffix.lower() == '.csv':
                report.to_csv(output_file)
                logger.info(f"Report saved to CSV: {output_file}")
            else:
                # Default to JSON
                json_file = output_file.with_suffix('.json')
                report.to_json(json_file)
                logger.info(f"Report saved to JSON: {json_file}")
            
            logger.info(f"Pipeline completed in {processing_time:.2f} seconds")
            return report
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
