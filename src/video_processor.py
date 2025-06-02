"""
Video processing module for extracting frames and faces from videos.
"""
import cv2
import numpy as np
from typing import Iterator, Tuple, List, Optional
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import time

from .models import FaceDetection, BoundingBox, DetectionConfig
from .face_detector import FaceDetector


class VideoProcessor:
    """Process videos to extract faces and generate detections."""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.face_detector = FaceDetector(config)
    
    def get_video_info(self, video_path: Path) -> Tuple[int, float, Tuple[int, int]]:
        """
        Get basic video information.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (total_frames, fps, (width, height))
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        return total_frames, fps, (width, height)
    
    def frame_generator(self, video_path: Path, skip_frames: int = 1) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Generate frames from video with optional frame skipping.
        
        Args:
            video_path: Path to video file
            skip_frames: Process every nth frame (1 = every frame)
            
        Yields:
            Tuple of (frame_number, frame_array)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frame_number = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_number % skip_frames == 0:
                    yield frame_number, frame
                
                frame_number += 1
                
        finally:
            cap.release()
    
    def timestamp_from_frame(self, frame_number: int, fps: float) -> str:
        """
        Convert frame number to timestamp string.
        
        Args:
            frame_number: Frame index
            fps: Frames per second
            
        Returns:
            Timestamp in format "hh:mm:ss.ms"
        """
        seconds = frame_number / fps
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    
    def process_video(self, video_path: Path, skip_frames: int = 1) -> List[FaceDetection]:
        """
        Process a single video file to extract all face detections.
        
        Args:
            video_path: Path to video file
            skip_frames: Process every nth frame
            
        Returns:
            List of FaceDetection objects
        """
        logger.info(f"Processing video: {video_path.name}")
        
        try:
            # Get video information
            total_frames, fps, (width, height) = self.get_video_info(video_path)
            logger.info(f"Video info - Frames: {total_frames}, FPS: {fps}, Size: {width}x{height}")
            
            detections = []
            frames_to_process = total_frames // skip_frames
            
            # Process frames with progress bar
            with tqdm(
                total=frames_to_process,
                desc=f"Processing {video_path.name}",
                unit="frames"
            ) as pbar:
                
                batch_frames = []
                batch_frame_numbers = []
                
                for frame_number, frame in self.frame_generator(video_path, skip_frames):
                    batch_frames.append(frame)
                    batch_frame_numbers.append(frame_number)
                    
                    # Process in batches for efficiency
                    if len(batch_frames) >= self.config.batch_size:
                        batch_detections = self._process_frame_batch(
                            batch_frames, batch_frame_numbers, fps, video_path.name
                        )
                        detections.extend(batch_detections)
                        
                        batch_frames = []
                        batch_frame_numbers = []
                    
                    pbar.update(1)
                
                # Process remaining frames
                if batch_frames:
                    batch_detections = self._process_frame_batch(
                        batch_frames, batch_frame_numbers, fps, video_path.name
                    )
                    detections.extend(batch_detections)
            
            logger.info(f"Found {len(detections)} face detections in {video_path.name}")
            return detections
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
            return []
    
    def _process_frame_batch(
        self,
        frames: List[np.ndarray],
        frame_numbers: List[int],
        fps: float,
        video_filename: str
    ) -> List[FaceDetection]:
        """
        Process a batch of frames to extract face detections.
        
        Args:
            frames: List of frame arrays
            frame_numbers: Corresponding frame numbers
            fps: Video FPS
            video_filename: Name of the video file
            
        Returns:
            List of FaceDetection objects
        """
        detections = []
        
        for frame, frame_number in zip(frames, frame_numbers):
            try:
                # Detect faces and extract embeddings
                face_results = self.face_detector.detect_and_extract(frame)
                
                # Convert to FaceDetection objects
                for i, (bbox, confidence, embedding) in enumerate(face_results):
                    timestamp = self.timestamp_from_frame(frame_number, fps)
                    
                    # Generate temporary face ID (will be updated during clustering)
                    face_id = f"{video_filename}_{frame_number}_{i}"
                    
                    detection = FaceDetection(
                        face_id=face_id,
                        video_filename=video_filename,
                        timestamp=timestamp,
                        bounding_box=bbox,
                        confidence=confidence,
                        embedding=embedding.tolist() if embedding is not None else None
                    )
                    
                    detections.append(detection)
                    
            except Exception as e:
                logger.warning(f"Error processing frame {frame_number}: {e}")
                continue
        
        return detections
    
    def process_videos(self, video_dir: Path, skip_frames: int = 30) -> List[FaceDetection]:
        """
        Process all videos in a directory.
        
        Args:
            video_dir: Directory containing video files
            skip_frames: Process every nth frame (30 = ~1 fps for 30fps video)
            
        Returns:
            List of all FaceDetection objects from all videos
        """
        logger.info(f"Processing videos from: {video_dir}")
        
        # Find all video files
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
        video_files = [
            f for f in video_dir.iterdir()
            if f.is_file() and f.suffix.lower() in video_extensions
        ]
        
        if not video_files:
            logger.warning(f"No video files found in {video_dir}")
            return []
        
        logger.info(f"Found {len(video_files)} video files")
        
        all_detections = []
        start_time = time.time()
        
        for video_file in video_files:
            video_detections = self.process_video(video_file, skip_frames)
            all_detections.extend(video_detections)
        
        processing_time = time.time() - start_time
        logger.info(f"Processed all videos in {processing_time:.2f} seconds")
        logger.info(f"Total detections: {len(all_detections)}")
        
        return all_detections
