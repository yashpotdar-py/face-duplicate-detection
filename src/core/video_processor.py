"""
Video processing module for extracting frames and detecting faces.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple, Optional, Dict, Any
from loguru import logger

from ..utils.config import Config


class VideoProcessor:
    """
    Process video files to extract frames for face detection.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the video processor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        logger.info("VideoProcessor initialized")
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get information about a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing video information
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            video_info = {
                "path": video_path,
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "duration_seconds": duration,
                "file_size_mb": Path(video_path).stat().st_size / (1024 * 1024)
            }
            
            logger.info(f"Video info: {Path(video_path).name} - {duration:.1f}s, {frame_count} frames")
            return video_info
            
        except Exception as e:
            logger.error(f"Error getting video info for {video_path}: {e}")
            return {"path": video_path, "error": str(e)}
    
    def extract_frames(
        self, 
        video_path: str, 
        frame_skip: Optional[int] = None,
        max_frames: Optional[int] = None
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Extract frames from a video file.
        
        Args:
            video_path: Path to the video file
            frame_skip: Number of frames to skip between extractions
            max_frames: Maximum number of frames to extract
            
        Yields:
            Tuple of (frame_number, frame_array)
        """
        if frame_skip is None:
            frame_skip = self.config.video_frame_skip
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            frame_number = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Extract frame at specified intervals
                if frame_number % frame_skip == 0:
                    yield frame_number, frame
                    extracted_count += 1
                    
                    # Check max frames limit
                    if max_frames and extracted_count >= max_frames:
                        break
                
                frame_number += 1
            
            cap.release()
            logger.info(f"Extracted {extracted_count} frames from {Path(video_path).name}")
            
        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {e}")
    
    def extract_frames_smart(
        self, 
        video_path: str,
        target_frames: int = 100
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Smart frame extraction that automatically determines frame skip to get target number of frames.
        
        Args:
            video_path: Path to the video file
            target_frames: Target number of frames to extract
            
        Yields:
            Tuple of (frame_number, frame_array)
        """
        try:
            video_info = self.get_video_info(video_path)
            
            if "error" in video_info:
                return
            
            total_frames = video_info["frame_count"]
            
            # Calculate optimal frame skip
            if total_frames <= target_frames:
                frame_skip = 1
            else:
                frame_skip = max(1, total_frames // target_frames)
            
            logger.info(f"Smart extraction: {total_frames} total frames, skip={frame_skip}, target={target_frames}")
            
            yield from self.extract_frames(video_path, frame_skip=frame_skip, max_frames=target_frames)
            
        except Exception as e:
            logger.error(f"Error in smart frame extraction for {video_path}: {e}")
    
    def extract_thumbnail(self, video_path: str, timestamp: float = 0.1) -> Optional[np.ndarray]:
        """
        Extract a single thumbnail frame from a video.
        
        Args:
            video_path: Path to the video file
            timestamp: Timestamp in seconds (or fraction) to extract
            
        Returns:
            Thumbnail frame as numpy array or None if failed
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Calculate frame position
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if timestamp < 1:  # Treat as fraction
                target_frame = int(frame_count * timestamp)
            else:  # Treat as seconds
                target_frame = int(fps * timestamp)
            
            target_frame = max(0, min(target_frame, frame_count - 1))
            
            # Seek to target frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            
            cap.release()
            
            if ret:
                return frame
            else:
                logger.warning(f"Could not extract thumbnail from {video_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting thumbnail from {video_path}: {e}")
            return None
    
    def process_video_alternative(self, video_path: str, output_fps: int = 1) -> Iterator[Tuple[float, np.ndarray]]:
        """
        Alternative video processing method using OpenCV only.
        
        Args:
            video_path: Path to the video file
            output_fps: Target FPS for frame extraction
            
        Yields:
            Tuple of (timestamp, frame_array)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Calculate frame interval
            interval = max(1, int(fps / output_fps)) if fps > 0 else 1
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_idx % interval == 0:
                    timestamp = frame_idx / fps if fps > 0 else frame_idx
                    yield timestamp, frame
                    
                frame_idx += 1
                        
            cap.release()
            logger.info(f"Processed video {Path(video_path).name}")
                        
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
    
    def validate_video_file(self, video_path: str) -> bool:
        """
        Validate if a video file can be processed.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            True if video is valid, False otherwise
        """
        try:
            # Check file exists
            if not Path(video_path).exists():
                return False
            
            # Try to open with OpenCV
            cap = cv2.VideoCapture(video_path)
            is_opened = cap.isOpened()
            
            if is_opened:
                # Try to read first frame
                ret, _ = cap.read()
                cap.release()
                return ret
            
            cap.release()
            return False
            
        except Exception:
            return False
    
    @staticmethod
    def get_supported_formats() -> list:
        """Get list of supported video formats."""
        return ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
