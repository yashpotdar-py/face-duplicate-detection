#!/usr/bin/env python3
"""
Debug script to extract and save video frames for inspection.
"""

import cv2
import numpy as np
from pathlib import Path
from src.core.video_processor import VideoProcessor
from src.core.face_detector import FaceDetector
from src.utils.config import Config

def debug_video_frames():
    """Extract and save sample frames from videos to debug face detection."""
    
    # Setup
    config = Config()
    video_processor = VideoProcessor(config)
    face_detector = FaceDetector(config)
    
    # Create debug output directory
    debug_dir = Path("debug_frames")
    debug_dir.mkdir(exist_ok=True)
    
    # Find a video file
    videos_dir = Path("data/Videos")
    video_files = list(videos_dir.glob("*.MP4"))
    
    if not video_files:
        print("No video files found!")
        return
    
    # Process first video
    video_file = video_files[0]
    print(f"Processing video: {video_file}")
    
    # Get video info
    video_info = video_processor.get_video_info(str(video_file))
    print(f"Video info: {video_info}")
    
    # Extract first few frames
    frame_count = 0
    for frame_number, frame in video_processor.extract_frames_smart(str(video_file), target_frames=3):
        print(f"\nFrame {frame_number}: shape={frame.shape}, dtype={frame.dtype}")
        print(f"  Min pixel value: {frame.min()}")
        print(f"  Max pixel value: {frame.max()}")
        print(f"  Mean pixel value: {frame.mean():.2f}")
        
        # Save frame
        frame_path = debug_dir / f"frame_{frame_number}.jpg"
        cv2.imwrite(str(frame_path), frame)
        print(f"  Saved frame to: {frame_path}")
        
        # Try face detection on this frame
        print("  Attempting face detection...")
        try:
            face_crops, embeddings, probs = face_detector.detect_faces(frame)
            print(f"  Faces detected: {len(face_crops)}")
            
            if len(face_crops) > 0:
                print(f"  Face probabilities: {probs}")
                for i, crop in enumerate(face_crops):
                    crop_path = debug_dir / f"frame_{frame_number}_face_{i}.jpg"
                    cv2.imwrite(str(crop_path), crop)
                    print(f"    Saved face crop to: {crop_path}")
        except Exception as e:
            print(f"  Face detection error: {e}")
        
        # Also try with different preprocessing
        print("  Trying with histogram equalization...")
        try:
            # Convert to grayscale, apply histogram equalization, then back to color
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            equalized = cv2.equalizeHist(gray)
            enhanced_frame = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
            
            # Save enhanced frame
            enhanced_path = debug_dir / f"frame_{frame_number}_enhanced.jpg"
            cv2.imwrite(str(enhanced_path), enhanced_frame)
            print(f"    Saved enhanced frame to: {enhanced_path}")
            
            # Try face detection on enhanced frame
            face_crops_enh, embeddings_enh, probs_enh = face_detector.detect_faces(enhanced_frame)
            print(f"    Enhanced frame faces detected: {len(face_crops_enh)}")
            
        except Exception as e:
            print(f"    Enhanced face detection error: {e}")
        
        frame_count += 1
        if frame_count >= 3:
            break
    
    print(f"\nDebug frames saved to: {debug_dir}")
    print("Check the saved frames to see what the video content looks like.")

if __name__ == "__main__":
    debug_video_frames()
