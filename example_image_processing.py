#!/usr/bin/env python3
"""
Example script demonstrating thumbnail duplicate detection functionality.

This script shows how to use the image processing components to detect
duplicate faces in thumbnail images and compare them with video results.
"""
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from loguru import logger

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src import DetectionConfig, ThumbnailDuplicateDetector, ImageProcessor
from src.models import FaceDetection, BoundingBox, DuplicateReport


def create_sample_thumbnails(output_dir: Path, num_images: int = 5) -> list[Path]:
    """
    Create sample thumbnail images for testing.
    
    Args:
        output_dir: Directory to save sample images
        num_images: Number of sample images to create
        
    Returns:
        List of paths to created image files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = []
    
    for i in range(num_images):
        # Create a synthetic thumbnail image
        thumbnail = np.zeros((300, 400, 3), dtype=np.uint8)
        
        # Add background gradient
        for y in range(300):
            thumbnail[y, :] = [50 + y//3, 100 + y//4, 150 + y//5]
        
        # Add face-like regions (some will be duplicates)
        if i in [0, 3]:  # Make images 0 and 3 have similar faces (duplicates)
            face_positions = [(150, 100, 80, 100)]
            base_color = (200, 180, 160)
        elif i in [1, 4]:  # Make images 1 and 4 have similar faces (duplicates)
            face_positions = [(120, 80, 90, 110)]
            base_color = (210, 170, 150)
        else:  # Unique face
            face_positions = [(100, 120, 85, 95)]
            base_color = (190, 185, 170)
        
        for x, y, w, h in face_positions:
            # Create face-like region
            cv2.rectangle(thumbnail, (x, y), (x + w, y + h), base_color, -1)
            
            # Add "eyes"
            eye_color = (50, 50, 50)
            cv2.circle(thumbnail, (x + w//3, y + h//3), 6, eye_color, -1)
            cv2.circle(thumbnail, (x + 2*w//3, y + h//3), 6, eye_color, -1)
            
            # Add "mouth"
            cv2.ellipse(thumbnail, (x + w//2, y + 2*h//3), (w//4, h//8), 0, 0, 180, eye_color, 2)
        
        # Add some noise to make it more realistic
        noise = np.random.normal(0, 10, thumbnail.shape).astype(np.uint8)
        thumbnail = cv2.add(thumbnail, noise)
        
        # Save thumbnail
        thumbnail_path = output_dir / f"sample_thumbnail_{i:03d}.jpg"
        cv2.imwrite(str(thumbnail_path), thumbnail)
        image_paths.append(thumbnail_path)
        
        logger.info(f"Created sample thumbnail: {thumbnail_path.name}")
    
    return image_paths


def demonstrate_thumbnail_processing():
    """Demonstrate basic thumbnail processing functionality."""
    logger.info("="*60)
    logger.info("THUMBNAIL PROCESSING DEMONSTRATION")
    logger.info("="*60)
    
    # Create sample thumbnails
    thumbnails_dir = Path("example_thumbnails")
    logger.info(f"Creating sample thumbnails in {thumbnails_dir}")
    
    sample_images = create_sample_thumbnails(thumbnails_dir, num_images=5)
    logger.info(f"Created {len(sample_images)} sample thumbnail images")
    
    # Configure detection
    config = DetectionConfig(
        min_confidence=0.5,
        match_threshold=0.6,
        batch_size=4,
        device="cuda" if __name__ == "__main__" else "cpu",  # Use CUDA if available
        log_level="INFO"
    )
    
    logger.info(f"Using device: {config.device}")
    
    # Initialize thumbnail detector
    thumbnail_detector = ThumbnailDuplicateDetector(config)
    
    # Run thumbnail duplicate detection
    output_file = Path("example_thumbnail_results.json")
    logger.info("Running thumbnail duplicate detection...")
    
    start_time = time.time()
    report = thumbnail_detector.run_thumbnail_detection(
        thumbnails_dir=thumbnails_dir,
        output_file=output_file,
        compare_with_videos=False
    )
    processing_time = time.time() - start_time
    
    # Display results
    logger.info("="*40)
    logger.info("RESULTS SUMMARY")
    logger.info("="*40)
    logger.info(f"Processing time: {processing_time:.2f} seconds")
    logger.info(f"Total faces detected: {report.total_faces}")
    logger.info(f"Unique faces: {report.unique_faces}")
    logger.info(f"Duplicate groups: {report.duplicate_groups}")
    logger.info(f"Results saved to: {output_file}")
    
    # Analyze duplicate groups
    if report.detections:
        face_groups = {}
        for detection in report.detections:
            if detection.face_id not in face_groups:
                face_groups[detection.face_id] = []
            face_groups[detection.face_id].append(detection)
        
        logger.info("\nDetailed Analysis:")
        for face_id, detections in face_groups.items():
            images = [d.video_filename for d in detections]
            logger.info(f"  Face {face_id}: {len(detections)} detection(s)")
            logger.info(f"    Images: {images}")
            
            if len(detections) > 1:
                logger.info("    🔍 DUPLICATE DETECTED!")
    
    return report, thumbnails_dir, output_file


def demonstrate_cross_media_analysis():
    """Demonstrate cross-media duplicate analysis."""
    logger.info("="*60)
    logger.info("CROSS-MEDIA ANALYSIS DEMONSTRATION")
    logger.info("="*60)
    
    # Create simulated video detections
    video_detections = [
        FaceDetection(
            face_id="temp_video_face_1",
            video_filename="bicycle_video_001.mp4",
            timestamp="00:00:15.000",
            bounding_box=BoundingBox(x=100, y=150, width=80, height=100),
            confidence=0.85,
            embedding=np.random.randn(512).tolist()
        ),
        FaceDetection(
            face_id="temp_video_face_2",
            video_filename="bicycle_video_002.mp4",
            timestamp="00:00:32.000",
            bounding_box=BoundingBox(x=120, y=180, width=75, height=95),
            confidence=0.78,
            embedding=np.random.randn(512).tolist()
        )
    ]
    
    # Run thumbnail processing to get image detections
    report, thumbnails_dir, _ = demonstrate_thumbnail_processing()
    
    if not report.detections:
        logger.warning("No thumbnail detections found for cross-media analysis")
        return
    
    # Configure for cross-media analysis
    config = DetectionConfig(match_threshold=0.6, device="cpu")
    image_processor = ImageProcessor(config)
    
    # Find cross-media duplicates
    logger.info("Searching for cross-media duplicates...")
    cross_duplicates = image_processor.find_duplicates_between_videos_and_images(
        video_detections=video_detections,
        image_detections=report.detections
    )
    
    # Display cross-media results
    logger.info(f"\nCross-Media Analysis Results:")
    logger.info(f"  Video detections: {len(video_detections)}")
    logger.info(f"  Thumbnail detections: {len(report.detections)}")
    logger.info(f"  Cross-media duplicates: {len(cross_duplicates)}")
    
    if cross_duplicates:
        logger.info("\nCross-media duplicate groups:")
        for face_id, detections in cross_duplicates.items():
            video_sources = [d.video_filename for d in detections if d.video_filename.endswith('.mp4')]
            image_sources = [d.video_filename for d in detections if not d.video_filename.endswith('.mp4')]
            
            logger.info(f"  Face ID: {face_id}")
            logger.info(f"    Videos: {video_sources}")
            logger.info(f"    Images: {image_sources}")
            logger.info("    🔗 Same person appears in both videos and thumbnails!")
    else:
        logger.info("  No cross-media duplicates found in this demo")
        logger.info("  (This is expected with synthetic data)")


def demonstrate_similarity_analysis():
    """Demonstrate similarity matrix analysis."""
    logger.info("="*60)
    logger.info("SIMILARITY ANALYSIS DEMONSTRATION")
    logger.info("="*60)
    
    # Run thumbnail processing first
    report, _, _ = demonstrate_thumbnail_processing()
    
    if not report.detections or len(report.detections) < 2:
        logger.warning("Need at least 2 detections for similarity analysis")
        return
    
    # Configure image processor
    config = DetectionConfig(device="cpu")
    image_processor = ImageProcessor(config)
    
    # Generate similarity matrix
    logger.info("Generating similarity matrix...")
    similarity_matrix = image_processor.generate_image_similarity_matrix(report.detections)
    
    if similarity_matrix.size > 0:
        logger.info(f"Generated {similarity_matrix.shape[0]}x{similarity_matrix.shape[1]} similarity matrix")
        
        # Find high similarity pairs
        high_similarity_pairs = []
        n = similarity_matrix.shape[0]
        threshold = 0.6
        
        for i in range(n):
            for j in range(i+1, n):
                if similarity_matrix[i, j] > threshold:
                    high_similarity_pairs.append((i, j, similarity_matrix[i, j]))
        
        logger.info(f"\nHigh similarity pairs (threshold > {threshold}):")
        if high_similarity_pairs:
            for i, j, similarity in high_similarity_pairs:
                det_i = report.detections[i]
                det_j = report.detections[j]
                logger.info(f"  {det_i.video_filename} <-> {det_j.video_filename}: {similarity:.3f}")
        else:
            logger.info("  No high similarity pairs found")
        
        # Display matrix summary
        logger.info(f"\nSimilarity matrix statistics:")
        logger.info(f"  Mean similarity: {similarity_matrix.mean():.3f}")
        logger.info(f"  Max similarity (off-diagonal): {np.max(similarity_matrix - np.eye(n)):.3f}")
        logger.info(f"  Min similarity: {similarity_matrix.min():.3f}")
    else:
        logger.warning("Could not generate similarity matrix")


def cleanup_demo_files():
    """Clean up demonstration files."""
    logger.info("Cleaning up demonstration files...")
    
    import shutil
    
    cleanup_items = [
        "example_thumbnails",
        "example_thumbnail_results.json"
    ]
    
    for item in cleanup_items:
        path = Path(item)
        try:
            if path.is_dir():
                shutil.rmtree(path)
                logger.info(f"✓ Removed directory: {path}")
            elif path.is_file():
                path.unlink()
                logger.info(f"✓ Removed file: {path}")
        except Exception as e:
            logger.warning(f"Could not remove {path}: {e}")


def main():
    """Main demonstration function."""
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        colorize=True
    )
    
    logger.info("Starting Image Processing Demonstration")
    logger.info("This demo shows thumbnail duplicate detection capabilities")
    
    try:
        # Run demonstrations
        demonstrate_thumbnail_processing()
        print()
        
        demonstrate_cross_media_analysis()
        print()
        
        demonstrate_similarity_analysis()
        print()
        
        # Ask user about cleanup
        response = input("\nDo you want to clean up demo files? (y/n): ")
        if response.lower() in ['y', 'yes']:
            cleanup_demo_files()
        else:
            logger.info("Demo files preserved for inspection")
        
        logger.info("✅ Demonstration completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Demonstration interrupted by user")
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        logger.debug("Full traceback:", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
