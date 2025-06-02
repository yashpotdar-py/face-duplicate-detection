#!/usr/bin/env python3
"""
Command-line interface for duplicate face detection in bicycle videos.

This script provides a production-grade solution for detecting duplicate faces
across multiple video files using GPU-accelerated deep learning models.
"""
import sys
import time
from pathlib import Path
from typing import Optional

import typer
import torch
from loguru import logger

from src import DetectionConfig, DuplicateFaceDetector, ThumbnailDuplicateDetector

app = typer.Typer(
    name="detect_duplicates",
    help="Detect duplicate faces across bicycle videos using GPU acceleration",
    add_completion=False
)


def setup_logging(log_level: str) -> None:
    """Configure loguru logging with the specified level."""
    logger.remove()  # Remove default handler
    
    # Add console handler with colors
    logger.add(
        sys.stderr,
        level=log_level.upper(),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        colorize=True
    )
    
    # Add file handler for detailed logs
    log_file = Path("face_detection.log")
    logger.add(
        log_file,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="1 week"
    )
    
    logger.info(f"Logging configured at {log_level.upper()} level")
    logger.info(f"Detailed logs saved to: {log_file.absolute()}")


def check_gpu_availability() -> str:
    """Check GPU availability and return device string."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA available with {device_count} GPU(s)")
        logger.info(f"Using GPU: {device_name}")
        return "cuda"
    else:
        logger.warning("CUDA not available, falling back to CPU")
        logger.warning("Performance will be significantly slower on CPU")
        return "cpu"


@app.command()
def detect(
    videos_dir: Path = typer.Option(
        ...,
        "--videos-dir",
        help="Path to directory containing video files",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True
    ),
    output_file: Path = typer.Option(
        ...,
        "--output-file", 
        help="Path to save the duplicate detection report (JSON or CSV)"
    ),
    thumbnails_dir: Optional[Path] = typer.Option(
        None,
        "--thumbnails-dir",
        help="Optional path to thumbnails directory (currently not used)",
        exists=False,
        file_okay=False,
        dir_okay=True
    ),
    min_confidence: float = typer.Option(
        0.5,
        "--min-confidence",
        help="Minimum confidence threshold for face detection (0.0-1.0)",
        min=0.0,
        max=1.0
    ),
    match_threshold: float = typer.Option(
        0.6,
        "--match-threshold", 
        help="Distance threshold for face matching (lower = stricter)",
        min=0.0,
        max=2.0
    ),
    batch_size: int = typer.Option(
        16,
        "--batch-size",
        help="Number of frames to process per GPU batch",
        min=1,
        max=64
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level",
        case_sensitive=False
    ),
    skip_frames: int = typer.Option(
        30,
        "--skip-frames",
        help="Process every Nth frame (30 = ~1fps for 30fps video)",
        min=1,
        max=300
    ),
    force_cpu: bool = typer.Option(
        False,
        "--force-cpu",
        help="Force CPU usage even if GPU is available"
    )
) -> None:
    """
    Detect duplicate faces across bicycle videos.
    
    This command processes all video files in the specified directory,
    detects faces in each frame, and identifies duplicate faces that
    appear across multiple videos.
    
    Example usage:
    
    \b
    python detect_duplicates.py \\
        --videos-dir data/Videos \\
        --output-file duplicate_report.json \\
        --min-confidence 0.6 \\
        --match-threshold 0.5 \\
        --batch-size 8 \\
        --log-level INFO
    """
    # Setup logging
    setup_logging(log_level)
    
    logger.info("="*60)
    logger.info("Face Duplicate Detection System v1.0.0")
    logger.info("="*60)
    
    # Check system requirements
    device = "cpu" if force_cpu else check_gpu_availability()
    
    # Validate input parameters
    if not videos_dir.exists():
        logger.error(f"Videos directory does not exist: {videos_dir}")
        raise typer.Exit(1)
    
    # Check for video files
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
    video_files = [
        f for f in videos_dir.iterdir()
        if f.is_file() and f.suffix.lower() in video_extensions
    ]
    
    if not video_files:
        logger.error(f"No video files found in {videos_dir}")
        logger.info(f"Supported formats: {', '.join(video_extensions)}")
        raise typer.Exit(1)
    
    logger.info(f"Found {len(video_files)} video files to process")
    
    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create configuration
    config = DetectionConfig(
        min_confidence=min_confidence,
        match_threshold=match_threshold,
        batch_size=batch_size,
        log_level=log_level.upper(),
        device=device
    )
    
    # Log configuration
    logger.info("Configuration:")
    logger.info(f"  Videos directory: {videos_dir}")
    logger.info(f"  Output file: {output_file}")
    logger.info(f"  Min confidence: {min_confidence}")
    logger.info(f"  Match threshold: {match_threshold}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Skip frames: {skip_frames}")
    logger.info(f"  Device: {device}")
    
    try:
        # Initialize detector
        logger.info("Initializing face detection models...")
        detector = DuplicateFaceDetector(config)
        
        # Run detection pipeline
        start_time = time.time()
        report = detector.run_detection(
            videos_dir=videos_dir,
            output_file=output_file,
            thumbnails_dir=thumbnails_dir,
            skip_frames=skip_frames
        )
        
        # Display results summary
        total_time = time.time() - start_time
        logger.info("="*60)
        logger.info("DETECTION COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        logger.info(f"Total face detections: {report.total_faces}")
        logger.info(f"Unique faces found: {report.unique_faces}")
        logger.info(f"Duplicate groups: {report.duplicate_groups}")
        logger.info(f"Results saved to: {output_file}")
        
        # Performance metrics
        if report.total_faces > 0:
            fps = report.total_faces / total_time
            logger.info(f"Processing rate: {fps:.2f} detections/second")
        
        # Efficiency warning for CPU usage
        if device == "cpu":
            logger.warning("CPU processing detected. For better performance:")
            logger.warning("1. Install CUDA-enabled PyTorch")
            logger.warning("2. Ensure NVIDIA GPU drivers are installed")
            logger.warning("3. Remove --force-cpu flag")
        
    except KeyboardInterrupt:
        logger.info("Detection interrupted by user")
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        logger.debug("Full traceback:", exc_info=True)
        raise typer.Exit(1)


@app.command()
def detect_thumbnails(
    thumbnails_dir: Path = typer.Option(
        ...,
        "--thumbnails-dir",
        help="Path to directory containing thumbnail images",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True
    ),
    output_file: Path = typer.Option(
        ...,
        "--output-file", 
        help="Path to save the duplicate detection report (JSON or CSV)"
    ),
    min_confidence: float = typer.Option(
        0.5,
        "--min-confidence",
        help="Minimum confidence threshold for face detection (0.0-1.0)",
        min=0.0,
        max=1.0
    ),
    match_threshold: float = typer.Option(
        0.6,
        "--match-threshold", 
        help="Distance threshold for face matching (lower = stricter)",
        min=0.0,
        max=2.0
    ),
    batch_size: int = typer.Option(
        16,
        "--batch-size",
        help="Number of images to process per batch",
        min=1,
        max=64
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level",
        case_sensitive=False
    ),
    force_cpu: bool = typer.Option(
        False,
        "--force-cpu",
        help="Force CPU usage even if GPU is available"
    ),
    compare_with_videos: bool = typer.Option(
        False,
        "--compare-videos",
        help="Compare thumbnail faces with video detections (requires video results)"
    ),
    video_results_file: Optional[Path] = typer.Option(
        None,
        "--video-results",
        help="Path to video detection results file (JSON) for comparison",
        exists=False
    )
) -> None:
    """
    Detect duplicate faces in thumbnail images.
    
    This command processes all image files in the specified directory,
    detects faces in each image, and identifies duplicate faces that
    appear across multiple thumbnails.
    
    Example usage:
    
    \b
    python detect_duplicates.py detect-thumbnails \\
        --thumbnails-dir data/Thumbnails \\
        --output-file thumbnail_duplicates.json \\
        --min-confidence 0.6 \\
        --match-threshold 0.5 \\
        --batch-size 8 \\
        --log-level INFO
    """
    # Setup logging
    setup_logging(log_level)
    
    logger.info("="*60)
    logger.info("Thumbnail Face Duplicate Detection System v1.0.0")
    logger.info("="*60)
    
    # Check system requirements
    device = "cpu" if force_cpu else check_gpu_availability()
    
    # Validate input parameters
    if not thumbnails_dir.exists():
        logger.error(f"Thumbnails directory does not exist: {thumbnails_dir}")
        raise typer.Exit(1)
    
    # Check for image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_files = [
        f for f in thumbnails_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        logger.error(f"No image files found in {thumbnails_dir}")
        logger.info(f"Supported formats: {', '.join(image_extensions)}")
        raise typer.Exit(1)
    
    logger.info(f"Found {len(image_files)} image files to process")
    
    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create configuration
    config = DetectionConfig(
        min_confidence=min_confidence,
        match_threshold=match_threshold,
        batch_size=batch_size,
        log_level=log_level.upper(),
        device=device
    )
    
    # Log configuration
    logger.info("Configuration:")
    logger.info(f"  Thumbnails directory: {thumbnails_dir}")
    logger.info(f"  Output file: {output_file}")
    logger.info(f"  Min confidence: {min_confidence}")
    logger.info(f"  Match threshold: {match_threshold}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Compare with videos: {compare_with_videos}")
    
    try:
        # Load video detections if comparison is requested
        video_detections = None
        if compare_with_videos and video_results_file:
            if not video_results_file.exists():
                logger.error(f"Video results file does not exist: {video_results_file}")
                raise typer.Exit(1)
            
            logger.info(f"Loading video detections from: {video_results_file}")
            import json
            with open(video_results_file, 'r') as f:
                video_data = json.load(f)
                
            from src.models import FaceDetection, BoundingBox
            video_detections = []
            for det_data in video_data.get('detections', []):
                bbox = BoundingBox(**det_data['bounding_box'])
                detection = FaceDetection(
                    face_id=det_data['face_id'],
                    video_filename=det_data['video_filename'],
                    timestamp=det_data['timestamp'],
                    bounding_box=bbox,
                    confidence=det_data['confidence'],
                    embedding=det_data.get('embedding')
                )
                video_detections.append(detection)
            
            logger.info(f"Loaded {len(video_detections)} video detections")
        
        # Initialize thumbnail detector
        logger.info("Initializing thumbnail face detection models...")
        thumbnail_detector = ThumbnailDuplicateDetector(config)
        
        # Run thumbnail detection pipeline
        start_time = time.time()
        report = thumbnail_detector.run_thumbnail_detection(
            thumbnails_dir=thumbnails_dir,
            output_file=output_file,
            compare_with_videos=compare_with_videos,
            video_detections=video_detections
        )
        
        # Display results summary
        total_time = time.time() - start_time
        logger.info("="*60)
        logger.info("THUMBNAIL DETECTION COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        logger.info(f"Total face detections: {report.total_faces}")
        logger.info(f"Unique faces found: {report.unique_faces}")
        logger.info(f"Duplicate groups: {report.duplicate_groups}")
        logger.info(f"Results saved to: {output_file}")
        
        # Performance metrics
        if report.total_faces > 0:
            fps = report.total_faces / total_time
            logger.info(f"Processing rate: {fps:.2f} detections/second")
        
        # Efficiency warning for CPU usage
        if device == "cpu":
            logger.warning("CPU processing detected. For better performance:")
            logger.warning("1. Install CUDA-enabled PyTorch")
            logger.warning("2. Ensure NVIDIA GPU drivers are installed")
            logger.warning("3. Remove --force-cpu flag")
        
    except KeyboardInterrupt:
        logger.info("Detection interrupted by user")
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        logger.debug("Full traceback:", exc_info=True)
        raise typer.Exit(1)


@app.command()
def compare_results(
    video_results: Path = typer.Option(
        ...,
        "--video-results",
        help="Path to video detection results file (JSON)",
        exists=True
    ),
    thumbnail_results: Path = typer.Option(
        ...,
        "--thumbnail-results", 
        help="Path to thumbnail detection results file (JSON)",
        exists=True
    ),
    output_file: Path = typer.Option(
        ...,
        "--output-file",
        help="Path to save the comparison report"
    ),
    match_threshold: float = typer.Option(
        0.6,
        "--match-threshold",
        help="Distance threshold for cross-medium matching",
        min=0.0,
        max=2.0
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level",
        case_sensitive=False
    )
) -> None:
    """
    Compare face detection results between videos and thumbnails.
    
    This command loads detection results from both video and thumbnail processing
    and identifies faces that appear in both mediums.
    
    Example usage:
    
    \b
    python detect_duplicates.py compare-results \\
        --video-results video_duplicates.json \\
        --thumbnail-results thumbnail_duplicates.json \\
        --output-file cross_media_comparison.json \\
        --match-threshold 0.5
    """
    # Setup logging
    setup_logging(log_level)
    
    logger.info("="*60)
    logger.info("Cross-Media Face Comparison")
    logger.info("="*60)
    
    try:
        # Load results
        logger.info("Loading detection results...")
        
        import json
        from src.models import FaceDetection, BoundingBox
        
        # Load video results
        with open(video_results, 'r') as f:
            video_data = json.load(f)
        
        video_detections = []
        for det_data in video_data.get('detections', []):
            bbox = BoundingBox(**det_data['bounding_box'])
            detection = FaceDetection(
                face_id=det_data['face_id'],
                video_filename=det_data['video_filename'],
                timestamp=det_data['timestamp'],
                bounding_box=bbox,
                confidence=det_data['confidence'],
                embedding=det_data.get('embedding')
            )
            video_detections.append(detection)
        
        # Load thumbnail results
        with open(thumbnail_results, 'r') as f:
            thumbnail_data = json.load(f)
        
        thumbnail_detections = []
        for det_data in thumbnail_data.get('detections', []):
            bbox = BoundingBox(**det_data['bounding_box'])
            detection = FaceDetection(
                face_id=det_data['face_id'],
                video_filename=det_data['video_filename'],
                timestamp=det_data['timestamp'],
                bounding_box=bbox,
                confidence=det_data['confidence'],
                embedding=det_data.get('embedding')
            )
            thumbnail_detections.append(detection)
        
        logger.info(f"Loaded {len(video_detections)} video detections")
        logger.info(f"Loaded {len(thumbnail_detections)} thumbnail detections")
        
        # Create configuration for comparison
        config = DetectionConfig(match_threshold=match_threshold)
        
        # Initialize image processor for comparison
        from src.image_processor import ImageProcessor
        processor = ImageProcessor(config)
        
        # Find cross-medium duplicates
        cross_duplicates = processor.find_duplicates_between_videos_and_images(
            video_detections=video_detections,
            image_detections=thumbnail_detections
        )
        
        # Save comparison results
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        comparison_data = {
            "summary": {
                "video_detections": len(video_detections),
                "thumbnail_detections": len(thumbnail_detections),
                "cross_medium_duplicates": len(cross_duplicates),
                "match_threshold": match_threshold,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "cross_duplicates": {}
        }
        
        for face_id, detections in cross_duplicates.items():
            comparison_data["cross_duplicates"][face_id] = [det.dict() for det in detections]
        
        with open(output_file, 'w') as f:
            json.dump(comparison_data, f, indent=2, default=str)
        
        # Display results
        logger.info("="*60)
        logger.info("COMPARISON COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Video detections: {len(video_detections)}")
        logger.info(f"Thumbnail detections: {len(thumbnail_detections)}")
        logger.info(f"Cross-medium duplicates: {len(cross_duplicates)}")
        logger.info(f"Results saved to: {output_file}")
        
        if cross_duplicates:
            logger.info("\nCross-medium duplicate summary:")
            for face_id, detections in cross_duplicates.items():
                video_count = sum(1 for d in detections if d.video_filename.endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')))
                image_count = len(detections) - video_count
                logger.info(f"  {face_id}: {video_count} video(s), {image_count} thumbnail(s)")
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        logger.debug("Full traceback:", exc_info=True)
        raise typer.Exit(1)


@app.command()
def version() -> None:
    """Display version information."""
    typer.echo("Face Duplicate Detection System v1.0.0")
    typer.echo("GPU-accelerated face detection and clustering for video analysis")


@app.command()
def check_gpu() -> None:
    """Check GPU availability and CUDA installation."""
    typer.echo("Checking GPU availability...")
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        typer.echo(f"✓ CUDA available with {device_count} GPU(s)")
        
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            memory_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
            typer.echo(f"  GPU {i}: {device_name} ({memory_gb:.1f} GB)")
        
        typer.echo(f"✓ PyTorch version: {torch.__version__}")
        typer.echo(f"✓ CUDA version: {torch.version.cuda}")
    else:
        typer.echo("✗ CUDA not available")
        typer.echo("Install CUDA-enabled PyTorch for GPU acceleration:")
        typer.echo("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")


if __name__ == "__main__":
    app()
