#!/usr/bin/env python3
"""
GPU-Accelerated Face Duplicate Detection System
Main CLI Application

Usage:
    python main.py --help
    python main.py process-images --input-dir data/Thumbnails
    python main.py process-videos --input-dir data/Videos
    python main.py process-all --data-dir data
"""

import typer
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich import print as rprint
import time
import json

from src.core.face_detector import FaceDetector
from src.core.duplicate_finder import DuplicateFinder
from src.core.video_processor import VideoProcessor
from src.utils.config import Config
from src.utils.logging import setup_logging, get_logger
from src.utils.visualizer import ResultVisualizer

# Initialize Typer app and Rich console
app = typer.Typer(
    name="face-duplicate-detector",
    help="🔍 GPU-Accelerated Face Duplicate Detection System",
    rich_markup_mode="rich"
)
console = Console()
logger = get_logger(__name__)


def show_system_info(config: Config):
    """Display system and configuration information."""
    device_info = config.get_device_info()
    
    # Create system info table
    table = Table(title="🖥️ System Information", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan", width=20)
    table.add_column("Value", style="green")
    
    table.add_row("Device", device_info["device"])
    table.add_row("CUDA Available", str(device_info["cuda_available"]))
    table.add_row("GPU Count", str(device_info["device_count"]))
    
    if device_info["cuda_available"]:
        table.add_row("GPU Name", device_info.get("device_name", "Unknown"))
        table.add_row("Memory Allocated", f"{device_info.get('memory_allocated', 0) / 1024**3:.2f} GB")
        table.add_row("Memory Reserved", f"{device_info.get('memory_reserved', 0) / 1024**3:.2f} GB")
    
    table.add_row("Similarity Threshold", str(config.similarity_threshold))
    table.add_row("Batch Size", str(config.batch_size))
    
    console.print(table)
    console.print()


@app.command()
def info(
    data_dir: str = typer.Option("data", "--data-dir", "-d", help="Data directory path")
):
    """
    📊 Show system information and data statistics.
    """
    config = Config(data_dir=data_dir)
    setup_logging(config.log_level, "results/face_detection.log")
    
    rprint("\n[bold blue]🚀 Face Duplicate Detection System[/bold blue]\n")
    
    show_system_info(config)
    
    # Show data directory info
    data_path = Path(data_dir)
    if data_path.exists():
        thumbnails_dir = data_path / "Thumbnails"
        videos_dir = data_path / "Videos"
        
        thumb_count = len(list(thumbnails_dir.glob("*.*"))) if thumbnails_dir.exists() else 0
        video_count = len(list(videos_dir.glob("*.*"))) if videos_dir.exists() else 0
        
        data_table = Table(title="📁 Data Directory Statistics", show_header=True, header_style="bold cyan")
        data_table.add_column("Type", style="yellow")
        data_table.add_column("Count", style="green")
        data_table.add_column("Directory", style="blue")
        
        data_table.add_row("Images", str(thumb_count), str(thumbnails_dir))
        data_table.add_row("Videos", str(video_count), str(videos_dir))
        
        console.print(data_table)
    else:
        rprint(f"[red]❌ Data directory not found: {data_dir}[/red]")


@app.command()
def process_images(
    input_dir: str = typer.Option("data/Thumbnails", "--input-dir", "-i", help="Input directory with images"),
    output_dir: str = typer.Option("results", "--output-dir", "-o", help="Output directory for results"),
    threshold: float = typer.Option(0.6, "--threshold", "-t", help="Similarity threshold for duplicates"),
    use_clustering: bool = typer.Option(False, "--clustering", help="Use clustering instead of threshold"),
    eps: float = typer.Option(0.3, "--eps", help="DBSCAN eps parameter (max distance for clustering)"),
    min_samples: int = typer.Option(2, "--min-samples", help="DBSCAN min_samples parameter"),
    save_crops: bool = typer.Option(True, "--save-crops", help="Save face crops"),
    generate_viz: bool = typer.Option(True, "--visualize", help="Generate visualizations"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Processing batch size")
):
    """
    🖼️ Process images to detect duplicate faces.
    """
    start_time = time.time()
    
    # Setup configuration
    config = Config(
        thumbnails_dir=input_dir,
        results_dir=output_dir,
        similarity_threshold=threshold,
        batch_size=batch_size,
        save_face_crops=save_crops
    )
    
    setup_logging(config.log_level, f"{output_dir}/face_detection.log")
    
    rprint(f"\n[bold green]🖼️ Processing Images from: {input_dir}[/bold green]\n")
    show_system_info(config)
    
    # Initialize components
    face_detector = FaceDetector(config)
    duplicate_finder = DuplicateFinder(config)
    
    # Find image files
    input_path = Path(input_dir)
    if not input_path.exists():
        rprint(f"[red]❌ Input directory not found: {input_dir}[/red]")
        raise typer.Exit(1)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [
        f for f in input_path.rglob("*") 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        rprint(f"[red]❌ No image files found in {input_dir}[/red]")
        raise typer.Exit(1)
    
    rprint(f"[cyan]📸 Found {len(image_files)} image files[/cyan]\n")
    
    # Process images with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("Processing images...", total=len(image_files))
        
        for image_file in image_files:
            try:
                # Process image
                face_data = face_detector.process_image(str(image_file))
                duplicate_finder.add_faces(face_data)
                
                progress.update(task, advance=1, description=f"Processed: {image_file.name}")
                
            except Exception as e:
                logger.error(f"Error processing {image_file}: {e}")
                progress.update(task, advance=1)
    
    # Find duplicates
    rprint("\n[yellow]🔍 Analyzing face similarities...[/yellow]")
    
    with Progress(SpinnerColumn(), TextColumn("Computing similarities..."), console=console) as progress:
        progress.add_task("Computing...", total=None)
        duplicate_finder.compute_similarity_matrix()
    
    # Find duplicate groups
    if use_clustering:
        rprint(f"[cyan]Using clustering method (eps={eps}, min_samples={min_samples})...[/cyan]")
        duplicate_groups = duplicate_finder.find_duplicates_clustering(eps=eps, min_samples=min_samples)
    else:
        rprint(f"[cyan]Using threshold method (threshold={threshold})...[/cyan]")
        duplicate_groups = duplicate_finder.find_duplicates_threshold(threshold)
    
    # Get report right after finding duplicates
    report = duplicate_finder.get_duplicate_report()
    
    # Generate and save results
    rprint("\n[yellow]💾 Generating results...[/yellow]")
    saved_files = duplicate_finder.save_results(output_dir)
    
    # Display results summary
    summary = report.get("summary", {})
    
    results_table = Table(title="🎯 Results Summary", show_header=True, header_style="bold green")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="yellow")
    
    results_table.add_row("Total Faces Detected", str(summary.get("total_faces", 0)))
    results_table.add_row("Duplicate Groups Found", str(summary.get("duplicate_groups", 0)))
    results_table.add_row("Total Duplicate Faces", str(summary.get("total_duplicates", 0)))
    results_table.add_row("Unique Faces", str(summary.get("unique_faces", 0)))
    
    console.print(results_table)
    
    # Show processing time
    processing_time = time.time() - start_time
    rprint(f"\n[green]✅ Processing completed in {processing_time:.2f} seconds[/green]")
    
    # Show memory usage
    memory_info = face_detector.get_memory_usage()
    if "allocated_gb" in memory_info:
        rprint(f"[blue]📊 GPU Memory Used: {memory_info['allocated_gb']:.2f} GB[/blue]")
    
    # Show saved files
    if saved_files:
        rprint(f"\n[bold cyan]💾 Results saved to:[/bold cyan]")
        for file_type, file_path in saved_files.items():
            rprint(f"  • {file_type}: {file_path}")
    
    # Generate visualizations
    rprint(f"\n[cyan]DEBUG: generate_viz={generate_viz}, face_database_length={len(duplicate_finder.face_database)}[/cyan]")
    if generate_viz and len(duplicate_finder.face_database) > 0:
        rprint("\n[yellow]🎨 Generating visualizations...[/yellow]")
        
        try:
            viz_output_dir = str(Path(output_dir) / "visualizations")
            visualizer = ResultVisualizer(viz_output_dir)
            
            # Use structured duplicate groups from the report instead of raw indices
            report_duplicate_groups = report.get("duplicate_groups", [])
            
            viz_files = visualizer.generate_all_visualizations(
                similarity_matrix=duplicate_finder.similarity_matrix,
                duplicate_groups=report_duplicate_groups,
                face_database=duplicate_finder.face_database
            )
            
            if viz_files:
                rprint(f"\n[bold magenta]🎨 Visualizations generated:[/bold magenta]")
                for viz_type, viz_path in viz_files.items():
                    rprint(f"  • {viz_type}: {viz_path}")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            rprint(f"[red]❌ Failed to generate visualizations: {e}[/red]")


@app.command()
def process_videos(
    input_dir: str = typer.Option("data/Videos", "--input-dir", "-i", help="Input directory with videos"),
    output_dir: str = typer.Option("results", "--output-dir", "-o", help="Output directory for results"),
    threshold: float = typer.Option(0.6, "--threshold", "-t", help="Similarity threshold for duplicates"),
    frame_skip: int = typer.Option(30, "--frame-skip", "-f", help="Process every Nth frame"),
    max_frames: int = typer.Option(100, "--max-frames", "-m", help="Maximum frames per video"),
    use_clustering: bool = typer.Option(False, "--clustering", help="Use clustering instead of threshold"),
    eps: float = typer.Option(0.3, "--eps", help="DBSCAN eps parameter (max distance for clustering)"),
    min_samples: int = typer.Option(2, "--min-samples", help="DBSCAN min_samples parameter"),
    batch_size: int = typer.Option(16, "--batch-size", "-b", help="Processing batch size")
):
    """
    🎬 Process videos to detect duplicate faces across frames.
    """
    start_time = time.time()
    
    # Setup configuration
    config = Config(
        videos_dir=input_dir,
        results_dir=output_dir,
        similarity_threshold=threshold,
        video_frame_skip=frame_skip,
        batch_size=batch_size
    )
    
    setup_logging(config.log_level, f"{output_dir}/face_detection.log")
    
    rprint(f"\n[bold blue]🎬 Processing Videos from: {input_dir}[/bold blue]\n")
    show_system_info(config)
    
    # Initialize components
    face_detector = FaceDetector(config)
    duplicate_finder = DuplicateFinder(config)
    video_processor = VideoProcessor(config)
    
    # Find video files
    input_path = Path(input_dir)
    if not input_path.exists():
        rprint(f"[red]❌ Input directory not found: {input_dir}[/red]")
        raise typer.Exit(1)
    
    video_extensions = video_processor.get_supported_formats()
    video_files = [
        f for f in input_path.rglob("*") 
        if f.is_file() and f.suffix.lower() in video_extensions
    ]
    
    if not video_files:
        rprint(f"[red]❌ No video files found in {input_dir}[/red]")
        raise typer.Exit(1)
    
    rprint(f"[cyan]🎬 Found {len(video_files)} video files[/cyan]\n")
    
    # Process videos with progress bar
    total_frames_processed = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        video_task = progress.add_task("Processing videos...", total=len(video_files))
        
        for video_file in video_files:
            try:
                # Validate video
                if not video_processor.validate_video_file(str(video_file)):
                    logger.warning(f"Skipping invalid video: {video_file}")
                    progress.update(video_task, advance=1)
                    continue
                
                video_name = video_file.stem
                frame_count = 0
                
                # Process frames
                for frame_number, frame in video_processor.extract_frames_smart(
                    str(video_file), target_frames=max_frames
                ):
                    face_data = face_detector.process_video_frame(frame, frame_number, video_name)
                    duplicate_finder.add_faces(face_data)
                    frame_count += 1
                
                total_frames_processed += frame_count
                progress.update(
                    video_task, 
                    advance=1, 
                    description=f"Processed: {video_name} ({frame_count} frames)"
                )
                
            except Exception as e:
                logger.error(f"Error processing {video_file}: {e}")
                progress.update(video_task, advance=1)
    
    rprint(f"\n[green]📊 Total frames processed: {total_frames_processed}[/green]")
    
    # Find duplicates
    rprint("\n[yellow]🔍 Analyzing face similarities across videos...[/yellow]")
    
    with Progress(SpinnerColumn(), TextColumn("Computing similarities..."), console=console) as progress:
        progress.add_task("Computing...", total=None)
        duplicate_finder.compute_similarity_matrix()
    
    # Find duplicate groups
    if use_clustering:
        rprint(f"[cyan]Using clustering method (eps={eps}, min_samples={min_samples})...[/cyan]")
        duplicate_groups = duplicate_finder.find_duplicates_clustering(eps=eps, min_samples=min_samples)
    else:
        rprint(f"[cyan]Using threshold method (threshold={threshold})...[/cyan]")
        duplicate_groups = duplicate_finder.find_duplicates_threshold(threshold)
    
    # Generate and save results
    rprint("\n[yellow]💾 Generating results...[/yellow]")
    saved_files = duplicate_finder.save_results(output_dir)
    
    # Enhanced results display
    total_faces = len(duplicate_finder.face_database)
    report = duplicate_finder.get_duplicate_report()
    summary = report.get("summary", {})
    
    if total_faces == 0:
        rprint("\n[yellow]⚠️  No faces detected in video frames.[/yellow]")
        rprint("[dim]This could be because:[/dim]")
        rprint("[dim]  • Videos contain no people (e.g., landscapes, transportation, etc.)[/dim]")
        rprint("[dim]  • Faces are too small or low quality to detect[/dim]")
        rprint("[dim]  • Lighting conditions make detection difficult[/dim]")
        rprint("[dim]  • Video compression artifacts affect face detection[/dim]")
        rprint("\n[cyan]💡 Tip: Try with videos containing clear, frontal faces[/cyan]")
        
        # Still show basic stats
        stats_table = Table(title="🎯 Video Processing Results", show_header=True, header_style="bold yellow")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="yellow")
        
        stats_table.add_row("Videos Processed", str(len(video_files)))
        stats_table.add_row("Frames Processed", str(total_frames_processed))
        stats_table.add_row("Total Faces Detected", str(total_faces))
        
        console.print(stats_table)
    else:
        results_table = Table(title="🎯 Video Processing Results", show_header=True, header_style="bold green")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="yellow")
        
        results_table.add_row("Videos Processed", str(len(video_files)))
        results_table.add_row("Frames Processed", str(total_frames_processed))
        results_table.add_row("Total Faces Detected", str(total_faces))
        results_table.add_row("Duplicate Groups Found", str(summary.get("duplicate_groups", 0)))
        results_table.add_row("Total Duplicate Faces", str(summary.get("total_duplicates", 0)))
        
        console.print(results_table)
    
    # Show processing time
    processing_time = time.time() - start_time
    rprint(f"\n[green]✅ Processing completed in {processing_time:.2f} seconds[/green]")
    
    # Show memory usage
    memory_info = face_detector.get_memory_usage()
    if "allocated_gb" in memory_info:
        rprint(f"[blue]📊 GPU Memory Used: {memory_info['allocated_gb']:.2f} GB[/blue]")
    
    # Show saved files
    if saved_files:
        rprint(f"\n[bold cyan]💾 Results saved to:[/bold cyan]")
        for file_type, file_path in saved_files.items():
            rprint(f"  • {file_type}: {file_path}")


@app.command()
def process_all(
    data_dir: str = typer.Option("data", "--data-dir", "-d", help="Data directory containing Thumbnails and Videos"),
    output_dir: str = typer.Option("results", "--output-dir", "-o", help="Output directory for results"),
    threshold: float = typer.Option(0.6, "--threshold", "-t", help="Similarity threshold for duplicates"),
    frame_skip: int = typer.Option(30, "--frame-skip", "-f", help="Process every Nth frame for videos"),
    max_frames: int = typer.Option(100, "--max-frames", "-m", help="Maximum frames per video"),
    use_clustering: bool = typer.Option(False, "--clustering", help="Use clustering instead of threshold"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Processing batch size")
):
    """
    🚀 Process both images and videos to find all duplicate faces.
    """
    start_time = time.time()
    
    data_path = Path(data_dir)
    if not data_path.exists():
        rprint(f"[red]❌ Data directory not found: {data_dir}[/red]")
        raise typer.Exit(1)
    
    thumbnails_dir = data_path / "Thumbnails"
    videos_dir = data_path / "Videos"
    
    rprint(f"\n[bold purple]🚀 Processing All Data from: {data_dir}[/bold purple]\n")
    
    # Process images if directory exists
    if thumbnails_dir.exists():
        rprint("[yellow]📸 Processing Images...[/yellow]")
        try:
            # Use the existing process_images logic but call directly
            ctx = typer.Context(process_images)
            ctx.invoke(
                process_images,
                input_dir=str(thumbnails_dir),
                output_dir=output_dir,
                threshold=threshold,
                use_clustering=use_clustering,
                save_crops=True,
                batch_size=batch_size
            )
        except Exception as e:
            logger.error(f"Error processing images: {e}")
    
    # Process videos if directory exists
    if videos_dir.exists():
        rprint("\n[yellow]🎬 Processing Videos...[/yellow]")
        try:
            # Use the existing process_videos logic but call directly
            ctx = typer.Context(process_videos)
            ctx.invoke(
                process_videos,
                input_dir=str(videos_dir),
                output_dir=output_dir,
                threshold=threshold,
                frame_skip=frame_skip,
                max_frames=max_frames,
                use_clustering=use_clustering,
                batch_size=batch_size
            )
        except Exception as e:
            logger.error(f"Error processing videos: {e}")
    
    total_time = time.time() - start_time
    rprint(f"\n[bold green]🎉 All processing completed in {total_time:.2f} seconds![/bold green]")


@app.command()
def visualize(
    results_dir: str = typer.Option("results", "--results-dir", "-r", help="Results directory path"),
    output_dir: str = typer.Option("results/visualizations", "--output-dir", "-o", help="Output directory for visualizations"),
    benchmark_file: Optional[str] = typer.Option(None, "--benchmark-file", "-b", help="Benchmark results JSON file")
):
    """
    🎨 Generate visualizations from existing results.
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        rprint(f"[red]❌ Results directory not found: {results_dir}[/red]")
        raise typer.Exit(1)
    
    rprint(f"\n[bold magenta]🎨 Generating Visualizations from: {results_dir}[/bold magenta]\n")
    
    # Initialize visualizer
    visualizer = ResultVisualizer(output_dir)
    
    # Load existing results
    try:
        # Load similarity matrix
        similarity_matrix = None
        sim_matrix_file = results_path / "similarity_matrix.npy"
        if sim_matrix_file.exists():
            similarity_matrix = np.load(sim_matrix_file)
            rprint("[green]✅ Loaded similarity matrix[/green]")
        
        # Load face database
        face_database = None
        face_db_file = results_path / "face_database.json"
        if face_db_file.exists():
            with open(face_db_file, 'r') as f:
                face_database = json.load(f)
            rprint("[green]✅ Loaded face database[/green]")
        
        # Load duplicate report to get groups
        duplicate_groups = []
        report_file = results_path / "duplicate_report.json"
        if report_file.exists():
            with open(report_file, 'r') as f:
                report = json.load(f)
                duplicate_groups = report.get("duplicate_groups", [])
            rprint("[green]✅ Loaded duplicate groups[/green]")
        
        # Load benchmark results if specified
        benchmark_results = None
        if benchmark_file:
            bench_path = Path(benchmark_file)
            if bench_path.exists():
                with open(bench_path, 'r') as f:
                    benchmark_results = json.load(f)
                rprint("[green]✅ Loaded benchmark results[/green]")
            else:
                rprint(f"[yellow]⚠️  Benchmark file not found: {benchmark_file}[/yellow]")
        
        # Generate visualizations
        rprint("\n[yellow]🎨 Creating visualizations...[/yellow]")
        
        viz_files = visualizer.generate_all_visualizations(
            similarity_matrix=similarity_matrix,
            duplicate_groups=duplicate_groups,
            face_database=face_database,
            benchmark_results=benchmark_results
        )
        
        if viz_files:
            # Create results table
            viz_table = Table(title="🎨 Generated Visualizations", show_header=True, header_style="bold magenta")
            viz_table.add_column("Visualization Type", style="cyan", width=25)
            viz_table.add_column("File Path", style="green")
            
            for viz_type, viz_path in viz_files.items():
                viz_table.add_row(viz_type.replace('_', ' ').title(), viz_path)
            
            console.print(viz_table)
            rprint(f"\n[bold green]✅ Generated {len(viz_files)} visualizations successfully![/bold green]")
        else:
            rprint("[yellow]⚠️  No visualizations were generated. Check if required data files exist.[/yellow]")
            
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
        rprint(f"[red]❌ Failed to generate visualizations: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def benchmark(
    data_dir: str = typer.Option("data", "--data-dir", "-d", help="Data directory path"),
    iterations: int = typer.Option(3, "--iterations", "-n", help="Number of benchmark iterations"),
    test_size: int = typer.Option(10, "--test-size", "-s", help="Number of images/videos to test"),
    save_results: bool = typer.Option(True, "--save-results", help="Save benchmark results to file")
):
    """
    🏃‍♂️ Run performance benchmarks comparing GPU vs CPU processing.
    """
    import gc
    import psutil
    import statistics
    from datetime import datetime
    
    rprint("\n[bold yellow]🏃‍♂️ Running Performance Benchmarks[/bold yellow]\n")
    
    # Setup paths
    data_path = Path(data_dir)
    if not data_path.exists():
        rprint(f"[red]❌ Data directory not found: {data_dir}[/red]")
        raise typer.Exit(1)
    
    thumbnails_dir = data_path / "Thumbnails"
    
    # Check if we have test data
    if not thumbnails_dir.exists() or not list(thumbnails_dir.glob("*.jpg")):
        rprint("[yellow]⚠️  No thumbnail images found. Creating test data...[/yellow]")
        _create_benchmark_test_data(data_path)
    
    # Get test files
    image_files = list(thumbnails_dir.glob("*.jpg"))[:test_size]
    
    if len(image_files) < 3:
        rprint(f"[red]❌ Need at least 3 images for benchmarking, found {len(image_files)}[/red]")
        raise typer.Exit(1)
    
    rprint(f"[cyan]📊 Benchmarking with {len(image_files)} images, {iterations} iterations[/cyan]\n")
    
    # Initialize benchmark results storage
    benchmark_results = {
        "timestamp": datetime.now().isoformat(),
        "system_info": {},
        "test_parameters": {
            "test_size": len(image_files),
            "iterations": iterations,
            "data_dir": str(data_dir)
        },
        "gpu_results": [],
        "cpu_results": [],
        "memory_usage": []
    }
    
    # Get system info
    try:
        # CPU info
        cpu_info = {
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            "memory_total": psutil.virtual_memory().total / (1024**3)  # GB
        }
        
        # GPU info
        if torch.cuda.is_available():
            gpu_info = {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory": torch.cuda.get_device_properties(0).total_memory / (1024**3),  # GB
                "cuda_version": torch.version.cuda
            }
        else:
            gpu_info = {"available": False}
        
        benchmark_results["system_info"] = {
            "cpu": cpu_info,
            "gpu": gpu_info
        }
        
        # Display system info
        sys_table = Table(title="🖥️ Benchmark System Info", show_header=True, header_style="bold magenta")
        sys_table.add_column("Component", style="cyan", width=15)
        sys_table.add_column("Details", style="green")
        
        sys_table.add_row("CPU Cores", str(cpu_info["cpu_count"]))
        sys_table.add_row("RAM", f"{cpu_info['memory_total']:.1f} GB")
        if gpu_info.get("available", True):
            sys_table.add_row("GPU", gpu_info["gpu_name"])
            sys_table.add_row("GPU Memory", f"{gpu_info['gpu_memory']:.1f} GB")
        else:
            sys_table.add_row("GPU", "Not Available")
        
        console.print(sys_table)
        
    except Exception as e:
        logger.warning(f"Could not gather system info: {e}")
    
    # Run GPU benchmarks
    if torch.cuda.is_available():
        rprint("\n[bold green]🚀 Running GPU Benchmarks...[/bold green]")
        gpu_times = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task("GPU Benchmark", total=iterations)
            
            for i in range(iterations):
                # Force GPU mode
                config = Config(
                    thumbnails_dir=str(thumbnails_dir),
                    results_dir="results/benchmark_gpu",
                    device="cuda",
                    use_gpu=True
                )
                
                # Clear cache
                torch.cuda.empty_cache()
                gc.collect()
                
                start_time = time.time()
                
                # Initialize components
                face_detector = FaceDetector(config)
                duplicate_finder = DuplicateFinder(config)
                
                # Process images
                for img_file in image_files:
                    face_data = face_detector.process_image(str(img_file))
                    duplicate_finder.add_faces(face_data)
                
                # Find duplicates
                duplicate_finder.compute_similarity_matrix()
                duplicate_finder.find_duplicates_threshold(0.6)
                
                end_time = time.time()
                iteration_time = end_time - start_time
                gpu_times.append(iteration_time)
                
                # Memory usage
                memory_used = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
                benchmark_results["memory_usage"].append({
                    "iteration": i + 1,
                    "device": "gpu",
                    "memory_gb": memory_used
                })
                
                progress.update(task, advance=1, 
                              description=f"GPU Iteration {i+1}/{iterations} - {iteration_time:.2f}s")
                
                # Clean up
                del face_detector, duplicate_finder
                torch.cuda.empty_cache()
                gc.collect()
        
        benchmark_results["gpu_results"] = gpu_times
        
        # GPU Stats
        gpu_stats = {
            "mean": statistics.mean(gpu_times),
            "median": statistics.median(gpu_times),
            "std": statistics.stdev(gpu_times) if len(gpu_times) > 1 else 0,
            "min": min(gpu_times),
            "max": max(gpu_times)
        }
        
        rprint(f"[green]✅ GPU Average: {gpu_stats['mean']:.2f}s (±{gpu_stats['std']:.2f}s)[/green]")
    
    # Run CPU benchmarks
    rprint("\n[bold blue]🔧 Running CPU Benchmarks...[/bold blue]")
    cpu_times = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("CPU Benchmark", total=iterations)
        
        for i in range(iterations):
            # Force CPU mode
            config = Config(
                thumbnails_dir=str(thumbnails_dir),
                results_dir="results/benchmark_cpu",
                device="cpu",
                use_gpu=False
            )
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            start_time = time.time()
            
            # Initialize components
            face_detector = FaceDetector(config)
            duplicate_finder = DuplicateFinder(config)
            
            # Process images
            for img_file in image_files:
                face_data = face_detector.process_image(str(img_file))
                duplicate_finder.add_faces(face_data)
            
            # Find duplicates
            duplicate_finder.compute_similarity_matrix()
            duplicate_finder.find_duplicates_threshold(0.6)
            
            end_time = time.time()
            iteration_time = end_time - start_time
            cpu_times.append(iteration_time)
            
            progress.update(task, advance=1, 
                          description=f"CPU Iteration {i+1}/{iterations} - {iteration_time:.2f}s")
            
            # Clean up
            del face_detector, duplicate_finder
            gc.collect()
    
    benchmark_results["cpu_results"] = cpu_times
    
    # CPU Stats
    cpu_stats = {
        "mean": statistics.mean(cpu_times),
        "median": statistics.median(cpu_times),
        "std": statistics.stdev(cpu_times) if len(cpu_times) > 1 else 0,
        "min": min(cpu_times),
        "max": max(cpu_times)
    }
    
    rprint(f"[blue]✅ CPU Average: {cpu_stats['mean']:.2f}s (±{cpu_stats['std']:.2f}s)[/blue]")
    
    # Create comparison table
    rprint("\n")
    comp_table = Table(title="📊 Performance Comparison", show_header=True, header_style="bold yellow")
    comp_table.add_column("Device", style="cyan", width=10)
    comp_table.add_column("Mean Time", style="yellow", width=12)
    comp_table.add_column("Std Dev", style="yellow", width=10)
    comp_table.add_column("Min Time", style="green", width=10)
    comp_table.add_column("Max Time", style="red", width=10)
    comp_table.add_column("Speedup", style="magenta", width=10)
    
    if torch.cuda.is_available() and gpu_times:
        speedup = cpu_stats["mean"] / gpu_stats["mean"]
        comp_table.add_row(
            "GPU", 
            f"{gpu_stats['mean']:.2f}s",
            f"±{gpu_stats['std']:.2f}s",
            f"{gpu_stats['min']:.2f}s",
            f"{gpu_stats['max']:.2f}s",
            f"{speedup:.2f}x"
        )
        benchmark_results["speedup"] = speedup
    
    comp_table.add_row(
        "CPU",
        f"{cpu_stats['mean']:.2f}s", 
        f"±{cpu_stats['std']:.2f}s",
        f"{cpu_stats['min']:.2f}s",
        f"{cpu_stats['max']:.2f}s",
        "1.00x"
    )
    
    console.print(comp_table)
    
    # Save results
    if save_results:
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        benchmark_file = results_dir / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(benchmark_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        rprint(f"\n[cyan]💾 Benchmark results saved to: {benchmark_file}[/cyan]")
    
    # Performance insights
    rprint("\n[bold cyan]🎯 Performance Insights:[/bold cyan]")
    
    if torch.cuda.is_available() and gpu_times:
        if speedup > 2:
            rprint(f"[green]🚀 Excellent GPU acceleration! {speedup:.1f}x faster than CPU[/green]")
        elif speedup > 1.5:
            rprint(f"[yellow]⚡ Good GPU performance! {speedup:.1f}x faster than CPU[/yellow]")
        else:
            rprint(f"[orange]⚠️  Limited GPU benefit. {speedup:.1f}x faster than CPU[/orange]")
            rprint("[dim]Consider larger batch sizes or more complex workloads for better GPU utilization[/dim]")
    else:
        rprint("[yellow]⚠️  GPU not available or not tested[/yellow]")
    
    rprint(f"[cyan]📈 Processed {len(image_files) * iterations} total images[/cyan]")


def _create_benchmark_test_data(data_path: Path):
    """Create test data for benchmarking if none exists."""
    try:
        # Check if we have the create_test_faces script
        script_path = Path("create_test_faces.py")
        if script_path.exists():
            rprint("[yellow]Creating test face images for benchmarking...[/yellow]")
            
            import subprocess
            result = subprocess.run([
                "python", "create_test_faces.py", 
                "--count", "10", 
                "--output-dir", str(data_path / "Thumbnails")
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                rprint("[green]✅ Test data created successfully[/green]")
            else:
                rprint(f"[red]❌ Failed to create test data: {result.stderr}[/red]")
        else:
            rprint("[yellow]⚠️  No test data creation script found[/yellow]")
            
    except Exception as e:
        rprint(f"[red]❌ Error creating test data: {e}[/red]")


if __name__ == "__main__":
    app()
