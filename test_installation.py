#!/usr/bin/env python3
"""
Test script to validate the face duplicate detection system installation.
"""

import sys
import importlib
from pathlib import Path
import torch
import cv2
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def test_imports():
    """Test all required imports."""
    console.print("[yellow]🔍 Testing imports...[/yellow]")
    
    imports_to_test = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"), 
        ("cv2", "OpenCV"),
        ("face_recognition", "Face Recognition"),
        ("facenet_pytorch", "FaceNet PyTorch"),
        ("typer", "Typer"),
        ("rich", "Rich"),
        ("loguru", "Loguru"),
        ("tqdm", "TQDM"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-learn"),
        ("moviepy", "MoviePy"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy")
    ]
    
    results = []
    all_passed = True
    
    for module_name, display_name in imports_to_test:
        try:
            importlib.import_module(module_name)
            results.append((display_name, "✅ Pass", "green"))
        except ImportError as e:
            results.append((display_name, f"❌ Fail: {e}", "red"))
            all_passed = False
    
    # Create results table
    table = Table(title="Import Test Results", show_header=True, header_style="bold magenta")
    table.add_column("Module", style="cyan", width=20)
    table.add_column("Status", width=30)
    
    for name, status, color in results:
        table.add_row(name, f"[{color}]{status}[/{color}]")
    
    console.print(table)
    return all_passed

def test_cuda():
    """Test CUDA availability."""
    console.print("\n[yellow]🔥 Testing CUDA support...[/yellow]")
    
    cuda_available = torch.cuda.is_available()
    
    table = Table(title="CUDA Test Results", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan", width=25)
    table.add_column("Value", style="green", width=30)
    
    table.add_row("CUDA Available", "✅ Yes" if cuda_available else "❌ No")
    
    if cuda_available:
        table.add_row("Device Count", str(torch.cuda.device_count()))
        table.add_row("Current Device", str(torch.cuda.current_device()))
        table.add_row("Device Name", torch.cuda.get_device_name())
        table.add_row("Memory Allocated", f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        table.add_row("Memory Reserved", f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    else:
        table.add_row("Note", "Will use CPU processing")
    
    console.print(table)
    return True

def test_project_structure():
    """Test project structure."""
    console.print("\n[yellow]📁 Testing project structure...[/yellow]")
    
    required_paths = [
        "src/__init__.py",
        "src/core/__init__.py",
        "src/core/face_detector.py",
        "src/core/duplicate_finder.py", 
        "src/core/video_processor.py",
        "src/utils/__init__.py",
        "src/utils/config.py",
        "src/utils/logging.py",
        "src/utils/image_utils.py",
        "main.py",
        "requirements.txt",
        "README.md"
    ]
    
    results = []
    all_found = True
    
    for path in required_paths:
        if Path(path).exists():
            results.append((path, "✅ Found", "green"))
        else:
            results.append((path, "❌ Missing", "red"))
            all_found = False
    
    table = Table(title="Project Structure Test", show_header=True, header_style="bold magenta")
    table.add_column("File/Directory", style="cyan", width=30)
    table.add_column("Status", width=15)
    
    for path, status, color in results:
        table.add_row(path, f"[{color}]{status}[/{color}]")
    
    console.print(table)
    return all_found

def test_basic_functionality():
    """Test basic functionality."""
    console.print("\n[yellow]⚙️ Testing basic functionality...[/yellow]")
    
    try:
        # Test config import
        from src.utils.config import Config
        config = Config()
        
        # Test face detector import
        from src.core.face_detector import FaceDetector
        
        # Test duplicate finder import
        from src.core.duplicate_finder import DuplicateFinder
        
        # Test video processor import
        from src.core.video_processor import VideoProcessor
        
        console.print("[green]✅ All core modules imported successfully[/green]")
        
        # Test creating a simple array and processing
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        console.print("[green]✅ NumPy array creation successful[/green]")
        
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Basic functionality test failed: {e}[/red]")
        return False

def test_sample_data():
    """Test sample data availability."""
    console.print("\n[yellow]📊 Testing sample data...[/yellow]")
    
    data_dir = Path("data")
    thumbnails_dir = data_dir / "Thumbnails"
    videos_dir = data_dir / "Videos"
    
    table = Table(title="Sample Data Test", show_header=True, header_style="bold magenta")
    table.add_column("Directory", style="cyan", width=20)
    table.add_column("Status", width=15)
    table.add_column("Count", width=10)
    
    # Check thumbnails
    if thumbnails_dir.exists():
        thumb_files = list(thumbnails_dir.glob("*.*"))
        table.add_row("Thumbnails", "[green]✅ Found[/green]", str(len(thumb_files)))
    else:
        table.add_row("Thumbnails", "[red]❌ Missing[/red]", "0")
    
    # Check videos
    if videos_dir.exists():
        video_files = list(videos_dir.glob("*.*"))
        table.add_row("Videos", "[green]✅ Found[/green]", str(len(video_files)))
    else:
        table.add_row("Videos", "[red]❌ Missing[/red]", "0")
    
    console.print(table)
    return True

def main():
    """Run all tests."""
    console.print(Panel.fit(
        "[bold blue]🚀 Face Duplicate Detection System - Installation Test[/bold blue]",
        border_style="blue"
    ))
    
    tests = [
        ("Import Test", test_imports),
        ("CUDA Test", test_cuda),
        ("Project Structure Test", test_project_structure),
        ("Basic Functionality Test", test_basic_functionality),
        ("Sample Data Test", test_sample_data)
    ]
    
    all_passed = True
    results_summary = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results_summary.append((test_name, "✅ Pass" if result else "❌ Fail"))
            if not result:
                all_passed = False
        except Exception as e:
            console.print(f"[red]❌ {test_name} failed with error: {e}[/red]")
            results_summary.append((test_name, f"❌ Error: {str(e)[:50]}..."))
            all_passed = False
    
    # Final summary
    console.print("\n" + "="*60)
    summary_table = Table(title="🎯 Test Summary", show_header=True, header_style="bold magenta")
    summary_table.add_column("Test", style="cyan", width=25)
    summary_table.add_column("Result", width=30)
    
    for test_name, result in results_summary:
        color = "green" if "✅" in result else "red"
        summary_table.add_row(test_name, f"[{color}]{result}[/{color}]")
    
    console.print(summary_table)
    
    if all_passed:
        console.print(Panel.fit(
            "[bold green]🎉 All tests passed! The system is ready to use.[/bold green]\n\n"
            "[cyan]Next steps:[/cyan]\n"
            "1. python main.py info\n"
            "2. python main.py process-images --input-dir data/Thumbnails\n"
            "3. python main.py process-videos --input-dir data/Videos",
            border_style="green",
            title="Success"
        ))
        return 0
    else:
        console.print(Panel.fit(
            "[bold red]❌ Some tests failed. Please check the installation.[/bold red]\n\n"
            "[cyan]Try:[/cyan]\n"
            "1. Run ./setup.sh again\n"
            "2. Check requirements.txt\n"
            "3. Verify CUDA installation (if needed)",
            border_style="red",
            title="Issues Found"
        ))
        return 1

if __name__ == "__main__":
    sys.exit(main())
