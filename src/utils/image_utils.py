"""
Image utilities for face processing.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
from loguru import logger


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image file safely.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Image as numpy array or None if failed
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            # Try with PIL as fallback
            pil_image = Image.open(image_path)
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


def resize_image(image: np.ndarray, target_size: Tuple[int, int], maintain_aspect: bool = True) -> np.ndarray:
    """
    Resize an image to target size.
    
    Args:
        image: Input image
        target_size: (width, height) tuple
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    if maintain_aspect:
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate aspect ratios
        aspect_ratio = w / h
        target_aspect = target_w / target_h
        
        if aspect_ratio > target_aspect:
            # Image is wider than target
            new_w = target_w
            new_h = int(target_w / aspect_ratio)
        else:
            # Image is taller than target
            new_h = target_h
            new_w = int(target_h * aspect_ratio)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # Pad to target size
        top = (target_h - new_h) // 2
        bottom = target_h - new_h - top
        left = (target_w - new_w) // 2
        right = target_w - new_w - left
        
        return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        return cv2.resize(image, target_size)


def save_face_crops(faces_data: List[dict], output_dir: str) -> List[str]:
    """
    Save face crops to disk.
    
    Args:
        faces_data: List of face dictionaries with crops
        output_dir: Output directory
        
    Returns:
        List of saved file paths
    """
    output_path = Path(output_dir)
    crops_dir = output_path / "face_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    for face in faces_data:
        try:
            face_id = face["face_id"]
            crop = face["crop"]
            
            # Convert RGB to BGR if needed
            if crop.shape[2] == 3:
                crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            else:
                crop_bgr = crop
            
            crop_path = crops_dir / f"{face_id}.jpg"
            cv2.imwrite(str(crop_path), crop_bgr)
            saved_files.append(str(crop_path))
            
        except Exception as e:
            logger.error(f"Error saving face crop {face.get('face_id', 'unknown')}: {e}")
    
    return saved_files


def create_duplicate_visualization(duplicate_groups: List[List[dict]], output_dir: str) -> str:
    """
    Create a visualization of duplicate face groups.
    
    Args:
        duplicate_groups: List of duplicate groups
        output_dir: Output directory
        
    Returns:
        Path to saved visualization
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Calculate grid size
        max_group_size = max(len(group) for group in duplicate_groups) if duplicate_groups else 0
        num_groups = min(len(duplicate_groups), 10)  # Limit to first 10 groups
        
        if num_groups == 0 or max_group_size == 0:
            return ""
        
        fig, axes = plt.subplots(num_groups, max_group_size, figsize=(max_group_size * 2, num_groups * 2))
        
        if num_groups == 1:
            axes = axes.reshape(1, -1)
        
        for group_idx, group in enumerate(duplicate_groups[:num_groups]):
            for face_idx, face in enumerate(group):
                if face_idx < max_group_size:
                    ax = axes[group_idx, face_idx] if num_groups > 1 else axes[face_idx]
                    
                    # Display face crop
                    crop = face.get("crop")
                    if crop is not None:
                        if len(crop.shape) == 3 and crop.shape[2] == 3:
                            # Convert BGR to RGB for matplotlib
                            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        else:
                            crop_rgb = crop
                        
                        ax.imshow(crop_rgb)
                        ax.set_title(f"Conf: {face.get('confidence', 0):.2f}", fontsize=8)
                    
                    ax.axis('off')
            
            # Hide empty subplots
            for face_idx in range(len(group), max_group_size):
                if num_groups > 1:
                    axes[group_idx, face_idx].axis('off')
                else:
                    axes[face_idx].axis('off')
        
        plt.suptitle("Duplicate Face Groups", fontsize=16)
        plt.tight_layout()
        
        viz_path = output_path / "duplicate_visualization.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(viz_path)
        
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        return ""


def validate_image_file(file_path: str) -> bool:
    """
    Validate if a file is a valid image.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if valid image, False otherwise
    """
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def get_image_info(image_path: str) -> dict:
    """
    Get information about an image file.
    
    Args:
        image_path: Path to the image
        
    Returns:
        Dictionary with image information
    """
    try:
        with Image.open(image_path) as img:
            return {
                "path": image_path,
                "format": img.format,
                "mode": img.mode,
                "size": img.size,
                "width": img.width,
                "height": img.height,
                "file_size_mb": Path(image_path).stat().st_size / (1024 * 1024)
            }
    except Exception as e:
        return {"path": image_path, "error": str(e)}
