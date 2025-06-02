#!/usr/bin/env python3
"""
Create test data with actual human faces for demonstrating the face duplicate detection system.
"""

import cv2
import numpy as np
from pathlib import Path

def create_synthetic_face_images():
    """Create synthetic test images with simple face-like patterns for testing."""
    
    test_dir = Path("test_data/faces")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create simple synthetic face images for testing
    faces = []
    
    # Create 3 sets of "duplicate" faces and 2 unique faces
    for person_id in range(5):
        for duplicate_id in range(3 if person_id < 3 else 1):  # 3 duplicates for first 3 people, 1 for others
            # Create a simple face-like image
            img = np.ones((200, 200, 3), dtype=np.uint8) * 220  # Light gray background
            
            # Draw a simple face
            center_x, center_y = 100, 100
            
            # Face outline (circle)
            cv2.circle(img, (center_x, center_y), 80, (200, 180, 160), -1)  # Skin color
            
            # Eyes
            eye_y = center_y - 20
            cv2.circle(img, (center_x - 25, eye_y), 8, (0, 0, 0), -1)  # Left eye
            cv2.circle(img, (center_x + 25, eye_y), 8, (0, 0, 0), -1)  # Right eye
            
            # Nose
            nose_points = np.array([
                [center_x - 5, center_y - 5],
                [center_x + 5, center_y - 5], 
                [center_x, center_y + 10]
            ], np.int32)
            cv2.fillPoly(img, [nose_points], (150, 120, 100))
            
            # Mouth
            cv2.ellipse(img, (center_x, center_y + 30), (20, 10), 0, 0, 180, (100, 50, 50), 2)
            
            # Add slight variations for each person
            # Person 1: Normal
            # Person 2: Slightly different eye position
            if person_id == 1:
                cv2.circle(img, (center_x - 20, eye_y - 3), 8, (0, 0, 0), -1)  # Left eye
                cv2.circle(img, (center_x + 30, eye_y - 3), 8, (0, 0, 0), -1)  # Right eye
            
            # Person 3: Different nose
            elif person_id == 2:
                nose_points = np.array([
                    [center_x - 8, center_y - 8],
                    [center_x + 8, center_y - 8], 
                    [center_x, center_y + 15]
                ], np.int32)
                cv2.fillPoly(img, [nose_points], (120, 100, 80))
            
            # Person 4: Different mouth
            elif person_id == 3:
                cv2.ellipse(img, (center_x, center_y + 35), (25, 15), 0, 0, 180, (80, 30, 30), 3)
            
            # Person 5: Different face shape
            elif person_id == 4:
                cv2.ellipse(img, (center_x, center_y), (60, 90), 0, 0, 360, (180, 160, 140), -1)
            
            # Add some noise for realism but keep duplicates similar
            if duplicate_id > 0:
                # Add slight variations for duplicates
                noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Save image
            filename = f"person_{person_id}_duplicate_{duplicate_id}.jpg"
            filepath = test_dir / filename
            cv2.imwrite(str(filepath), img)
            
            print(f"Created: {filepath}")
    
    return test_dir

def test_system_with_synthetic_faces():
    """Test the face duplicate detection system with synthetic faces."""
    
    print("Creating synthetic face test data...")
    test_dir = create_synthetic_face_images()
    
    print(f"\nTest images created in: {test_dir}")
    print("Files created:")
    for img_file in sorted(test_dir.glob("*.jpg")):
        print(f"  - {img_file.name}")
    
    return test_dir

if __name__ == "__main__":
    test_dir = test_system_with_synthetic_faces()
    print(f"\nTo test the system, run:")
    print(f"python main.py process-images --input-dir {test_dir} --output-dir results/synthetic_test")
