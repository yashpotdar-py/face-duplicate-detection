import cv2
import numpy as np
import face_recognition
from sklearn.metrics.pairwise import cosine_similarity
import os
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# Initialize rich console
console = Console()

# === PARAMETERS ===
folder_path = "data/Thumbnails"  # <- Your images folder path
save_faces_folder = "detected_faces"  # <- Folder to save detected faces
similarity_threshold = 0.35
blur_threshold = 100

unique_faces = []            # confirmed embeddings
unique_face_images = []      # set of image filenames for each confirmed face

total_detected = 0
saved_face_counter = 0

# === Create save folder if not exists ===
if not os.path.exists(save_faces_folder):
    os.makedirs(save_faces_folder)


def is_blurry(image, threshold=blur_threshold):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


# Get list of image files
image_files = [f for f in os.listdir(folder_path)
               if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))]

console.print(Panel.fit(
    f"[bold cyan]Face Duplicate Detection[/bold cyan]\n"
    f"Processing {len(image_files)} images from '{folder_path}'",
    border_style="cyan"
))

# === Process each image in folder with custom progress display ===
with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    TimeRemainingColumn(),
    console=console,
    refresh_per_second=4
) as progress:

    task = progress.add_task("Processing images...", total=len(image_files))

    for idx, filename in enumerate(image_files, 1):
        # Update progress description with current image info
        progress.update(
            task, description=f"Processing images... [{idx}/{len(image_files)}] {filename[:30]}")

        image_path = os.path.join(folder_path, filename)
        frame = cv2.imread(image_path)

        if frame is None:
            console.print(f"[red]❌ Error reading image: {filename}[/red]")
            progress.advance(task)
            continue

        if is_blurry(frame):
            console.print(
                f"[dim]⚠️  {filename} is blurry - [red]Skipped[/red][/dim]")
            progress.advance(task)
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Use face_recognition for detection (more accurate than Haar cascades)
        face_locations = face_recognition.face_locations(
            rgb, model="hog")  # Use "cnn" for better accuracy but slower
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        total_detected += len(face_locations)

        if len(face_locations) > 0:
            console.print(
                f"[green]👤 {filename}: Detected {len(face_locations)} faces[/green]")

        for face_idx, ((top, right, bottom, left), encoding) in enumerate(zip(face_locations, face_encodings)):
            face_img = rgb[top:bottom, left:right]

            # === Save every detected face ===
            saved_face_counter += 1
            image_base = os.path.splitext(filename)[0]
            save_path = os.path.join(
                save_faces_folder, f"{image_base}_face{saved_face_counter}.jpg")
            cv2.imwrite(save_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))

            try:
                matched = False
                for idx_face, u in enumerate(unique_faces):
                    similarity = cosine_similarity([encoding], [u])[0][0]
                    if similarity > (1 - similarity_threshold):
                        unique_face_images[idx_face].add(filename)
                        matched = True
                        console.print(
                            f"[green]✅ Existing Face #{idx_face+1}[/green] - Similarity: [cyan]{similarity:.3f}[/cyan]")
                        break

                if not matched:
                    unique_faces.append(encoding)
                    unique_face_images.append(set([filename]))
                    console.print(
                        f"[bold magenta]🆕 New Unique Face #{len(unique_faces)}[/bold magenta]")

            except Exception as e:
                console.print(f"[red]❌ Error processing face: {e}[/red]")
                continue

        progress.advance(task)

# === Final Output ===
console.print("\n")

# Create summary table
table = Table(title="Face Detection Summary", style="cyan")
table.add_column("Metric", style="bold")
table.add_column("Value", style="green")

table.add_row("Total Faces Detected", str(total_detected))
table.add_row("Confirmed Unique Faces", str(len(unique_faces)))
table.add_row("Images Processed", str(len(image_files)))

console.print(table)

# Create unique faces table
if unique_face_images:
    faces_table = Table(title="Unique Faces Distribution", style="magenta")
    faces_table.add_column("Face ID", style="bold cyan")
    faces_table.add_column("Found in Images", style="yellow")
    faces_table.add_column("Count", style="green")

    for i, imgs in enumerate(unique_face_images, 1):
        faces_table.add_row(
            f"Face {i}",
            ", ".join(list(imgs)[:3]) + ("..." if len(imgs) > 3 else ""),
            str(len(imgs))
        )

    console.print("\n")
    console.print(faces_table)

console.print(
    f"\n[dim]💾 All detected faces saved to: {save_faces_folder}/[/dim]")
