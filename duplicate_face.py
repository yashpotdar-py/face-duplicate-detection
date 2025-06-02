import os
import face_recognition
import time
import csv
import psutil  # To track CPU & RAM usage
import GPUtil   # To track GPU usage
from rich import print
from rich.progress import Progress, SpinnerColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn, TextColumn
from rich.console import Console
from rich.panel import Panel
from datetime import datetime

console = Console()
BATCH_SIZE = 100


def get_system_usage():
    """Fetches CPU, RAM, and GPU usage."""
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent
    gpus = GPUtil.getGPUs()
    # Get first GPU's usage if available
    gpu_usage = gpus[0].load * 100 if gpus else 0

    return cpu_usage, ram_usage, gpu_usage


def process_batch(image_batch, known_face_encodings, known_face_paths, duplicates, unique_images, corrupted_images, progress, process_task, start_process_time):
    total_face_count = 0
    duplicate_count = 0
    processed_count = 0
    unique_count = 0  # Reset unique count per batch for progress display only

    for image_path in image_batch:
        processed_count += 1

        cpu_usage, ram_usage, gpu_usage = get_system_usage()  # Get system usage
        progress.update(process_task, advance=1,
                        description=f"[bold cyan]Processing 📸 {os.path.basename(image_path)}[/]",
                        processed=processed_count, cpu=cpu_usage, ram=ram_usage, gpu=gpu_usage)

        try:
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)

            total_face_count += len(face_encodings)
            progress.update(process_task, faces=total_face_count)

            for current_encoding in face_encodings:
                if not known_face_encodings:
                    known_face_encodings.append(current_encoding)
                    known_face_paths.append(image_path)
                    unique_images.add(image_path)
                    unique_count += 1
                else:
                    found_duplicate = False
                    for i, known_encoding in enumerate(known_face_encodings):
                        match = face_recognition.compare_faces(
                            [known_encoding], current_encoding, tolerance=0.32)[0]
                        if match:
                            duplicates.setdefault(
                                known_face_paths[i], []).append(image_path)
                            found_duplicate = True
                            duplicate_count += 1
                            progress.update(
                                process_task, duplicates=duplicate_count)
                            break

                    if not found_duplicate:
                        known_face_encodings.append(current_encoding)
                        known_face_paths.append(image_path)
                        unique_images.add(image_path)
                        unique_count += 1
                        progress.update(process_task, unique=unique_count)

        except Exception as e:
            console.print(f"[red]Error processing {image_path}: {str(e)}[/]")
            corrupted_images.append(image_path)

    return total_face_count, duplicate_count, processed_count


def save_results(batch_number, duplicates, corrupted_images):
    """Saves duplicate and corrupted face data to text and CSV files."""
    batch_dir = "batch_results"
    os.makedirs(batch_dir, exist_ok=True)

    text_file = os.path.join(batch_dir, f"output_batch_{batch_number}.txt")
    csv_file = os.path.join(batch_dir, f"duplicates_batch_{batch_number}.csv")
    corrupted_file = os.path.join(
        batch_dir, f"corrupted_batch_{batch_number}.txt")

    with open(text_file, "w") as txt:
        txt.write(f"Duplicate faces found in batch {batch_number}:\n\n")
        for original, dups in duplicates.items():
            txt.write(f"Image {original} has {len(dups)} duplicates:\n")
            for dup in dups:
                txt.write(f"- {dup}\n")
            txt.write("\n")

    with open(csv_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Original Image", "Duplicate Image"])
        for original, dups in duplicates.items():
            for dup in dups:
                writer.writerow([original, dup])

    with open(corrupted_file, "w") as corrupted_txt:
        corrupted_txt.write(
            f"Corrupted or unreadable images in batch {batch_number}:\n")
        for img in corrupted_images:
            corrupted_txt.write(f"{img}\n")

    console.print(
        f"[green]Results saved: {text_file}, {csv_file}, {corrupted_file}[/]")


def find_duplicate_faces_in_directory(directory):
    known_face_encodings = []
    known_face_paths = []
    duplicates = {}
    unique_images = set()
    corrupted_images = []  # Track unreadable images
    total_face_count = 0
    duplicate_count = 0
    processed_count = 0
    start_process_time = time.time()
    batch_number = 1

    image_files = [os.path.join(root, file) for root, _, files in os.walk(
        directory) for file in files if file.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_batches = [image_files[i:i + BATCH_SIZE]
                     for i in range(0, len(image_files), BATCH_SIZE)]

    for batch in image_batches:
        with Progress(
            SpinnerColumn(style="green"),
            TextColumn("[bold progress.description]{task.description}"),
            BarColumn(complete_style="green", finished_style="bright_green"),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("[cyan]Processed: {task.fields[processed]}"),
            TextColumn("[yellow]Faces: {task.fields[faces]}"),
            TextColumn("[green]Unique (Batch): {task.fields[unique]}"),
            TextColumn("[red]Duplicates: {task.fields[duplicates]}"),
            TextColumn("[magenta]Corrupted: {task.fields[corrupted]}"),
            TextColumn("[blue]CPU: {task.fields[cpu]}%"),
            TextColumn("[purple]RAM: {task.fields[ram]}%"),
            TextColumn("[bold orange]GPU: {task.fields[gpu]}%"),
            console=console,
            expand=True
        ) as progress:
            process_task = progress.add_task(f"[cyan]Processing batch {batch_number}...", total=len(
                batch), processed=0, faces=0, unique=0, duplicates=0, corrupted=0, cpu=0, ram=0, gpu=0)

            batch_faces, batch_duplicates, batch_processed = process_batch(
                batch, known_face_encodings, known_face_paths, duplicates,
                unique_images, corrupted_images, progress, process_task, start_process_time
            )

            total_face_count += batch_faces
            duplicate_count += batch_duplicates
            processed_count += batch_processed

            save_results(batch_number, duplicates, corrupted_images)

            batch_number += 1

    total_time = time.time() - start_process_time
    cpu_usage, ram_usage, gpu_usage = get_system_usage()

    console.print(Panel.fit(
        f"✨ [bold green]Processing Complete![/]\n"
        f"📸 Total Images Processed: [cyan]{processed_count}[/]\n"
        f"👤 Total Faces Detected: [yellow]{total_face_count}[/]\n"
        f"🆕 Unique Faces: [green]{len(unique_images)}[/]\n"
        f"🔁 Duplicate Faces: [red]{duplicate_count}[/]\n"
        f"⚠️ Corrupted/Missing Images: [magenta]{len(corrupted_images)}[/]\n"
        f"⏳ Total Time: [blue]{total_time:.2f} seconds[/]\n"
        f"🖥 CPU Usage: [blue]{cpu_usage}%[/]\n"
        f"💾 RAM Usage: [purple]{ram_usage}%[/]\n"
        f"🎮 GPU Usage: [orange]{gpu_usage}%[/]",
        border_style="green",
        title="Final Summary"
    ))


directory_path = 'detected_faces'

console.print(Panel.fit(
    "🚀 [bold blue]Starting Batch Duplicate Face Detection...[/]",
    border_style="blue",
    title="Face Recognition System",
    subtitle="Mera Baccha Hai Tu"
))

find_duplicate_faces_in_directory(directory_path)
