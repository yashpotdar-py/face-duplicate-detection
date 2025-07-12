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
BATCH_SIZE = 6161


def get_system_usage():
    """Fetches CPU, RAM, and GPU usage."""
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent
    gpus = GPUtil.getGPUs()
    # Get first GPU's usage if available
    gpu_usage = gpus[0].load * 100 if gpus else 0

    return cpu_usage, ram_usage, gpu_usage


def process_batch(image_batch, known_face_encodings, known_face_paths, duplicates, unique_images, corrupted_images, no_face_images, progress, process_task, start_process_time):
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

            # If no faces detected, treat as unique image
            if len(face_encodings) == 0:
                unique_images.add(image_path)
                no_face_images.add(image_path)
                unique_count += 1
                progress.update(process_task, unique=unique_count)
                continue

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
            unique_images.add(image_path)  # Add corrupted images to unique set
            unique_count += 1
            progress.update(process_task, unique=unique_count)

    return total_face_count, duplicate_count, processed_count


def save_results(batch_number, duplicates, corrupted_images, unique_images):
    """Saves duplicate, unique, and corrupted face data to text and CSV files."""
    batch_dir = "batch_results"
    os.makedirs(batch_dir, exist_ok=True)

    text_file = os.path.join(batch_dir, f"output_batch_{batch_number}.txt")
    csv_file = os.path.join(batch_dir, f"duplicates_batch_{batch_number}.csv")
    unique_csv_file = os.path.join(
        batch_dir, f"unique_faces_batch_{batch_number}.csv")
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

    # Save unique faces to CSV (including no faces and corrupted images)
    with open(unique_csv_file, "w", newline="") as unique_csvfile:
        writer = csv.writer(unique_csvfile)
        writer.writerow(["Image Path", "Filename", "Status"])

        # Add unique images with faces
        for unique_path in unique_images:
            filename = os.path.basename(unique_path)
            writer.writerow([unique_path, filename, "Unique Face"])

        # Add corrupted images
        for corrupted_path in corrupted_images:
            filename = os.path.basename(corrupted_path)
            writer.writerow([corrupted_path, filename, "Corrupted/Unreadable"])

    with open(corrupted_file, "w") as corrupted_txt:
        corrupted_txt.write(
            f"Corrupted or unreadable images in batch {batch_number}:\n")
        for img in corrupted_images:
            corrupted_txt.write(f"{img}\n")

    console.print(
        f"[green]Results saved: {text_file}, {csv_file}, {unique_csv_file}, {corrupted_file}[/]")


def find_duplicate_faces_in_directory(directory):
    known_face_encodings = []
    known_face_paths = []
    duplicates = {}
    unique_images = set()
    no_face_images = set()
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
                unique_images, corrupted_images, no_face_images, progress, process_task, start_process_time
            )

            total_face_count += batch_faces
            duplicate_count += batch_duplicates
            processed_count += batch_processed

            save_results(batch_number, duplicates,
                         corrupted_images, unique_images)

            batch_number += 1

    # Save final consolidated unique faces CSV
    final_unique_csv = os.path.join("batch_results", "all_unique_faces.csv")
    with open(final_unique_csv, "w", newline="") as final_unique_csvfile:
        writer = csv.writer(final_unique_csvfile)
        writer.writerow(["Image Path", "Filename", "Directory", "Status"])

        # Add unique images with faces
        for unique_path in unique_images:
            if unique_path not in corrupted_images and unique_path not in no_face_images:
                filename = os.path.basename(unique_path)
                directory_name = os.path.basename(os.path.dirname(unique_path))
                writer.writerow(
                    [unique_path, filename, directory_name, "Unique Face"])

        # Add images with no faces
        for no_face_path in no_face_images:
            filename = os.path.basename(no_face_path)
            directory_name = os.path.basename(os.path.dirname(no_face_path))
            writer.writerow(
                [no_face_path, filename, directory_name, "No Face Detected"])

        # Add corrupted images
        for corrupted_path in corrupted_images:
            filename = os.path.basename(corrupted_path)
            directory_name = os.path.basename(os.path.dirname(corrupted_path))
            writer.writerow([corrupted_path, filename,
                            directory_name, "Corrupted/Unreadable"])

    console.print(
        f"[green]Final unique faces CSV saved: {final_unique_csv}[/]")

    total_time = time.time() - start_process_time
    cpu_usage, ram_usage, gpu_usage = get_system_usage()

    console.print(Panel.fit(
        f"✨ [bold green]Processing Complete![/]\n"
        f"📸 Total Images Processed: [cyan]{processed_count}[/]\n"
        f"👤 Total Faces Detected: [yellow]{total_face_count}[/]\n"
        f"🆕 Unique Faces: [green]{len(unique_images) - len(no_face_images) - len(corrupted_images)}[/]\n"
        f"👻 No Face Detected: [orange]{len(no_face_images)}[/]\n"
        f"💥 Corrupted Images: [magenta]{len(corrupted_images)}[/]\n"
        f"🔁 Duplicate Faces: [red]{duplicate_count}[/]\n"
        f"⏳ Total Time: [blue]{total_time:.2f} seconds[/]\n"
        f"🖥 CPU Usage: [blue]{cpu_usage}%[/]\n"
        f"💾 RAM Usage: [purple]{ram_usage}%[/]\n"
        f"🎮 GPU Usage: [orange]{gpu_usage}%[/]",
        border_style="green",
        title="Final Summary"
    ))


directory_path = 'detected_images/detected_images'

console.print(Panel.fit(
    "🚀 [bold blue]Starting Batch Duplicate Face Detection...[/]",
    border_style="blue",
    title="Face Recognition System",
    subtitle="Summary"
))

find_duplicate_faces_in_directory(directory_path)
