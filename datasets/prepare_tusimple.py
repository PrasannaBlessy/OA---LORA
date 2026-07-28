#!/usr/bin/env python3
"""
prepare_tusimple.py

Reproducibly select exactly 1,000 images from the complete TuSimple
train_set/clips hierarchy.

Expected input structure:
    TuSimple/
    └── train_set/
        └── clips/
            ├── 0313-1/
            │   ├── 60/
            │   │   ├── *.jpg
            │   │   └── ...
            │   ├── 120/
            │   └── ...
            └── ...

The script:
1. Recursively finds all JPG/JPEG/PNG images under train_set/clips.
2. Sorts paths deterministically.
3. Selects exactly N images using a fixed random seed.
4. Copies selected images into an output directory.
5. Writes a CSV manifest containing the original relative path,
   selected filename, source sequence, and source frame folder.
6. Writes a JSON summary for reproducibility.

Example:
    python prepare_tusimple.py ^
        --tusimple_root "D:/datasets/TuSimple" ^
        --output_dir "D:/OA-LoRA/datasets/TuSimple/selected_images" ^
        --num_images 1000 ^
        --seed 469

The same input dataset + same arguments + same seed produce the same
selection, provided the source file paths are unchanged.
"""

import argparse
import csv
import hashlib
import json
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproducibly select exactly 1,000 TuSimple training images."
    )

    parser.add_argument(
        "--tusimple_root",
        type=Path,
        required=True,
        help="Root directory of the TuSimple dataset."
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where selected images and metadata will be saved."
    )

    parser.add_argument(
        "--num_images",
        type=int,
        default=1000,
        help="Number of images to select. Default: 1000."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=469,
        help="Random seed for reproducible selection. Default: 469."
    )

    parser.add_argument(
        "--copy_mode",
        choices=["copy", "hardlink"],
        default="copy",
        help="How to place selected images in output_dir. Default: copy."
    )

    return parser.parse_args()


def find_clips_dir(tusimple_root: Path) -> Path:
    """
    Locate train_set/clips. Supports the expected TuSimple structure
    and a direct path supplied as the dataset root.
    """
    candidates = [
        tusimple_root / "train_set" / "clips",
        tusimple_root / "clips",
    ]

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not find 'train_set/clips' or 'clips' under: "
        f"{tusimple_root}"
    )


def collect_images(clips_dir: Path):
    """
    Recursively collect all image files and return them in deterministic
    lexicographic order.
    """
    images = [
        p for p in clips_dir.rglob("*")
        if p.is_file() and p.suffix in IMAGE_EXTENSIONS
    ]

    images.sort(key=lambda p: p.relative_to(clips_dir).as_posix().lower())

    return images


def sha256_file(path: Path, chunk_size=1024 * 1024):
    """
    Calculate SHA-256 hash of a file so the manifest can uniquely identify
    the selected source image.
    """
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def get_source_info(image_path: Path, clips_dir: Path):
    """
    Extract useful path information from the TuSimple hierarchy.

    Example relative path:
        0313-1/60/00000.jpg

    sequence:
        0313-1

    frame_folder:
        60
    """
    relative_path = image_path.relative_to(clips_dir)
    parts = relative_path.parts

    sequence = parts[0] if len(parts) >= 1 else ""
    frame_folder = parts[1] if len(parts) >= 2 else ""

    return relative_path.as_posix(), sequence, frame_folder


def copy_or_link(source: Path, destination: Path, mode: str):
    destination.parent.mkdir(parents=True, exist_ok=True)

    if mode == "hardlink":
        try:
            destination.hardlink_to(source)
            return
        except OSError:
            print(
                f"Warning: hardlink failed for {source.name}; "
                "falling back to normal copy."
            )

    shutil.copy2(source, destination)


def main():
    args = parse_args()

    if args.num_images <= 0:
        raise ValueError("--num_images must be greater than 0.")

    tusimple_root = args.tusimple_root.resolve()
    output_dir = args.output_dir.resolve()

    clips_dir = find_clips_dir(tusimple_root)

    print("=" * 70)
    print("TuSimple Reproducible Dataset Preparation")
    print("=" * 70)
    print(f"Dataset root : {tusimple_root}")
    print(f"Clips folder : {clips_dir}")
    print(f"Output       : {output_dir}")
    print(f"Requested    : {args.num_images}")
    print(f"Random seed  : {args.seed}")
    print("=" * 70)

    all_images = collect_images(clips_dir)

    print(f"\nTotal images found: {len(all_images)}")

    if len(all_images) < args.num_images:
        raise RuntimeError(
            f"Only {len(all_images)} images were found, but "
            f"{args.num_images} images were requested."
        )

    # Use a fixed seed and a deterministic, pre-sorted list.
    # This makes the selected subset reproducible.
    rng = random.Random(args.seed)
    selected_images = rng.sample(all_images, args.num_images)

    # Sort selected paths before assigning output filenames so the manifest
    # and generated filenames are deterministic.
    selected_images.sort(
        key=lambda p: p.relative_to(clips_dir).as_posix().lower()
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "tusimple_selection.csv"
    summary_path = output_dir / "selection_summary.json"

    manifest_rows = []

    print("\nCopying selected images...")

    for index, source_path in enumerate(selected_images, start=1):
        relative_path, sequence, frame_folder = get_source_info(
            source_path, clips_dir
        )

        # Sequential filenames make the final dataset simple to use.
        output_name = f"{index:06d}{source_path.suffix.lower()}"
        destination_path = output_dir / output_name

        copy_or_link(
            source_path,
            destination_path,
            args.copy_mode
        )

        file_hash = sha256_file(source_path)

        manifest_rows.append({
            "selected_index": index,
            "output_filename": output_name,
            "source_relative_path": relative_path,
            "source_sequence": sequence,
            "source_frame_folder": frame_folder,
            "sha256": file_hash,
        })

        if index % 100 == 0 or index == args.num_images:
            print(f"  Processed {index}/{args.num_images}")

    # Save CSV manifest.
    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "selected_index",
                "output_filename",
                "source_relative_path",
                "source_sequence",
                "source_frame_folder",
                "sha256",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Save reproducibility metadata.
    summary = {
        "dataset": "TuSimple",
        "source_root": str(tusimple_root),
        "source_clips_directory": str(clips_dir),
        "selection_method": "Uniform random sampling without replacement "
                            "from all discovered training images after "
                            "deterministic path sorting.",
        "num_images_available": len(all_images),
        "num_images_selected": len(selected_images),
        "random_seed": args.seed,
        "copy_mode": args.copy_mode,
        "image_extensions": sorted(IMAGE_EXTENSIONS),
        "manifest": manifest_path.name,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Selected images : {len(selected_images)}")
    print(f"Image directory : {output_dir}")
    print(f"CSV manifest    : {manifest_path}")
    print(f"JSON summary    : {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
