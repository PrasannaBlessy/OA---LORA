#!/usr/bin/env python3
"""
prepare_culane.py

Reproducibly select exactly 2,000 CULane training images.

Expected CULane structures supported:

1) Standard CULane layout:
    CULane/
    ├── driver_23_30frame/
    ├── driver_161_90frame/
    ├── driver_182_30frame/
    ├── ...
    ├── list/
    │   ├── train_gt.txt
    │   ├── val.txt
    │   └── test.txt
    └── laneseg_label_w16/

2) CULane extracted under another root directory.

The script:
1. Searches for CULane image files.
2. Uses CULane list/train_gt.txt when available to identify training images.
3. Falls back to excluding obvious test/validation directories if the
   training list is unavailable.
4. Sorts candidate paths deterministically.
5. Selects exactly N images using a fixed random seed.
6. Copies selected images to a clean output directory.
7. Creates a CSV manifest with source paths and SHA-256 hashes.
8. Creates a JSON summary documenting the selection procedure.

Default:
    2,000 images
    seed = 469

Example:
    python prepare_culane.py ^
        --culane_root "D:/datasets/CULane" ^
        --output_dir "D:/OA-LoRA/datasets/CULane/selected_images" ^
        --num_images 2000 ^
        --seed 469
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
        description="Reproducibly select exactly 2,000 CULane training images."
    )

    parser.add_argument(
        "--culane_root",
        type=Path,
        required=True,
        help="Root directory of the extracted CULane dataset."
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
        default=2000,
        help="Number of images to select. Default: 2000."
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
        help="How to place selected images. Default: copy."
    )

    return parser.parse_args()


def find_training_list(culane_root: Path):
    """
    Locate a CULane training list.

    Common names include:
        list/train_gt.txt
        list/train.txt
        train_gt.txt
        train.txt
    """
    candidates = [
        culane_root / "list" / "train_gt.txt",
        culane_root / "list" / "train.txt",
        culane_root / "train_gt.txt",
        culane_root / "train.txt",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    # Search recursively if the list is nested differently.
    for name in ["train_gt.txt", "train.txt"]:
        matches = sorted(
            culane_root.rglob(name),
            key=lambda p: p.as_posix().lower()
        )
        if matches:
            return matches[0]

    return None


def parse_training_list(list_path: Path):
    """
    Read CULane training-list entries.

    CULane list files commonly contain paths such as:
        /driver_23_30frame/05151645_0418.MP4/00000.jpg

    A line may contain additional fields in some dataset variants.
    The first whitespace-separated field is treated as the image path.
    """
    entries = []

    with list_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            image_entry = line.split()[0]
            image_entry = image_entry.lstrip("/\\")
            entries.append(image_entry.replace("\\", "/"))

    # Remove duplicates while preserving deterministic ordering.
    return sorted(set(entries), key=str.lower)


def resolve_training_images(culane_root: Path, list_path: Path):
    """
    Resolve paths from train_gt.txt/train.txt against the CULane root.

    Also tries common dataset-root variants if the exact path does not exist.
    """
    entries = parse_training_list(list_path)
    resolved = []
    missing = []

    for entry in entries:
        candidates = [
            culane_root / entry,
            culane_root / entry.lstrip("/"),
            culane_root / "images" / entry,
        ]

        found = None

        for candidate in candidates:
            if candidate.is_file():
                found = candidate.resolve()
                break

        if found is not None:
            resolved.append(found)
        else:
            # Search by normalized relative suffix as a fallback.
            matches = [
                p for p in culane_root.rglob(Path(entry).name)
                if p.is_file() and
                p.as_posix().lower().endswith(entry.lower())
            ]

            if matches:
                matches.sort(key=lambda p: p.as_posix().lower())
                resolved.append(matches[0].resolve())
            else:
                missing.append(entry)

    # Remove duplicate files and sort deterministically.
    resolved = sorted(
        set(resolved),
        key=lambda p: p.relative_to(culane_root.resolve()).as_posix().lower()
    )

    return resolved, missing


def collect_fallback_training_images(culane_root: Path):
    """
    Fallback when no CULane training list is available.

    This method recursively collects images while excluding directories whose
    names strongly indicate validation/test data or annotation directories.

    This fallback is less authoritative than using train_gt.txt.
    """
    excluded_tokens = {
        "test",
        "testing",
        "val",
        "validation",
        "laneseg_label_w16",
        "laneseg_label",
        "list",
    }

    images = []

    for path in culane_root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix not in IMAGE_EXTENSIONS:
            continue

        relative_parts = {
            part.lower()
            for part in path.relative_to(culane_root).parts
        }

        if relative_parts.intersection(excluded_tokens):
            continue

        images.append(path.resolve())

    return sorted(
        set(images),
        key=lambda p: p.relative_to(culane_root.resolve()).as_posix().lower()
    )


def sha256_file(path: Path, chunk_size=1024 * 1024):
    """Calculate SHA-256 hash of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def copy_or_link(source: Path, destination: Path, mode: str):
    """Copy or hardlink a selected image."""
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

    culane_root = args.culane_root.resolve()
    output_dir = args.output_dir.resolve()

    if not culane_root.is_dir():
        raise FileNotFoundError(
            f"CULane root directory does not exist: {culane_root}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CULane Reproducible Dataset Preparation")
    print("=" * 70)
    print(f"Dataset root : {culane_root}")
    print(f"Output       : {output_dir}")
    print(f"Requested    : {args.num_images}")
    print(f"Random seed  : {args.seed}")
    print("=" * 70)

    # Prefer the official CULane training list.
    training_list = find_training_list(culane_root)

    missing_entries = []
    selection_source = ""

    if training_list is not None:
        print(f"\nTraining list found: {training_list}")

        candidate_images, missing_entries = resolve_training_images(
            culane_root,
            training_list
        )

        selection_source = (
            "CULane training list "
            f"({training_list.relative_to(culane_root).as_posix()})"
        )

        print(
            f"Training images resolved from list: {len(candidate_images)}"
        )

        if missing_entries:
            print(
                f"Warning: {len(missing_entries)} entries from the training "
                "list could not be resolved."
            )

    else:
        print(
            "\nWARNING: No train_gt.txt or train.txt was found."
        )
        print(
            "Using recursive image discovery with test/validation/"
            "annotation directories excluded."
        )

        candidate_images = collect_fallback_training_images(culane_root)

        selection_source = (
            "Recursive image discovery with test/validation/annotation "
            "directories excluded"
        )

    print(f"Total candidate images: {len(candidate_images)}")

    if len(candidate_images) < args.num_images:
        raise RuntimeError(
            f"Only {len(candidate_images)} candidate images were found, "
            f"but {args.num_images} images were requested."
        )

    # Fixed seed + deterministic sorted candidate list.
    rng = random.Random(args.seed)

    selected_images = rng.sample(
        candidate_images,
        args.num_images
    )

    # Sort selected paths before assigning output filenames.
    selected_images.sort(
        key=lambda p: p.relative_to(culane_root).as_posix().lower()
    )

    manifest_path = output_dir / "culane_selection.csv"
    summary_path = output_dir / "selection_summary.json"

    manifest_rows = []

    print("\nCopying selected images...")

    for index, source_path in enumerate(selected_images, start=1):
        relative_path = source_path.relative_to(culane_root).as_posix()

        output_name = f"{index:06d}{source_path.suffix.lower()}"
        destination_path = output_dir / output_name

        copy_or_link(
            source_path,
            destination_path,
            args.copy_mode
        )

        manifest_rows.append({
            "selected_index": index,
            "output_filename": output_name,
            "source_relative_path": relative_path,
            "sha256": sha256_file(source_path),
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
                "sha256",
            ],
        )

        writer.writeheader()
        writer.writerows(manifest_rows)

    # Save reproducibility metadata.
    summary = {
        "dataset": "CULane",
        "source_root": str(culane_root),
        "selection_source": selection_source,
        "selection_method": (
            "Uniform random sampling without replacement from the "
            "deterministically sorted candidate training-image list."
        ),
        "num_candidate_images": len(candidate_images),
        "num_images_selected": len(selected_images),
        "random_seed": args.seed,
        "copy_mode": args.copy_mode,
        "image_extensions": sorted(IMAGE_EXTENSIONS),
        "training_list_found": (
            str(training_list.relative_to(culane_root).as_posix())
            if training_list is not None
            else None
        ),
        "unresolved_training_list_entries": len(missing_entries),
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
