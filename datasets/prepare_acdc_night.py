import os
import shutil
from pathlib import Path

# ============================================================
# ACDC Nighttime Dataset Preparation
# ============================================================

# CHANGE THIS PATH TO YOUR EXTRACTED ACDC DATASET
ACDC_ROOT = Path(r"D:\Datasets\ACDC")

# Output directory for OA-LoRA environmental data
OUTPUT_ROOT = Path(r"D:\OA-LoRA\dataset\environmental\acdc_night")

# Create output folders
for split in ["train", "val", "test"]:
    (OUTPUT_ROOT / split / "images").mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# Function to find image files
# ============================================================

def get_images(folder):
    extensions = {".png", ".jpg", ".jpeg"}

    return [
        file for file in folder.rglob("*")
        if file.is_file() and file.suffix.lower() in extensions
    ]


# ============================================================
# Locate ACDC nighttime folders
# ============================================================

# IMPORTANT:
# Modify these paths according to the folder structure
# created after extracting rgb_anon_trainvaltest.zip.

night_train = ACDC_ROOT / "rgb_anon" / "train" / "night"
night_val = ACDC_ROOT / "rgb_anon" / "val" / "night"
night_test = ACDC_ROOT / "rgb_anon" / "test" / "night"


# ============================================================
# Copy images
# ============================================================

def copy_images(source_folder, destination_folder):

    if not source_folder.exists():
        print(f"\nWARNING: Folder not found:")
        print(source_folder)
        return 0

    images = get_images(source_folder)

    count = 0

    for image in images:

        destination = destination_folder / image.name

        shutil.copy2(image, destination)

        count += 1

    return count


# ============================================================
# Process dataset
# ============================================================

train_count = copy_images(
    night_train,
    OUTPUT_ROOT / "train" / "images"
)

val_count = copy_images(
    night_val,
    OUTPUT_ROOT / "val" / "images"
)

test_count = copy_images(
    night_test,
    OUTPUT_ROOT / "test" / "images"
)


# ============================================================
# Print summary
# ============================================================

print("\n======================================")
print("ACDC NIGHTTIME DATASET PREPARATION")
print("======================================")

print(f"Training images   : {train_count}")
print(f"Validation images : {val_count}")
print(f"Test images       : {test_count}")

print("--------------------------------------")
print(f"Total images      : {train_count + val_count + test_count}")
print("--------------------------------------")

print("\nDataset preparation completed.")
print(f"Output directory: {OUTPUT_ROOT}")
