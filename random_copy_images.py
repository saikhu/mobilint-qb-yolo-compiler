# random_copy_images.py
import os
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ---- CONFIG ----
src_dir = Path("/home/mobilint/Desktop/usman/NIPA_Data_2025_v8/train/images/")
dst_dir = Path("calibration_images")   # will be created in current folder
num_to_copy = 1000
random.seed(42)
# -----------------

# Check source dir
if not src_dir.exists() or not src_dir.is_dir():
    raise SystemExit(f"Source directory does not exist or is not a directory: {src_dir}")

# Create destination dir if needed
dst_dir.mkdir(parents=True, exist_ok=True)

# Collect image files
image_files = [
    p for p in src_dir.iterdir()
    if p.is_file() and p.suffix.lower() in IMG_EXTS
]

total_images = len(image_files)
if total_images == 0:
    raise SystemExit(f"No image files found in source dir: {src_dir}")

if num_to_copy > total_images:
    print(f"[WARN] Requested {num_to_copy} images but only {total_images} available.")
    num_to_copy = total_images

chosen_files = random.sample(image_files, num_to_copy)

print(f"Copying {num_to_copy} images from {src_dir} -> {dst_dir}")
for src_path in chosen_files:
    dst_path = dst_dir / src_path.name
    shutil.copy2(src_path, dst_path)
    print(f"  {src_path.name} -> {dst_path}")

print("Done ✅")
