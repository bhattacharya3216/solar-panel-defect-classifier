"""
src/data_prep.py

Splits data/raw/faulty_solar_panel/<class>/ images into:
  - data/multiclass/{train,val,test}/<class>/         (all 6 classes, for Stage 2)
  - data/binary/{train,val,test}/{clean,defective}/    (clean vs everything else, for Stage 1)

Usage:
    python src/data_prep.py
"""

import shutil
import random
from pathlib import Path

# ---- CONFIG ----
RAW_DIR = Path("data/raw/faulty_solar_panel")
MULTICLASS_DIR = Path("data/multiclass")
BINARY_DIR = Path("data/binary")

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42
CLEAN_CLASS_NAME = "clean"   # matched case-insensitively against folder names

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

random.seed(SEED)


def get_class_folders(raw_dir: Path):
    """Return list of class folder names found under raw_dir."""
    return [d.name for d in raw_dir.iterdir() if d.is_dir()]


def get_images(class_path: Path):
    """Return only actual image files, filtering out stray non-image files."""
    return [f for f in class_path.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS]


def split_files(files: list, ratios: dict):
    """Shuffle and split a list of file paths into train/val/test."""
    files = files.copy()
    random.shuffle(files)
    n = len(files)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])

    return {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }


def copy_files(file_list: list, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for f in file_list:
        shutil.copy2(f, dest_dir / f.name)


def build_multiclass_split(class_folders: list):
    print("Building multiclass split...")
    for class_name in class_folders:
        class_path = RAW_DIR / class_name
        images = get_images(class_path)
        splits = split_files(images, SPLIT_RATIOS)

        for split_name, files in splits.items():
            dest = MULTICLASS_DIR / split_name / class_name
            copy_files(files, dest)

        print(f"  {class_name}: {len(images)} images -> "
              f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")


def build_binary_split(class_folders: list):
    print("\nBuilding binary split...")
    for class_name in class_folders:
        class_path = RAW_DIR / class_name
        images = get_images(class_path)
        splits = split_files(images, SPLIT_RATIOS)

        binary_label = "clean" if class_name.lower() == CLEAN_CLASS_NAME else "defective"

        for split_name, files in splits.items():
            dest = BINARY_DIR / split_name / binary_label
            copy_files(files, dest)

        print(f"  {class_name} -> {binary_label}: {len(images)} images")


if __name__ == "__main__":
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Expected raw data at {RAW_DIR.resolve()}, but it doesn't exist. "
            f"Check the folder name/path."
        )

    classes = get_class_folders(RAW_DIR)
    if not classes:
        raise RuntimeError(f"No class subfolders found in {RAW_DIR.resolve()}")

    print(f"Found classes: {classes}\n")

    build_multiclass_split(classes)
    build_binary_split(classes)

    print("\nDone. Check data/multiclass/ and data/binary/")