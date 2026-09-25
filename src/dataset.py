"""
src/dataset.py

Builds tf.data.Dataset objects from the folder structure created by data_prep.py.

Usage:
    from src.dataset import get_binary_datasets, get_multiclass_datasets

    train_ds, val_ds, test_ds, class_names = get_binary_datasets()
"""

import tensorflow as tf
from tensorflow.keras import layers

# ---- CONFIG ----
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

BINARY_DIR = "data/binary"
MULTICLASS_DIR = "data/multiclass"


# ---- Augmentation (applied only to training data) ----
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
], name="data_augmentation")


def _load_split(directory: str, split: str, label_mode: str):
    """Loads one split (train/val/test) from a directory using image_dataset_from_directory."""
    return tf.keras.utils.image_dataset_from_directory(
        f"{directory}/{split}",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode=label_mode,
        seed=SEED,
        shuffle=(split == "train"),  # only shuffle training data
    )


def _prepare(ds: tf.data.Dataset, augment: bool = False):
    # """Applies normalization, optional augmentation, and prefetching."""
    # normalization = layers.Rescaling(1. / 255)

    # ds = ds.map(lambda x, y: (normalization(x), y),
    #             num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)

    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)


def get_binary_datasets():
    """Returns (train_ds, val_ds, test_ds, class_names) for the binary clean/defective task."""
    train_raw = _load_split(BINARY_DIR, "train", label_mode="binary")
    class_names = train_raw.class_names  # capture before mapping

    val_raw = _load_split(BINARY_DIR, "val", label_mode="binary")
    test_raw = _load_split(BINARY_DIR, "test", label_mode="binary")

    train_ds = _prepare(train_raw, augment=True)
    val_ds = _prepare(val_raw, augment=False)
    test_ds = _prepare(test_raw, augment=False)

    return train_ds, val_ds, test_ds, class_names


def get_multiclass_datasets():
    """Returns (train_ds, val_ds, test_ds, class_names) for the 6-class defect task."""
    train_raw = _load_split(MULTICLASS_DIR, "train", label_mode="categorical")
    class_names = train_raw.class_names

    val_raw = _load_split(MULTICLASS_DIR, "val", label_mode="categorical")
    test_raw = _load_split(MULTICLASS_DIR, "test", label_mode="categorical")

    train_ds = _prepare(train_raw, augment=True)
    val_ds = _prepare(val_raw, augment=False)
    test_ds = _prepare(test_raw, augment=False)

    return train_ds, val_ds, test_ds, class_names


if __name__ == "__main__":
    # Quick sanity check when run directly
    print("Loading binary datasets...")
    train_ds, val_ds, test_ds, class_names = get_binary_datasets()
    print(f"Binary class names: {class_names}")
    for images, labels in train_ds.take(1):
        print(f"Batch shape: {images.shape}, labels shape: {labels.shape}")

    print("\nLoading multiclass datasets...")
    train_ds_mc, val_ds_mc, test_ds_mc, class_names_mc = get_multiclass_datasets()
    print(f"Multiclass class names: {class_names_mc}")
    for images, labels in train_ds_mc.take(1):
        print(f"Batch shape: {images.shape}, labels shape: {labels.shape}")