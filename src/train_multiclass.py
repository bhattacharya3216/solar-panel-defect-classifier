"""
src/train_multiclass.py

Trains the multi-class (6-class) defect classifier.
Phase 1: feature extraction (frozen backbone)
Phase 2: fine-tuning (partial unfreeze)

Usage:
    python src/train_multiclass.py
"""

import tensorflow as tf
from pathlib import Path
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from dataset import get_multiclass_datasets
from model_multiclass import build_multiclass_model, compile_model

# ---- CONFIG ----
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 12
PHASE1_LR = 1e-4
PHASE2_LR = 1e-5
FINE_TUNE_AT = 60


def compute_multiclass_class_weights(train_ds, num_classes, boost=None):
    """Computes class weights to counter imbalance. `boost` is an optional
    {class_index: multiplier} dict to manually push specific weak classes further."""
    labels = np.stack([y.numpy() for _, y in train_ds.unbatch()])
    integer_labels = np.argmax(labels, axis=1)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=integer_labels,
    )
    weights_dict = {i: w for i, w in enumerate(class_weights)}

    if boost:
        for class_idx, multiplier in boost.items():
            weights_dict[class_idx] *= multiplier

    return weights_dict


def get_callbacks(checkpoint_name: str, log_name: str):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / checkpoint_name),
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.CSVLogger(
            f"reports/{log_name}", append=False
        ),
    ]


def main():
    print("Loading datasets...")
    train_ds, val_ds, test_ds, class_names = get_multiclass_datasets()
    num_classes = len(class_names)
    print(f"Classes: {class_names}")

    class_weights = compute_multiclass_class_weights(train_ds, num_classes)
    print(f"Class weights: {class_weights}")

    # ---- Phase 1: Feature extraction ----
    print("\n=== Phase 1: Feature extraction (frozen backbone) ===")
    model = build_multiclass_model(num_classes=num_classes, base_trainable=False)
    model = compile_model(model, learning_rate=PHASE1_LR)

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE1_EPOCHS,
        callbacks=get_callbacks("multiclass_model_phase1.keras", "training_log_multiclass_phase1.csv"),
        class_weight=class_weights,
    )

    # ---- Phase 2: Fine-tuning ----
    print("\n=== Phase 2: Fine-tuning (partial unfreeze) ===")
    model = build_multiclass_model(num_classes=num_classes, base_trainable=True, fine_tune_at=FINE_TUNE_AT)
    model.load_weights(str(MODELS_DIR / "multiclass_model_phase1.keras"))
    model = compile_model(model, learning_rate=PHASE2_LR)

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE2_EPOCHS,
        callbacks=get_callbacks("multiclass_model_phase2.keras", "training_log_multiclass_phase2.csv"),
        class_weight=class_weights,
    )

    # ---- Save final model ----
    final_path = MODELS_DIR / "multiclass_model.keras"
    model.save(final_path)
    print(f"\nFinal model saved to {final_path}")

    # ---- Save class names for later use in predict.py ----
    with open(MODELS_DIR / "multiclass_class_names.txt", "w") as f:
        f.write("\n".join(class_names))

    # ---- Test set evaluation ----
    print("\n=== Test set evaluation ===")
    results = model.evaluate(test_ds)
    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()