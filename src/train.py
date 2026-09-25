"""
src/train.py

Trains the binary (clean vs defective) classifier.
Phase 1: feature extraction (frozen backbone)
Phase 2: fine-tuning (partial unfreeze), optional, run after Phase 1

Usage:
    python src/train.py
"""

import tensorflow as tf
from pathlib import Path

from dataset import get_binary_datasets
from model_binary import build_binary_model, compile_model

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# ---- CONFIG ----
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 5
PHASE1_LR = 1e-4
PHASE2_LR = 1e-5
FINE_TUNE_AT = 100  # layer index to start unfreezing from in MobileNetV3Small

def compute_binary_class_weights(train_ds):
    """Computes class weights to counter imbalance."""
    labels = np.concatenate([y.numpy() for _, y in train_ds.unbatch()])
    labels = labels.flatten()

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=labels,
    )
    return {0: class_weights[0], 1: class_weights[1]}

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
    train_ds, val_ds, test_ds, class_names = get_binary_datasets()
    print(f"Classes: {class_names}")

    # ---- Phase 1: Feature extraction ----
    print("\n=== Phase 1: Feature extraction (frozen backbone) ===")
    model = build_binary_model(base_trainable=False)
    model = compile_model(model, learning_rate=PHASE1_LR)

    class_weights = compute_binary_class_weights(train_ds)
    print(f"Class weights: {class_weights}")

    history1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=PHASE1_EPOCHS,
            callbacks=get_callbacks("binary_model_phase1.keras", "training_log_phase1.csv"),
            class_weight=class_weights,
        )

    # ---- Phase 2: Fine-tuning ----
    print("\n=== Phase 2: Fine-tuning (partial unfreeze) ===")
    model = build_binary_model(base_trainable=True, fine_tune_at=FINE_TUNE_AT)
    # Reload phase 1 weights before continuing
    model.load_weights(str(MODELS_DIR / "binary_model_phase1.keras"))
    model = compile_model(model, learning_rate=PHASE2_LR)

    history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=PHASE2_EPOCHS,
    callbacks=get_callbacks("binary_model_phase2.keras", "training_log_phase2.csv"),
    class_weight=class_weights,   
    )

    # ---- Save final model ----
    final_path = MODELS_DIR / "binary_model.keras"
    model.save(final_path)
    print(f"\nFinal model saved to {final_path}")

    # ---- Quick test set check ----
    print("\n=== Test set evaluation ===")
    results = model.evaluate(test_ds)
    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()