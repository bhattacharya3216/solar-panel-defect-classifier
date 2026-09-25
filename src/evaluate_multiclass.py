"""
src/evaluate_multiclass.py

Evaluates the trained multiclass model on the test set:
- Confusion matrix (6x6, shows which classes get confused with which)
- Classification report (per-class precision/recall/F1)

Usage:
    python src/evaluate_multiclass.py
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from pathlib import Path

from dataset import get_multiclass_datasets

MODEL_PATH = "models/multiclass_model.keras"
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


def get_predictions(model, dataset):
    """Runs the model over a dataset, returns true class indices + predicted class indices."""
    y_true = []
    y_pred = []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    return np.array(y_true), np.array(y_pred)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap="Blues", values_format="d", ax=ax, xticks_rotation=45)
    plt.title("Confusion Matrix - Multiclass Defect Classifier")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")
    return cm


def main():
    print("Loading model and test data...")
    model = tf.keras.models.load_model(MODEL_PATH)
    _, _, test_ds, class_names = get_multiclass_datasets()
    print(f"Classes: {class_names}")

    y_true, y_pred = get_predictions(model, test_ds)

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    cm = plot_confusion_matrix(y_true, y_pred, class_names, REPORTS_DIR / "confusion_matrix_multiclass.png")
    print("Confusion matrix:")
    print(cm)


if __name__ == "__main__":
    main()