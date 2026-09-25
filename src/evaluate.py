"""
src/evaluate.py

Evaluates the trained binary model on the test set:
- Confusion matrix
- Classification report (precision/recall/F1 per class)
- Precision-recall curve across thresholds (to help pick a better cutoff than 0.5)

Usage:
    python src/evaluate.py
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)
from pathlib import Path

from dataset import get_binary_datasets

MODEL_PATH = "models/binary_model.keras"
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


def get_predictions(model, dataset):
    """Runs the model over a dataset and returns true labels + predicted probabilities."""
    y_true = []
    y_prob = []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().flatten())
        y_prob.extend(preds.flatten())

    return np.array(y_true), np.array(y_prob)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - Binary Classifier")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")
    return cm


def plot_precision_recall_curve(y_true, y_prob, save_path):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, precisions[:-1], label="Precision")
    plt.plot(thresholds, recalls[:-1], label="Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision & Recall vs Threshold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved precision-recall curve to {save_path}")

    return precisions, recalls, thresholds


def threshold_sweep(y_true, y_prob, thresholds_to_try=None):
    """Prints precision/recall/F1 for defective class across a range of thresholds."""
    from sklearn.metrics import precision_score, recall_score, f1_score

    if thresholds_to_try is None:
        thresholds_to_try = [0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]

    print(f"\n{'Threshold':<12}{'Precision':<12}{'Recall':<12}{'F1':<12}")
    best_f1 = -1
    best_t = 0.5

    for t in thresholds_to_try:
        y_pred = (y_prob >= t).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"{t:<12.3f}{p:<12.3f}{r:<12.3f}{f1:<12.3f}")
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    print(f"\nBest F1 threshold: {best_t:.3f} (F1={best_f1:.3f})")
    return best_t


def main():
    print("Loading model and test data...")
    model = tf.keras.models.load_model(MODEL_PATH)
    _, _, test_ds, class_names = get_binary_datasets()
    print(f"Classes: {class_names}")

    y_true, y_prob = get_predictions(model, test_ds)
    y_pred_default = (y_prob >= 0.5).astype(int)

    # ---- Classification report at default threshold ----
    print("\n=== Classification Report (threshold=0.5) ===")
    print(classification_report(y_true, y_pred_default, target_names=class_names))

    # ---- Confusion matrix ----
    cm = plot_confusion_matrix(
        y_true, y_pred_default, class_names,
        REPORTS_DIR / "confusion_matrix_binary.png"
    )
    print("Confusion matrix:")
    print(cm)

    # ---- Precision-recall curve across thresholds ----
    precisions, recalls, thresholds = plot_precision_recall_curve(
        y_true, y_prob, REPORTS_DIR / "precision_recall_curve.png"
    )

    # ---- Suggest a better threshold ----
    best_t = threshold_sweep(y_true, y_prob)
    y_pred_adjusted = (y_prob >= best_t).astype(int)
    print(f"\n=== Classification Report (threshold={best_t:.3f}) ===")
    print(classification_report(y_true, y_pred_adjusted, target_names=class_names))


if __name__ == "__main__":
    main()