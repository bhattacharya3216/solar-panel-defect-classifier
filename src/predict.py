"""
src/predict.py

End-to-end inference: given an image, first runs the binary classifier
(clean vs defective). If defective, optionally runs the multiclass
classifier to identify the specific defect type.

Usage:
    from src.predict import predict

    result = predict("path/to/image.jpg")
    result = predict("path/to/image.jpg", include_multiclass=True)
"""

import numpy as np
import tensorflow as tf
from pathlib import Path

IMG_SIZE = (224, 224)
BINARY_MODEL_PATH = "models/binary_model.keras"
MULTICLASS_MODEL_PATH = "models/multiclass_model.keras"
MULTICLASS_CLASS_NAMES_PATH = "models/multiclass_class_names.txt"

BINARY_THRESHOLD = 0.2  # chosen via evaluate.py's threshold sweep (F1-optimal)

# Lazy-loaded models (loaded once, reused across calls)
_binary_model = None
_multiclass_model = None
_multiclass_class_names = None


def _load_binary_model():
    global _binary_model
    if _binary_model is None:
        _binary_model = tf.keras.models.load_model(BINARY_MODEL_PATH)
    return _binary_model


def _load_multiclass_model():
    global _multiclass_model, _multiclass_class_names
    if _multiclass_model is None:
        _multiclass_model = tf.keras.models.load_model(MULTICLASS_MODEL_PATH)
        with open(MULTICLASS_CLASS_NAMES_PATH) as f:
            _multiclass_class_names = [line.strip() for line in f if line.strip()]
    return _multiclass_model, _multiclass_class_names


def _load_and_preprocess_image(image_path: str):
    """Loads an image and prepares it for the model (no manual rescaling —
    both models have built-in normalization, matching dataset.py's approach)."""
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array


def predict(image_path: str, include_multiclass: bool = False):
    """
    Runs binary classification, and optionally multiclass, on a single image.

    Returns a dict:
        {
            "binary_label": "clean" | "defective",
            "binary_confidence": float,
            "multiclass_label": str or None,
            "multiclass_confidence": float or None
        }
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_array = _load_and_preprocess_image(image_path)

    # ---- Stage 1: Binary ----
    binary_model = _load_binary_model()
    binary_prob = binary_model.predict(img_array, verbose=0)[0][0]
    is_defective = binary_prob >= BINARY_THRESHOLD
    binary_label = "defective" if is_defective else "clean"

    result = {
        "binary_label": binary_label,
        "binary_confidence": float(binary_prob),
        "multiclass_label": None,
        "multiclass_confidence": None,
    }

    # ---- Stage 2: Multiclass (only if defective AND requested) ----
    if is_defective and include_multiclass:
        multiclass_model, class_names = _load_multiclass_model()
        probs = multiclass_model.predict(img_array, verbose=0)[0]
        predicted_idx = np.argmax(probs)

        result["multiclass_label"] = class_names[predicted_idx]
        result["multiclass_confidence"] = float(probs[predicted_idx])

    return result


if __name__ == "__main__":
    # Hardcoded path for quick testing — replace with a real filename from your test set
    image_path = "data/binary/test/defective/Bird (101).jpg"
    include_multiclass = True

    result = predict(image_path, include_multiclass=include_multiclass)

    print(f"\nImage: {image_path}")
    print(f"Binary: {result['binary_label']} (confidence: {result['binary_confidence']:.3f})")
    if result["multiclass_label"]:
        print(f"Defect type: {result['multiclass_label']} (confidence: {result['multiclass_confidence']:.3f})")