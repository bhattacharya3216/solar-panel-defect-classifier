"""
src/model_multiclass.py

Stage 2: Multi-class defect classifier (6 classes) using transfer learning.
Only trained/run on images already classified as "defective" by Stage 1.

Usage:
    from src.model_multiclass import build_multiclass_model

    model = build_multiclass_model(num_classes=6)
"""

import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (224, 224)


def build_multiclass_model(num_classes: int, base_trainable: bool = False, fine_tune_at: int = None):
    """
    Builds the multi-class defect classifier.

    Args:
        num_classes: number of defect types to classify.
        base_trainable: if False, backbone is frozen (Phase 1 - feature extraction).
                         if True, backbone is trainable (Phase 2 - fine-tuning).
        fine_tune_at: if base_trainable=True, freeze all layers BEFORE this index,
                      only unfreeze layers from this index onward. If None and
                      base_trainable=True, the entire backbone is unfrozen.
    """
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )

    base_model.trainable = base_trainable

    if base_trainable and fine_tune_at is not None:
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs, training=False if not base_trainable else None)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="multiclass_defect_classifier")
    return model


def compile_model(model, learning_rate: float = 1e-4):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    return model


if __name__ == "__main__":
    # Quick sanity check
    model = build_multiclass_model(num_classes=6)
    model = compile_model(model)
    model.summary()