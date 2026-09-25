"""
api/main.py

FastAPI service wrapping the binary + multiclass solar panel defect
classification pipeline. Accepts image uploads only.

Usage (local dev):
    uvicorn api.main:app --reload
"""

import sys
from pathlib import Path

# Allow importing from src/ regardless of where uvicorn is launched from
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
import numpy as np

from predict import (
    _load_binary_model,
    _load_multiclass_model,
    _load_and_preprocess_image,
    BINARY_THRESHOLD,
)
from api.schemas import PredictionResponse

app = FastAPI(
    title="Solar Panel Defect Classifier",
    description="Two-stage image classification: clean/defective, then defect type",
    version="1.0.0",
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}
IMG_SIZE = (224, 224)


def _preprocess_uploaded_image(file_bytes: bytes) -> np.ndarray:
    """Loads image bytes (from an upload) into the same array format
    predict.py's _load_and_preprocess_image produces from a file path."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array


@app.get("/")
def root():
    return {"status": "ok", "message": "Solar Panel Defect Classifier API"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(
    file: UploadFile = File(...),
    include_multiclass: bool = True,
):
    # ---- Validate input is actually an image ----
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Only JPEG/PNG images are accepted.",
        )

    file_bytes = await file.read()

    try:
        img_array = _preprocess_uploaded_image(file_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded file as an image.")

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