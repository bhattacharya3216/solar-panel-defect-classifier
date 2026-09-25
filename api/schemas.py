"""
api/schemas.py

Request/response models for the prediction API.
"""

from pydantic import BaseModel
from typing import Optional


class PredictionResponse(BaseModel):
    binary_label: str
    binary_confidence: float
    multiclass_label: Optional[str] = None
    multiclass_confidence: Optional[float] = None


class ErrorResponse(BaseModel):
    detail: str