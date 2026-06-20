from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionClass(BaseModel):
    label: str
    probability: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    predicted_class: str
    melanoma_probability: float = Field(ge=0.0, le=1.0)
    all_probabilities: list[PredictionClass]
    uncertainty: float = Field(ge=0.0)
    confidence_level: str
    requires_review: bool
    review_reason: str | None = None
    is_calibrated: bool = True
    model_version: str
    latency_ms: float
    image_hash: str


class ExplainResponse(BaseModel):
    prediction: PredictionResponse
    gradcam_heatmap_b64: str
    attention_regions: list[str]
