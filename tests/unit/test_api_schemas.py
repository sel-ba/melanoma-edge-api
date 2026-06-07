from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.schemas.response import PredictionClass, PredictionResponse


def _valid_response_kwargs() -> dict:
    return {
        "predicted_class": "benign",
        "melanoma_probability": 0.15,
        "all_probabilities": [
            PredictionClass(label="benign", probability=0.85),
            PredictionClass(label="melanoma", probability=0.15),
        ],
        "uncertainty": 0.1,
        "confidence_level": "high",
        "requires_review": False,
        "review_reason": None,
        "is_calibrated": True,
        "model_version": "test-v1",
        "latency_ms": 42.5,
        "image_hash": "abc123",
    }


class TestPredictionResponse:

    def test_valid_response(self) -> None:
        resp = PredictionResponse(**_valid_response_kwargs())
        assert resp.predicted_class == "benign"
        assert resp.melanoma_probability == 0.15

    def test_rejects_probability_above_one(self) -> None:
        kwargs = _valid_response_kwargs()
        kwargs["melanoma_probability"] = 1.5
        with pytest.raises(ValidationError):
            PredictionResponse(**kwargs)

    def test_rejects_negative_probability(self) -> None:
        kwargs = _valid_response_kwargs()
        kwargs["melanoma_probability"] = -0.1
        with pytest.raises(ValidationError):
            PredictionResponse(**kwargs)

    def test_rejects_negative_uncertainty(self) -> None:
        kwargs = _valid_response_kwargs()
        kwargs["uncertainty"] = -0.5
        with pytest.raises(ValidationError):
            PredictionResponse(**kwargs)

    def test_review_reason_optional(self) -> None:
        kwargs = _valid_response_kwargs()
        kwargs["review_reason"] = None
        resp = PredictionResponse(**kwargs)
        assert resp.review_reason is None


class TestPredictionClass:

    def test_valid_class(self) -> None:
        pc = PredictionClass(label="melanoma", probability=0.87)
        assert pc.probability == 0.87

    def test_rejects_probability_above_one(self) -> None:
        with pytest.raises(ValidationError):
            PredictionClass(label="melanoma", probability=1.01)

    def test_rejects_negative_probability(self) -> None:
        with pytest.raises(ValidationError):
            PredictionClass(label="benign", probability=-0.01)
