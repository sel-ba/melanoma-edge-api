from __future__ import annotations

import structlog

from src.api.schemas.response import PredictionClass, PredictionResponse

logger = structlog.get_logger()


def get_fallback_response(error: Exception, image_hash: str) -> PredictionResponse:
    """Return a safe fallback response when inference fails.

    In medical AI, the correct response to a system error is NOT to crash
    and return HTTP 500.  Instead we return a structured response with
    ``requires_review=True`` so the clinical device can display:
    "System uncertain — consult specialist."
    """
    logger.error(
        "inference_fallback_triggered",
        error=str(error),
        error_type=type(error).__name__,
        image_hash=image_hash,
    )

    return PredictionResponse(
        predicted_class="unknown",
        melanoma_probability=0.5,
        all_probabilities=[
            PredictionClass(label="benign", probability=0.5),
            PredictionClass(label="melanoma", probability=0.5),
        ],
        uncertainty=1.0,
        confidence_level="low",
        requires_review=True,
        review_reason="inference_error_fallback",
        is_calibrated=False,
        model_version="fallback",
        latency_ms=0.0,
        image_hash=image_hash,
    )
