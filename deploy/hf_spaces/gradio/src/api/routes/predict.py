from __future__ import annotations

import hashlib
import time

import structlog
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from src.api.schemas.response import PredictionClass, PredictionResponse
from src.inference.fallback import get_fallback_response
from src.monitoring.latency_tracker import record_prediction

router = APIRouter(tags=["prediction"])
logger = structlog.get_logger()


@router.post("/predict", response_model=PredictionResponse)
async def predict_melanoma(request: Request, file: UploadFile = File(...)):
    start_time = time.perf_counter()

    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG or PNG.",
        )

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large. Maximum size is 10MB.")

    image_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
    cache = request.app.state.cache
    cached = cache.get(image_hash)
    if cached is not None:
        record_prediction(
            predicted_class=cached.predicted_class,
            confidence_level=cached.confidence_level,
            latency_ms=0.0,
            melanoma_prob=cached.melanoma_probability,
            requires_review=cached.requires_review,
            review_reason=cached.review_reason,
            model_version=cached.model_version,
            quantization="onnx",
            cache_hit=True,
        )
        # append to recent predictions buffer for drift monitoring
        try:
            request.app.state.recent_predictions.append(
                {
                    "melanoma_probability": float(cached.melanoma_probability),
                    "uncertainty": float(cached.uncertainty),
                    "requires_review": bool(cached.requires_review),
                    "latency_ms": float(0.0),
                }
            )
        except Exception:
            pass
        return cached

    preprocessor = request.app.state.preprocessor
    try:
        preprocessed = await run_in_threadpool(preprocessor.process_bytes, image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Image preprocessing failed: {exc}")

    engine = request.app.state.engine
    try:
        result = await run_in_threadpool(engine.predict, preprocessed["tensor"])
    except Exception as exc:
        # Return safe fallback instead of HTTP 500
        logger.error("inference_failed", error=str(exc), image_hash=image_hash)
        return get_fallback_response(exc, image_hash)

    melanoma_prob = float(result["melanoma_probability"])
    uncertainty = 1.0 - abs(2 * melanoma_prob - 1.0)
    if uncertainty < 0.2:
        confidence_level = "high"
    elif uncertainty < 0.4:
        confidence_level = "medium"
    else:
        confidence_level = "low"

    requires_review, review_reason = engine.is_high_confidence(melanoma_prob, uncertainty)
    latency_ms = (time.perf_counter() - start_time) * 1000

    response = PredictionResponse(
        predicted_class=result["predicted_class"],
        melanoma_probability=melanoma_prob,
        all_probabilities=[
            PredictionClass(label="benign", probability=float(result["probs"][0])),
            PredictionClass(label="melanoma", probability=float(result["probs"][1])),
        ],
        uncertainty=float(uncertainty),
        confidence_level=confidence_level,
        requires_review=requires_review,
        review_reason=review_reason,
        is_calibrated=False,
        model_version="efficientnet-b0-fp32-onnx",
        latency_ms=round(latency_ms, 2),
        image_hash=image_hash,
    )

    # Record Prometheus metrics
    record_prediction(
        predicted_class=result["predicted_class"],
        confidence_level=confidence_level,
        latency_ms=latency_ms,
        melanoma_prob=melanoma_prob,
        requires_review=requires_review,
        review_reason=review_reason,
        model_version="efficientnet-b0-fp32-onnx",
        quantization="onnx",
        cache_hit=False,
    )

    # append to recent predictions buffer for drift monitoring
    try:
        request.app.state.recent_predictions.append(
            {
                "melanoma_probability": melanoma_prob,
                "uncertainty": float(uncertainty),
                "requires_review": bool(requires_review),
                "latency_ms": float(latency_ms),
            }
        )
    except Exception:
        pass

    cache.set(image_hash, response)
    return response
