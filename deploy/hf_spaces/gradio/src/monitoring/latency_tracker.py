from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

PREDICTION_COUNT = Counter(
    "melanoma_predictions_total",
    "Total predictions made",
    ["predicted_class", "confidence_level"],
)

INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds",
    "Inference latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0],
)

REVIEW_FLAGS = Counter(
    "predictions_requiring_review_total",
    "Predictions flagged for human review",
    ["reason"],
)

MELANOMA_PROBABILITY_HIST = Histogram(
    "melanoma_probability_distribution",
    "Distribution of melanoma probabilities",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

CACHE_HITS = Counter(
    "prediction_cache_hits_total",
    "Cache hit count",
)

MODEL_INFO = Gauge(
    "model_info",
    "Model metadata",
    ["model_version", "quantization"],
)


def record_prediction(
    *,
    predicted_class: str,
    confidence_level: str,
    latency_ms: float,
    melanoma_prob: float,
    requires_review: bool,
    review_reason: str | None,
    model_version: str,
    quantization: str,
    cache_hit: bool = False,
) -> None:
    PREDICTION_COUNT.labels(predicted_class, confidence_level).inc()
    INFERENCE_LATENCY.observe(max(latency_ms, 0.0) / 1000.0)
    MELANOMA_PROBABILITY_HIST.observe(melanoma_prob)
    if requires_review:
        REVIEW_FLAGS.labels(review_reason or "unspecified").inc()
    if cache_hit:
        CACHE_HITS.inc()
    MODEL_INFO.labels(model_version, quantization).set(1)


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
