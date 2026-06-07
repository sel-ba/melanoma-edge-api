from __future__ import annotations

from src.inference.fallback import get_fallback_response


class TestFallbackResponse:
    """Fallback responses must always be safe (requires_review=True)."""

    def test_requires_review(self) -> None:
        resp = get_fallback_response(RuntimeError("test"), "hash123")
        assert resp.requires_review is True

    def test_confidence_level_is_low(self) -> None:
        resp = get_fallback_response(RuntimeError("test"), "hash123")
        assert resp.confidence_level == "low"

    def test_predicted_class_is_unknown(self) -> None:
        resp = get_fallback_response(RuntimeError("test"), "hash123")
        assert resp.predicted_class == "unknown"

    def test_preserves_image_hash(self) -> None:
        resp = get_fallback_response(RuntimeError("test"), "abc_hash_456")
        assert resp.image_hash == "abc_hash_456"

    def test_review_reason_set(self) -> None:
        resp = get_fallback_response(ValueError("broken"), "h")
        assert resp.review_reason == "inference_error_fallback"

    def test_not_calibrated(self) -> None:
        resp = get_fallback_response(RuntimeError("test"), "h")
        assert resp.is_calibrated is False

    def test_probability_is_neutral(self) -> None:
        resp = get_fallback_response(RuntimeError("test"), "h")
        assert resp.melanoma_probability == 0.5
