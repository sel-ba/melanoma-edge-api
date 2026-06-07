from __future__ import annotations

import pytest


class TestIsHighConfidence:
    """Test the threshold-based review logic in ONNXInferenceEngine.

    We test the method in isolation by constructing an engine with known
    thresholds.  The engine uses:
        melanoma_threshold = 0.5
        uncertainty_threshold = 0.2
        low_confidence_threshold = 0.15
        boundary_margin = 0.15  (hard-coded in engine)
    """

    @pytest.fixture(autouse=True)
    def _setup(self, onnx_model_path, threshold_config_path):
        from src.inference.engine import ONNXInferenceEngine

        self.engine = ONNXInferenceEngine(
            model_path=str(onnx_model_path),
            threshold_config_path=str(threshold_config_path),
        )

    def test_near_boundary_requires_review(self) -> None:
        """Probability within 0.15 of threshold → requires review."""
        requires, reason = self.engine.is_high_confidence(0.48, uncertainty=0.1)
        assert requires is True
        assert reason == "prediction_near_decision_boundary"

    def test_high_uncertainty_requires_review(self) -> None:
        """Uncertainty > 0.2 → requires review."""
        requires, reason = self.engine.is_high_confidence(0.9, uncertainty=0.3)
        assert requires is True
        assert reason == "high_epistemic_uncertainty"

    def test_confident_benign_no_review(self) -> None:
        """Very low probability + low uncertainty → no review."""
        requires, reason = self.engine.is_high_confidence(0.05, uncertainty=0.05)
        assert requires is False
        assert reason is None

    def test_confident_melanoma_no_review(self) -> None:
        """High probability + low uncertainty → no review."""
        requires, reason = self.engine.is_high_confidence(0.95, uncertainty=0.05)
        assert requires is False
        assert reason is None

    def test_boundary_priority_over_uncertainty(self) -> None:
        """When both boundary and uncertainty trigger, boundary wins (checked first)."""
        requires, reason = self.engine.is_high_confidence(0.50, uncertainty=0.3)
        assert requires is True
        assert reason == "prediction_near_decision_boundary"
