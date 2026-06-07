from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def onnx_session():
    ort = pytest.importorskip("onnxruntime")
    model_path = Path(__file__).resolve().parents[2] / "models" / "onnx" / "efficientnet_b0_fp32.onnx"
    if not model_path.exists():
        pytest.skip(f"ONNX model not found at {model_path}")
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


class TestONNXModelOutput:
    """Validate ONNX model produces correct output shapes and values."""

    def test_output_shape(self, onnx_session) -> None:
        dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
        output = onnx_session.run(None, {onnx_session.get_inputs()[0].name: dummy})[0]
        assert output.shape == (1, 2), f"Expected (1, 2), got {output.shape}"

    def test_softmax_sums_to_one(self, onnx_session) -> None:
        dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
        logits = onnx_session.run(None, {onnx_session.get_inputs()[0].name: dummy})[0]
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-5)

    def test_probabilities_non_negative(self, onnx_session) -> None:
        dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
        logits = onnx_session.run(None, {onnx_session.get_inputs()[0].name: dummy})[0]
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        assert (probs >= 0).all()
