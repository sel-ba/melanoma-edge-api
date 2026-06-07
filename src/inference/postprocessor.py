from __future__ import annotations

import numpy as np

CLASS_NAMES = {0: "benign", 1: "melanoma"}


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def postprocess(logits: np.ndarray, temperature: float = 1.0) -> dict:
    """Convert raw ONNX logits to calibrated probabilities."""
    calibrated_logits = logits / max(temperature, 1e-6)
    probs = softmax(calibrated_logits)[0]

    melanoma_prob = float(probs[1])
    predicted_idx = int(probs.argmax())
    predicted_class = CLASS_NAMES[predicted_idx]

    return {
        "probs": probs,
        "melanoma_probability": melanoma_prob,
        "predicted_class": predicted_class,
        "predicted_class_idx": predicted_idx,
    }


def confidence_level(
    melanoma_prob: float, uncertainty: float
) -> str:
    """Return human-readable confidence level."""
    if uncertainty < 0.2:
        return "high"
    elif uncertainty < 0.4:
        return "medium"
    return "low"


def proxy_uncertainty(melanoma_prob: float) -> float:
    """Compute a simple uncertainty proxy from probability alone.

    When MC-Dropout is unavailable (ONNX-only mode), use distance from 0.5
    as a coarse uncertainty estimate.
    """
    return float(1.0 - abs(2.0 * melanoma_prob - 1.0))
