from __future__ import annotations

import numpy as np


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error.

    Lower is better. Well-calibrated model has ECE < 0.05.
    Medical AI target: ECE < 0.10.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            accuracy_in_bin = float(labels[in_bin].mean())
            confidence_in_bin = float(probs[in_bin].mean())
            ece += (in_bin.sum() / len(probs)) * abs(
                confidence_in_bin - accuracy_in_bin
            )

    return float(ece)


def compute_reliability_curve(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> dict[str, np.ndarray]:
    """Return bin confidences and accuracies for reliability diagram."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    confidences = []
    accuracies = []

    for i in range(n_bins):
        in_bin = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            confidences.append(float(probs[in_bin].mean()))
            accuracies.append(float(labels[in_bin].mean()))
        else:
            confidences.append(float((bin_boundaries[i] + bin_boundaries[i + 1]) / 2))
            accuracies.append(0.0)

    return {
        "confidences": np.array(confidences),
        "accuracies": np.array(accuracies),
    }
