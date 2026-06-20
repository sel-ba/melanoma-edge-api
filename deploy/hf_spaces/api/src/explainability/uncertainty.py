from __future__ import annotations

import numpy as np
import torch


def mc_dropout_predict(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    n_passes: int = 30,
    device: str = "cpu",
) -> dict:
    """Monte Carlo Dropout uncertainty estimation."""

    model.eval()

    def enable_dropout(module):
        if isinstance(
            module,
            (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d),
        ):
            module.train()

    model.apply(enable_dropout)

    input_tensor = input_tensor.unsqueeze(0).to(device)
    predictions = []

    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            predictions.append(probs[0])

    predictions = np.array(predictions)
    mean_probs = predictions.mean(axis=0)
    std_probs = predictions.std(axis=0)

    eps = 1e-8
    predictive_entropy = -np.sum(mean_probs * np.log(mean_probs + eps))
    individual_entropies = -np.sum(predictions * np.log(predictions + eps), axis=1).mean()
    mutual_information = predictive_entropy - individual_entropies

    return {
        "mean_probs": mean_probs,
        "std_probs": std_probs,
        "predictive_entropy": float(predictive_entropy),
        "mutual_information": float(mutual_information),
        "melanoma_probability": float(mean_probs[1]),
        "uncertainty": float(std_probs[1]),
        "is_uncertain": float(std_probs[1]) > 0.15,
    }
