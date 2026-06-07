from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class TemperatureScaling(nn.Module):
    """Temperature scaling for confidence calibration."""

    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(
        self,
        model: torch.nn.Module,
        val_loader,
        device: str = "cpu",
        max_iter: int = 50,
    ) -> float:
        model.eval()
        logits_list = []
        labels_list = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = model(images)
                logits_list.append(logits.detach().cpu())
                labels_list.append(labels.detach().cpu())

        all_logits = torch.cat(logits_list)
        all_labels = torch.cat(labels_list)

        optimizer = torch.optim.LBFGS(
            [self.temperature],
            lr=0.01,
            max_iter=max_iter,
        )
        criterion = nn.CrossEntropyLoss()

        def eval_closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(all_logits)
            loss = criterion(scaled_logits, all_labels)
            loss.backward()
            return loss

        optimizer.step(eval_closure)

        optimal_t = float(self.temperature.item())
        return optimal_t

    def compute_ece(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            in_bin = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
            if in_bin.sum() > 0:
                accuracy_in_bin = labels[in_bin].mean()
                confidence_in_bin = probs[in_bin].mean()
                ece += (in_bin.sum() / len(probs)) * abs(confidence_in_bin - accuracy_in_bin)

        return float(ece)
