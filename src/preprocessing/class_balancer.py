from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def get_class_weights(
    labels: list[int],
    strategy: Literal['inverse_freq', 'effective_num'] = 'effective_num',
    beta: float = 0.999,
) -> torch.Tensor:
    class_counts = np.bincount(labels)
    n_classes = len(class_counts)

    if strategy == 'inverse_freq':
        weights = 1.0 / (class_counts + 1e-6)
    elif strategy == 'effective_num':
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / (effective_num + 1e-6)
    else:
        raise ValueError(f'Unknown strategy: {strategy}')

    weights = weights / weights.sum() * n_classes
    return torch.FloatTensor(weights)


def get_weighted_sampler(labels: list[int], class_weights: torch.Tensor) -> WeightedRandomSampler:
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(labels), replacement=True)
