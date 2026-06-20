from __future__ import annotations

import torch
import torch.nn.utils.prune as prune


def apply_structured_pruning(
    model: torch.nn.Module, amount: float = 0.2
) -> torch.nn.Module:
    """Apply L1-norm structured pruning to Conv2d and Linear layers.

    Structured pruning removes entire channels/neurons (not individual weights),
    which yields actual speedups on CPU without sparse-matrix support.

    amount: fraction of channels/neurons to prune per layer (0.0–1.0).
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.ln_structured(
                module, name="weight", amount=amount, n=1, dim=0
            )
            prune.remove(module, "weight")
        elif isinstance(module, torch.nn.Linear):
            prune.ln_structured(
                module, name="weight", amount=amount, n=1, dim=0
            )
            prune.remove(module, "weight")
    return model


def get_pruning_summary(model: torch.nn.Module) -> dict[str, float]:
    """Return sparsity ratio per layer type."""
    total_params = 0
    zero_params = 0
    for _, param in model.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
    sparsity = zero_params / total_params if total_params > 0 else 0.0
    return {
        "total_parameters": total_params,
        "zero_parameters": int(zero_params),
        "sparsity": sparsity,
    }
