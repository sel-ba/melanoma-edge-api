from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Gradient-weighted Class Activation Mapping."""

    def __init__(self, model: torch.nn.Module, target_layer_name: str) -> None:
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        target_layer = self._get_layer(self.target_layer_name)

        def forward_hook(_module, _input, output):
            self.activations = output.detach()

        def backward_hook(_module, _grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def _get_layer(self, layer_name: str) -> torch.nn.Module:
        module = self.model
        for part in layer_name.split("."):
            if part in module._modules:
                module = module._modules[part]
            else:
                module = getattr(module, part)
        return module

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> Tuple[np.ndarray, int, float]:
        self.model.eval()
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device).unsqueeze(0).requires_grad_(True)

        output = self.model(input_tensor)
        probs = torch.softmax(output, dim=1)
        predicted_class = int(output.argmax(dim=1).item())
        confidence = float(probs[0, predicted_class].item())

        if target_class is None:
            target_class = predicted_class

        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture gradients or activations.")

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        weighted_activations = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(weighted_activations)

        cam = cam.squeeze().detach().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, predicted_class, confidence

    def overlay_on_image(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
    ) -> np.ndarray:
        h, w = original_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET,
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = (1 - alpha) * original_image + alpha * heatmap_colored
        return np.clip(overlay, 0, 255).astype(np.uint8)
