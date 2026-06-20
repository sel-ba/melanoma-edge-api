from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_size: int = 224,
    opset_version: int = 17,
    dynamic_axes: bool = True,
) -> str:
    """Export PyTorch model to ONNX format and validate parity."""
    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    dummy_input = torch.randn(1, 3, input_size, input_size, device=device)

    input_names = ["input_image"]
    output_names = ["class_logits"]

    axes = {}
    if dynamic_axes:
        axes = {
            "input_image": {0: "batch_size"},
            "class_logits": {0: "batch_size"},
        }

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=axes,
        do_constant_folding=True,
        export_params=True,
    )

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    _verify_onnx_parity(model, output_path, dummy_input)

    return str(output_path)


def _verify_onnx_parity(
    model: torch.nn.Module,
    onnx_path: Path,
    test_input: torch.Tensor,
    tolerance: float = 1e-4,
) -> None:
    """Verify that ONNX model outputs match PyTorch outputs."""
    model.eval()
    with torch.no_grad():
        pytorch_output = model(test_input).detach().cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_output = session.run(
        None,
        {"input_image": test_input.detach().cpu().numpy()},
    )[0]

    max_diff = float(np.abs(pytorch_output - onnx_output).max())
    if max_diff > tolerance:
        raise ValueError(
            f"ONNX parity check FAILED: max diff {max_diff:.2e} > {tolerance}"
        )
