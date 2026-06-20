"""INT8 model quantization for ONNX Runtime.

.. warning::

   EfficientNet-B0 is **not compatible** with INT8 quantization at acceptable
   accuracy.  All strategies tested (dynamic QInt8, dynamic QUInt8, static QDQ
   with entropy/minmax calibration, per-channel QDQ) produced **10–15 % AUC
   drops** (0.91 → 0.76–0.81).  This is caused by SiLU activations and
   depthwise-separable convolutions — both are fundamentally hostile to
   integer quantisation.

   The FP32 ONNX model (16.6 MB) is recommended for production.  Keep this
   module for architectures that *do* benefit from INT8 (e.g. ResNet50).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)

try:
    from onnxruntime.quantization import preprocess as _ort_preprocess

    _PREPROCESS_FN: Callable | None = _ort_preprocess.preprocess
except Exception:
    try:
        from onnxruntime.quantization import quant_pre_process as _ort_quant_pre_process

        _PREPROCESS_FN = _ort_quant_pre_process
    except Exception:
        _PREPROCESS_FN = None


class CalibrationReader(CalibrationDataReader):
    """Calibration data reader for static INT8 quantization."""

    def __init__(self, calibration_loader, n_samples: int = 300) -> None:
        self.data = []
        count = 0
        for batch in calibration_loader:
            for img in batch["image"]:
                self.data.append({"input_image": img.unsqueeze(0).cpu().numpy()})
                count += 1
                if count >= n_samples:
                    break
            if count >= n_samples:
                break
        self.iter_next = iter(self.data)

    def get_next(self):
        return next(self.iter_next, None)

    def rewind(self) -> None:
        self.iter_next = iter(self.data)


def quantize_model_int8(
    fp32_onnx_path: str,
    output_path: str,
    calibration_loader=None,
    strategy: str = "dynamic",
    quant_format: str = "qdq",
    activation_type: str = "quint8",
    per_channel: bool = True,
    n_samples: int = 300,
) -> str:
    """Quantize ONNX model to INT8 (dynamic or static)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if strategy == "dynamic":
        quantize_dynamic(
            model_input=fp32_onnx_path,
            model_output=str(output_path),
            weight_type=QuantType.QInt8,
        )
    elif strategy == "static":
        if calibration_loader is None:
            raise ValueError("Static quantization requires calibration_loader")
        calibration_reader = CalibrationReader(calibration_loader, n_samples=n_samples)

        model_input = fp32_onnx_path
        if _PREPROCESS_FN is not None:
            preprocess_path = output_path.with_suffix(".preprocess.onnx")
            try:
                _PREPROCESS_FN(fp32_onnx_path, str(preprocess_path))
                model_input = str(preprocess_path)
            except TypeError:
                _PREPROCESS_FN(input_model_path=fp32_onnx_path, output_model_path=str(preprocess_path))
                model_input = str(preprocess_path)

        if isinstance(quant_format, str):
            if quant_format.lower() == "qdq":
                resolved_format = QuantFormat.QDQ
            elif quant_format.lower() in {"qoperator", "qoperator"}:
                resolved_format = QuantFormat.QOperator
            else:
                raise ValueError(f"Unknown quant_format: {quant_format}")
        else:
            resolved_format = quant_format

        if isinstance(activation_type, str):
            if activation_type.lower() == "qint8":
                resolved_activation = QuantType.QInt8
            elif activation_type.lower() == "quint8":
                resolved_activation = QuantType.QUInt8
            else:
                raise ValueError(f"Unknown activation_type: {activation_type}")
        else:
            resolved_activation = activation_type

        quantize_static(
            model_input=model_input,
            model_output=str(output_path),
            calibration_data_reader=calibration_reader,
            quant_format=resolved_format,
            activation_type=resolved_activation,
            weight_type=QuantType.QInt8,
            per_channel=per_channel,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    original_size = Path(fp32_onnx_path).stat().st_size / 1024 / 1024
    quantized_size = output_path.stat().st_size / 1024 / 1024

    print(f"Original: {original_size:.2f} MB")
    print(f"Quantized: {quantized_size:.2f} MB")
    print(f"Compression: {original_size / quantized_size:.1f}x")

    return str(output_path)
