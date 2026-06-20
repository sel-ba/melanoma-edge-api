# ADR 001: Model Architecture Choice

## Status
Accepted

## Context
Need a CNN architecture for melanoma detection on edge devices (Raspberry Pi 4, Jetson Nano).
Requirements: <5MB model size, <200ms inference on CPU, >0.85 ROC-AUC.

## Considered Options
1. **EfficientNet-B0** — 5.3M params, 0.39ms/im on V100, well-studied
2. **MobileNetV3-Small** — 2.5M params, faster inference, lower accuracy
3. **ResNet50** — 25.6M params, high accuracy, too large for edge
4. **ViT-Tiny** — Transformer-based, worse on small datasets

## Decision
**EfficientNet-B0** selected as primary. ResNet50 kept as baseline for comparison.

## Consequences
- INT8 quantization required for <5MB target (FP32 is ~20MB)
- Two-phase training (frozen backbone then full fine-tune) needed for small dataset
- ONNX opset ≥15 required for EfficientNet operations
