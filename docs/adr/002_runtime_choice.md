# ADR 002: Inference Runtime Choice

## Status
Accepted

## Context
Need a production inference runtime that works identically on x86 laptops,
ARM Raspberry Pi, and NVIDIA Jetson with no code changes.

## Considered Options
1. **ONNX Runtime** — hardware-agnostic, Microsoft-maintained, graph optimizations
2. **TorchScript** — PyTorch-native, fast but PyTorch-only ecosystem
3. **TensorRT** — fastest on NVIDIA, NVIDIA-only
4. **TFLite** — best for Android/embedded, Google ecosystem
5. **CoreML** — fastest on Apple Silicon, Apple-only

## Decision
**ONNX Runtime** selected for maximum portability across edge hardware targets.

## Consequences
- Production Docker image does NOT need PyTorch (~1GB savings)
- INT8 quantization works via ONNX Runtime quantization tools
- MC-Dropout uncertainty requires separate PyTorch model (not available in ONNX)
- Proxy uncertainty used for ONNX-only inference path
