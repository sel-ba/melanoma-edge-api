#!/bin/bash
set -euo pipefail

echo "=== Melanoma Detection API — Jetson Setup ==="

# Jetson devices have CUDA — use TensorRT execution provider

# Install onnxruntime with GPU support
pip install onnxruntime-gpu

# Or run Docker with NVIDIA runtime
docker run --rm \
    --runtime nvidia \
    -e ORT_PROVIDER=TensorrtExecutionProvider \
    -p 8080:8080 \
    melanoma-api:latest
