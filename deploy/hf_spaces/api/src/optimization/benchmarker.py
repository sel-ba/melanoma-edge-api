from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def benchmark_onnx_model(
    onnx_path: str,
    input_size: int = 224,
    n_warmup: int = 20,
    n_runs: int = 200,
    providers: list[str] | None = None,
) -> dict:
    """Benchmark ONNX model latency on CPU."""
    if providers is None:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(onnx_path, providers=providers)
    dummy_input = np.random.randn(1, 3, input_size, input_size).astype(np.float32)

    for _ in range(n_warmup):
        session.run(None, {"input_image": dummy_input})

    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        session.run(None, {"input_image": dummy_input})
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    latencies = np.array(latencies)

    results = {
        "model_path": onnx_path,
        "model_size_mb": Path(onnx_path).stat().st_size / 1024 / 1024,
        "providers": providers,
        "n_runs": n_runs,
        "mean_ms": float(latencies.mean()),
        "std_ms": float(latencies.std()),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms": float(latencies.min()),
        "max_ms": float(latencies.max()),
        "throughput_fps": float(1000 / latencies.mean()),
    }

    return results
