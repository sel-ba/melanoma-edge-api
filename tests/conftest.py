from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONNX_MODEL_PATH = PROJECT_ROOT / "models" / "onnx" / "efficientnet_b0_fp32.onnx"
THRESHOLDS_PATH = PROJECT_ROOT / "configs" / "inference" / "thresholds.yaml"


def _make_jpeg_bytes(width: int = 224, height: int = 224) -> bytes:
    """Create a synthetic RGB JPEG image and return its bytes."""
    rng = np.random.RandomState(42)
    pixels = rng.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(pixels, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture()
def test_image_bytes() -> bytes:
    """Synthetic 224×224 JPEG image."""
    return _make_jpeg_bytes(224, 224)


@pytest.fixture()
def small_image_bytes() -> bytes:
    """Synthetic 100×100 JPEG image."""
    return _make_jpeg_bytes(100, 100)


@pytest.fixture()
def large_image_bytes() -> bytes:
    """Synthetic 500×500 JPEG image."""
    return _make_jpeg_bytes(500, 500)


@pytest.fixture()
def onnx_model_path() -> Path:
    if not ONNX_MODEL_PATH.exists():
        pytest.skip(f"ONNX model not found at {ONNX_MODEL_PATH}")
    return ONNX_MODEL_PATH


@pytest.fixture()
def threshold_config_path() -> Path:
    if not THRESHOLDS_PATH.exists():
        pytest.skip(f"Thresholds config not found at {THRESHOLDS_PATH}")
    return THRESHOLDS_PATH
