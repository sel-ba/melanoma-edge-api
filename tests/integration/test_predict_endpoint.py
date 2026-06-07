from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ONNX_MODEL = PROJECT_ROOT / "models" / "onnx" / "efficientnet_b0_fp32.onnx"

pytestmark = pytest.mark.skipif(
    not ONNX_MODEL.exists(),
    reason=f"ONNX model not found at {ONNX_MODEL}",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from src.api.main import app

    with TestClient(app) as c:
        yield c


def _make_jpeg() -> bytes:
    rng = np.random.RandomState(42)
    pixels = rng.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(pixels, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestPredictEndpoint:

    def test_valid_jpeg_returns_200(self, client) -> None:
        jpeg = _make_jpeg()
        resp = client.post(
            "/api/v1/predict",
            files={"file": ("test.jpg", jpeg, "image/jpeg")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "predicted_class" in data
        assert "melanoma_probability" in data
        assert "requires_review" in data
        assert data["confidence_level"] in ("high", "medium", "low")

    def test_invalid_content_type_returns_422(self, client) -> None:
        resp = client.post(
            "/api/v1/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert resp.status_code == 422

    def test_valid_png_returns_200(self, client) -> None:
        rng = np.random.RandomState(99)
        pixels = rng.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(pixels, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        resp = client.post(
            "/api/v1/predict",
            files={"file": ("test.png", buf.getvalue(), "image/png")},
        )
        assert resp.status_code == 200

    def test_cache_hit_on_repeat(self, client) -> None:
        jpeg = _make_jpeg()
        r1 = client.post(
            "/api/v1/predict",
            files={"file": ("test.jpg", jpeg, "image/jpeg")},
        )
        r2 = client.post(
            "/api/v1/predict",
            files={"file": ("test.jpg", jpeg, "image/jpeg")},
        )
        assert r1.status_code == 200
        assert r2.status_code == 200
        # Same image → same hash → same result
        assert r1.json()["image_hash"] == r2.json()["image_hash"]
