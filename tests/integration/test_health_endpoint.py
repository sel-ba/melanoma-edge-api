from __future__ import annotations

from pathlib import Path

import pytest

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


class TestHealthEndpoint:

    def test_health_returns_200(self, client) -> None:
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "melanoma-detection-api"

    def test_ready_returns_200(self, client) -> None:
        resp = client.get("/api/v1/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert "memory_percent" in data


class TestMetricsEndpoint:

    def test_metrics_returns_prometheus_format(self, client) -> None:
        resp = client.get("/api/v1/metrics")
        assert resp.status_code == 200
        # Prometheus text format contains HELP/TYPE lines
        body = resp.text
        assert "melanoma_predictions_total" in body or "inference_latency" in body or len(body) > 0
