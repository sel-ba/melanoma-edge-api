from __future__ import annotations

from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os

import structlog
import yaml
from fastapi import FastAPI

from src.api.middleware.logging import RequestLoggingMiddleware
from src.monitoring.logger import setup_logging
from src.api.routes import health, metrics, predict, drift
from src.inference.cache import PredictionCache
from src.inference.engine import ONNXInferenceEngine
from src.inference.preprocessor import ImagePreprocessor

logger = structlog.get_logger()

_explain_enabled = os.getenv("ENABLE_EXPLAIN", "1").lower() in {"1", "true", "yes"}
_explain_import_error: str | None = None
if _explain_enabled:
    try:
        from src.api.routes import explain
    except Exception as exc:
        _explain_enabled = False
        _explain_import_error = str(exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()

    if not _explain_enabled and _explain_import_error:
        logger.warning("explain_disabled", reason=_explain_import_error)

    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "configs" / "inference" / "api.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    def resolve_path(value: str, default: str) -> Path:
        candidate = Path(value or default)
        if not candidate.is_absolute():
            return project_root / candidate
        return candidate

    model_path = resolve_path(
        config.get("model_path", "models/onnx/efficientnet_b0_fp32.onnx"),
        "models/onnx/efficientnet_b0_fp32.onnx",
    )
    thresholds_path = resolve_path(
        config.get("thresholds_path", "configs/inference/thresholds.yaml"),
        "configs/inference/thresholds.yaml",
    )

    app.state.engine = ONNXInferenceEngine(
        model_path=str(model_path),
        threshold_config_path=str(thresholds_path),
    )
    app.state.preprocessor = ImagePreprocessor(input_size=224)
    app.state.cache = PredictionCache(max_size=1000, ttl_seconds=3600)
    app.state.thread_pool = ThreadPoolExecutor(max_workers=int(config.get("workers", 2)))
    # In-memory ring buffer for recent production predictions (used by drift checks)
    app.state.recent_predictions: list[dict] = []

    yield

    app.state.thread_pool.shutdown(wait=True)


app = FastAPI(
    title="Melanoma Detection API",
    description="Edge-deployable dermatoscopy AI (research use only)",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(RequestLoggingMiddleware)

app.include_router(health.router, prefix="/api/v1")
app.include_router(predict.router, prefix="/api/v1")
app.include_router(metrics.router, prefix="/api/v1")
app.include_router(drift.router, prefix="/api/v1")
if _explain_enabled:
    app.include_router(explain.router, prefix="/api/v1")
