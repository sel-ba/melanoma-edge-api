from __future__ import annotations

from pathlib import Path

import onnxruntime as ort
import psutil
from fastapi import APIRouter, Request, Response

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict:
    return {"status": "healthy", "service": "melanoma-detection-api"}


@router.get("/ready")
async def readiness_check(request: Request):
    issues = []
    engine = request.app.state.engine
    model_path = Path(engine.model_path)
    if not model_path.exists():
        issues.append("model_file_missing")

    disk = psutil.disk_usage("/")
    if disk.percent > 90:
        issues.append(f"low_disk_space:{disk.percent:.0f}%")

    mem = psutil.virtual_memory()
    if mem.percent > 90:
        issues.append(f"high_memory_usage:{mem.percent:.0f}%")

    if issues:
        return Response(content=str({"status": "not_ready", "issues": issues}), status_code=503)

    return {
        "status": "ready",
        "ort_providers": ort.get_available_providers(),
        "memory_percent": mem.percent,
        "disk_percent": disk.percent,
    }
