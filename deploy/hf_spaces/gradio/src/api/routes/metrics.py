from __future__ import annotations

from fastapi import APIRouter, Response

from src.monitoring.latency_tracker import render_metrics

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
def metrics() -> Response:
    payload, content_type = render_metrics()
    return Response(content=payload, media_type=content_type)
