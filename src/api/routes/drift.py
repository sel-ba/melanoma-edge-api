from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter(tags=["monitoring"])


@router.get("/monitor/drift")
def run_drift_check(
    request: Request,
    reference_data_path: str = Query(..., description="Path to reference CSV for drift comparison"),
    window_size: int = Query(500, description="Number of recent predictions to consider"),
):
    try:
        from src.monitoring.drift_detector import DriftMonitor
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=f"Drift monitoring unavailable: {exc}")

    recent = getattr(request.app.state, "recent_predictions", None)
    if recent is None:
        raise HTTPException(status_code=500, detail="Prediction history not initialized on the app.")

    if len(recent) == 0:
        return {"status": "no_predictions", "count": 0}

    try:
        monitor = DriftMonitor(reference_data_path)
        result = monitor.check_prediction_drift(production_predictions=recent, window_size=window_size)
        return result
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Reference data file not found.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
