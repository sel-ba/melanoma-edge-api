from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric
from evidently.report import Report


class DriftMonitor:
    """Monitor production predictions for distribution shift."""

    def __init__(self, reference_data_path: str, reports_dir: str = "reports/drift") -> None:
        self.reference_df = pd.read_csv(reference_data_path)
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def check_prediction_drift(
        self,
        production_predictions: list[dict],
        window_size: int = 500,
    ) -> dict:
        if len(production_predictions) < window_size:
            return {"status": "insufficient_data", "count": len(production_predictions)}

        current_df = pd.DataFrame(
            [
                {
                    "melanoma_probability": p["melanoma_probability"],
                    "uncertainty": p["uncertainty"],
                    "requires_review": int(p["requires_review"]),
                    "latency_ms": p["latency_ms"],
                }
                for p in production_predictions[-window_size:]
            ]
        )

        report = Report(
            metrics=[
                DatasetDriftMetric(),
                ColumnDriftMetric(column_name="melanoma_probability"),
                ColumnDriftMetric(column_name="uncertainty"),
            ]
        )

        reference_sample = self.reference_df.sample(
            min(window_size, len(self.reference_df)),
            random_state=42,
        )

        report.run(
            reference_data=reference_sample,
            current_data=current_df,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"drift_report_{timestamp}.html"
        report.save_html(str(report_path))

        result = report.as_dict()
        drift_detected = result["metrics"][0]["result"].get("dataset_drift", False)

        return {
            "drift_detected": drift_detected,
            "report_path": str(report_path),
            "window_size": window_size,
            "timestamp": timestamp,
        }
