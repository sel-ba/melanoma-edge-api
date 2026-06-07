# Monitoring Runbook

## Metrics Endpoint
- Prometheus metrics: `GET /api/v1/metrics`
- Health check: `GET /api/v1/health`
- Readiness check: `GET /api/v1/ready`

## Drift Detection
- Run drift check: `GET /api/v1/monitor/drift?reference_data_path=path/to/reference.csv&window_size=500`
- Reports saved to: `reports/drift/`

## Key Metrics
- `melanoma_predictions_total` — total predictions by class and confidence
- `inference_latency_seconds` — inference latency histogram
- `predictions_requiring_review_total` — flagged predictions by reason
- `melanoma_probability_distribution` — output probability distribution

## Alert Thresholds
- Latency warning: >150ms
- Latency critical: >300ms
- Drift detected: dataset_drift=True in Evidently report
- Disk usage: >90%
- Memory usage: >90%
