# Architecture Decision Records

## System Design

### Overview
Edge-deployable melanoma detection API running on resource-constrained hardware.
Offline-first, sub-200ms inference, uncertainty-aware.

### Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Primary model | EfficientNet-B0 | Best accuracy/parameter tradeoff at <5MB |
| Inference runtime | ONNX Runtime | Hardware-agnostic, runs on CPU/GPU/NPU |
| API framework | FastAPI | Async-native, auto-docs, production-proven |
| Experiment tracking | MLflow | Self-hostable, no cloud lock-in |
| Dataset versioning | DVC | Git-compatible, S3/local storage backends |
| Data validation | Great Expectations | Declarative, generates HTML reports |
| Monitoring | Evidently AI + Prometheus | Drift detection, self-hostable metrics |
| Containerization | Docker | Universal edge compatibility |
| Quantization | INT8 via ONNX | 4x size reduction, ~2x speed on CPU |

### Pipeline Flow
```
Raw Data → Validation (GE) → Preprocessing → Training (MLflow)
→ Export ONNX → Quantize INT8 → FastAPI → Docker → Edge Deploy
```

### Safety Design
- Predictions near decision boundary → `requires_review=True`
- High epistemic uncertainty → flagged for clinical review
- System errors → structured fallback response (not HTTP 500)
- All predictions include confidence levels and uncertainty estimates
