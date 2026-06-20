---
title: Melanoma Detection API
emoji: ""
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Melanoma Detection — REST API

FastAPI backend for melanoma risk assessment.
Includes Swagger docs at `/docs`, Grad-CAM explainability, and Prometheus metrics.

**Endpoints:**
- `POST /api/v1/predict` — ONNX inference
- `POST /api/v1/explain` — Grad-CAM heatmaps
- `GET /api/v1/health` — Health check
- `GET /docs` — Interactive Swagger UI

**Research use only — not a medical device.**
