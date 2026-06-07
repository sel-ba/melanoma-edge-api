# Deployment Runbook

## Local Development
```bash
make serve          # Start API on http://localhost:8080
make test           # Run test suite
make lint           # Run lint checks
```

## Docker Deployment
```bash
make docker-build   # Build production image
make docker-run     # Run container locally
docker-compose up   # Full local stack (API + MLflow + Prometheus)
```

## Edge Deployment
See `deploy/edge/raspi/setup.sh` for Raspberry Pi 4.
See `deploy/edge/jetson/setup.sh` for NVIDIA Jetson.

## Model Update
1. Train new model: `make train`
2. Export to ONNX: `make export`
3. Quantize: `make quantize`
4. Benchmark: `make benchmark`
5. Rebuild Docker: `make docker-build`
6. Deploy to edge: follow edge setup script
