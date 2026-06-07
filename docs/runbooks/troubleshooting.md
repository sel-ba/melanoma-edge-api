# Troubleshooting Runbook

## API returns 503 on /ready
- Check: Is the ONNX model file present? `ls models/onnx/`
- Check: Is disk >90% full? `df -h /`
- Check: Is memory >90% used? `free -h`

## Predictions always return benign
- Model may not be trained yet. Check `models/onnx/efficientnet_b0_int8.onnx` exists.
- Thresholds may be wrong. Check `configs/inference/thresholds.yaml`.

## High latency (>200ms)
- Check CPU usage: `top`
- Switch to INT8 model (smaller, faster)
- Reduce `intra_op_num_threads` in `engine.py`

## MLflow server not accessible
- Start with: `mlflow server --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root ./mlflow/artifacts --host 0.0.0.0 --port 5000`
- Or: `docker-compose up mlflow`

## Module import errors
- Run: `uv pip install -e ".[dev]"`
- Verify venv: `.venv/bin/python --version` should show 3.10.x
