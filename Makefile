SHELL := /bin/bash

.PHONY: help setup serve test lint lint-fix docker-build docker-run train export quantize benchmark

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Bootstrap development environment
	@if [[ ! -d .venv ]]; then uv venv .venv --python 3.10.20; fi
	uv pip install -e ".[dev]"

serve: ## Start local API server with hot-reload
	uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8080

test: ## Run test suite with coverage
	uv run pytest tests/ -v --cov=src --cov-report=term-missing

lint: ## Run linter and format check
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

lint-fix: ## Auto-fix lint issues
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/

train: ## Run training pipeline
	uv run python scripts/run_training.py --config configs/training/efficientnet_b0.yaml

export: ## Export model to ONNX
	uv run python -m src.optimization.onnx_exporter

quantize: ## Quantize ONNX model to INT8
	uv run python -m src.optimization.quantizer

benchmark: ## Run edge benchmarks
	uv run python -m src.optimization.benchmarker

docker-build: ## Build production Docker image
	docker build -f docker/Dockerfile.api -t melanoma-api:latest .

docker-run: ## Run API in Docker
	docker run --rm -p 8080:8080 -v $(PWD)/models/onnx:/app/models/onnx:ro -v $(PWD)/configs:/app/configs:ro melanoma-api:latest
