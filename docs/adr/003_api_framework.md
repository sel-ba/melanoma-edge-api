# ADR 003: API Framework Choice

## Status
Accepted

## Context
Need a REST API framework for serving predictions on edge devices.

## Considered Options
1. **FastAPI** — async-native, auto-docs, Pydantic validation, production-proven
2. **Flask** — simple, well-known, synchronous, less performant under load
3. **aiohttp** — async, lower-level, no auto-docs
4. **Triton Inference Server** — purpose-built for ML, heavy, cloud-oriented

## Decision
**FastAPI** selected. Auto-generated OpenAPI docs provide a built-in integration
interface for clinical systems. Pydantic schemas ensure type safety.

## Consequences
- Async endpoint design avoids blocking the event loop during CPU-bound inference
- ThreadPoolExecutor used for ONNX inference (CPU-bound) in predict endpoint
- Structured logging via structlog integrates cleanly with FastAPI middleware
