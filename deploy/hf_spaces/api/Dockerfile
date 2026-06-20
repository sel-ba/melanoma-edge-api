# ============================================================
# Hugging Face Spaces — FastAPI Backend (Full with Explainability)
# ============================================================
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r melanoma && useradd -r -g melanoma melanoma

WORKDIR /app

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    fastapi>=0.111 \
    uvicorn>=0.29 \
    numpy>=1.26 \
    Pillow>=10.0 \
    PyYAML>=6.0 \
    python-multipart>=0.0.9 \
    structlog>=24.4 \
    onnxruntime>=1.18 \
    psutil>=5.9 \
    prometheus-client>=0.20 \
    pandas>=2.2 \
    evidently>=0.4 \
    torch>=2.3 \
    torchvision>=0.18 \
    timm>=1.0 \
    albumentations>=1.4 \
    opencv-python-headless>=4.9

COPY src/ ./src/
COPY configs/ ./configs/
COPY models/onnx/ ./models/onnx/
COPY models/checkpoints/ ./models/checkpoints/

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV NO_ALBUMENTATIONS_UPDATE=1
ENV ENABLE_EXPLAIN=1

USER melanoma

EXPOSE 7860

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
