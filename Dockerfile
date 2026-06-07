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
    gradio>=6.15.2 \
    plotly>=6.7.0 \
    kaleido>=1.3.0 \
    qrcode[pil]>=8.2 \
    numpy>=1.26 \
    onnxruntime>=1.18 \
    Pillow>=10.0 \
    PyYAML>=6.0

COPY src/ ./src/
COPY configs/ ./configs/
COPY models/onnx/ ./models/onnx/
COPY app.py .

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

USER melanoma

EXPOSE 7860

CMD ["python", "app.py"]
