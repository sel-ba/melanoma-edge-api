#!/usr/bin/env bash
# ============================================================
# Prepare Hugging Face Spaces deployment directories.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# After running: each subdirectory contains a complete,
# self-contained Space ready to push to Hugging Face.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "==> Project root: $PROJECT_ROOT"
echo "==> Preparing Gradio Space (deploy/hf_spaces/gradio/) ..."

# ── Gradio Space ─────────────────────────────────────────────
GRADIO_DIR="$SCRIPT_DIR/gradio"
rm -rf "$GRADIO_DIR"
mkdir -p "$GRADIO_DIR"

# Dockerfile
cp "$PROJECT_ROOT/Dockerfile" "$GRADIO_DIR/Dockerfile"

# Application
cp "$PROJECT_ROOT/app.py" "$GRADIO_DIR/app.py"

# Source code
cp -r "$PROJECT_ROOT/src" "$GRADIO_DIR/src"

# Configs
cp -r "$PROJECT_ROOT/configs" "$GRADIO_DIR/configs"

# Models (copy only what's needed)
mkdir -p "$GRADIO_DIR/models/onnx"
cp "$PROJECT_ROOT/models/onnx/efficientnet_b0_fp32.onnx" "$GRADIO_DIR/models/onnx/"
mkdir -p "$GRADIO_DIR/models/checkpoints"
cp "$PROJECT_ROOT/models/checkpoints/best_model.pt" "$GRADIO_DIR/models/checkpoints/"

# .gitattributes for Git LFS
cat > "$GRADIO_DIR/.gitattributes" << 'LFS'
*.onnx filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
LFS

# README
cat > "$GRADIO_DIR/README.md" << 'EOF'
---
title: Melanoma Detection
emoji: ""
colorFrom: red
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Melanoma Detection — Gradio Demo

AI-powered melanoma risk assessment from dermoscopic images.
Uses EfficientNet-B0 with Grad-CAM explainability and clinical safety flags.

**Research use only — not a medical device.**
EOF

echo "   Done. Files in: $GRADIO_DIR"
ls -la "$GRADIO_DIR"

# ── API Space ────────────────────────────────────────────────
echo ""
echo "==> Preparing API Space (deploy/hf_spaces/api/) ..."

API_DIR="$SCRIPT_DIR/api"
rm -rf "$API_DIR"
mkdir -p "$API_DIR"

# Dockerfile (rename from Dockerfile.hf_api)
cp "$PROJECT_ROOT/Dockerfile.hf_api" "$API_DIR/Dockerfile"

# Requirements
cp "$PROJECT_ROOT/requirements.full.txt" "$API_DIR/requirements.txt"

# Source code
cp -r "$PROJECT_ROOT/src" "$API_DIR/src"

# Configs
cp -r "$PROJECT_ROOT/configs" "$API_DIR/configs"

# Models
mkdir -p "$API_DIR/models/onnx"
cp "$PROJECT_ROOT/models/onnx/efficientnet_b0_fp32.onnx" "$API_DIR/models/onnx/"
mkdir -p "$API_DIR/models/checkpoints"
cp "$PROJECT_ROOT/models/checkpoints/best_model.pt" "$API_DIR/models/checkpoints/"

# .gitattributes for Git LFS
cat > "$API_DIR/.gitattributes" << 'LFS'
*.onnx filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
LFS

# README
cat > "$API_DIR/README.md" << 'EOF'
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
EOF

echo "   Done. Files in: $API_DIR"
ls -la "$API_DIR"

# ── Summary ──────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "  Both Spaces are ready to deploy!"
echo "=============================================="
echo ""
echo "  Gradio Space:  $GRADIO_DIR"
echo "  API Space:     $API_DIR"
echo ""
echo "  Next steps:"
echo "  1. Clone your HF Space repos:"
echo "     git clone https://huggingface.co/spaces/YOUR_USER/melanoma-gradio"
echo "     git clone https://huggingface.co/spaces/YOUR_USER/melanoma-api"
echo ""
echo "  2. Copy files into each:"
echo "     cp -r $GRADIO_DIR/* melanoma-gradio/"
echo "     cp -r $API_DIR/* melanoma-api/"
echo ""
echo "  3. Push:"
echo "     cd melanoma-gradio && git add . && git commit -m 'Deploy' && git push"
echo "     cd melanoma-api   && git add . && git commit -m 'Deploy' && git push"
echo ""
echo "  That's it! HF Spaces will auto-build both Docker images."
echo "=============================================="
