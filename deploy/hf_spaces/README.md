# Deploy to Hugging Face Spaces (Free)

Two separate Spaces for the Melanoma Detection API — a **Gradio demo** and a **FastAPI backend**. Both run entirely on Hugging Face's free tier (16 GB RAM, 2 vCPU, 50 GB disk).

> **The `gradio/` and `api/` directories are pre-built and ready to push.**
> Run `./setup.sh` to regenerate them if needed.

---

## Quick Deploy (2 minutes)

After creating both Spaces on HF, just copy each directory's contents into the cloned repo and push:

```bash
# Clone your Space repos (replace YOUR_USER with your HF username)
git clone https://huggingface.co/spaces/YOUR_USER/melanoma-gradio
git clone https://huggingface.co/spaces/YOUR_USER/melanoma-api

# Copy pre-built files
cp -r deploy/hf_spaces/gradio/* melanoma-gradio/
cp -r deploy/hf_spaces/api/*    melanoma-api/

# Push both
cd melanoma-gradio && git add . && git commit -m "Deploy Gradio demo" && git push && cd ..
cd melanoma-api   && git add . && git commit -m "Deploy FastAPI backend" && git push && cd ..
```

HF Spaces will auto-build both Docker images (~10 min first build). That's it.

---

---

## Prerequisites

- A [Hugging Face account](https://huggingface.co/join) (free)
- [Git LFS](https://git-lfs.com/) installed (`git lfs install`)
- Docker installed locally (for optional pre-testing)

---

## Space 1: Gradio Demo (Full with Grad-CAM)

Interactive Gradio web UI with predict, explain (Grad-CAM), model comparison, and about tabs.

### 1. Create the Space

1. Go to https://huggingface.co/new-space
2. **Space name**: `melanoma-gradio`
3. **SDK**: **Docker**
4. **Docker template**: Blank
5. Click **Create Space**

### 2. Clone and Push

```bash
# Clone the empty Space repo
git clone https://huggingface.co/spaces/YOUR_USERNAME/melanoma-gradio
cd melanoma-gradio

# Copy files from your project
cp /path/to/PFA/code/Dockerfile .
cp /path/to/PFA/code/app.py .
cp -r /path/to/PFA/code/src .
cp -r /path/to/PFA/code/configs .
cp -r /path/to/PFA/code/models/onnx .
cp -r /path/to/PFA/code/models/checkpoints .

# Track large files with Git LFS
git lfs track "*.onnx" "*.pt"

# Push — HF Spaces will auto-build and deploy
git add .
git commit -m "Deploy Gradio demo with Grad-CAM"
git push
```

### 3. Wait for Build

HF Spaces builds the Docker image (~5–10 minutes first time). Once done, your demo is live at:

```
https://YOUR_USERNAME-melanoma-gradio.hf.space
```

---

## Space 2: FastAPI Backend (REST API + Swagger)

Full REST API with Swagger docs at `/docs`, `/api/v1/predict`, `/api/v1/explain`, Prometheus metrics, and health checks.

### 1. Create the Space

1. Go to https://huggingface.co/new-space
2. **Space name**: `melanoma-api`
3. **SDK**: **Docker**
4. **Docker template**: Blank
5. Click **Create Space**

### 2. Clone and Push

```bash
# Clone the empty Space repo
git clone https://huggingface.co/spaces/YOUR_USERNAME/melanoma-api
cd melanoma-api

# Use the HF-specific API Dockerfile as the main Dockerfile
cp /path/to/PFA/code/Dockerfile.hf_api ./Dockerfile

# Copy other required files
cp /path/to/PFA/code/requirements.full.txt .
cp -r /path/to/PFA/code/src .
cp -r /path/to/PFA/code/configs .
cp -r /path/to/PFA/code/models/onnx .
cp -r /path/to/PFA/code/models/checkpoints .

# Track large files
git lfs track "*.onnx" "*.pt"

# Push
git add .
git commit -m "Deploy FastAPI backend"
git push
```

### 3. Live Endpoints

| Endpoint | URL |
|----------|-----|
| Swagger Docs | `https://YOUR_USERNAME-melanoma-api.hf.space/docs` |
| ReDoc | `https://YOUR_USERNAME-melanoma-api.hf.space/redoc` |
| Health Check | `https://YOUR_USERNAME-melanoma-api.hf.space/api/v1/health` |
| Predict | `POST https://YOUR_USERNAME-melanoma-api.hf.space/api/v1/predict` |
| Explain | `POST https://YOUR_USERNAME-melanoma-api.hf.space/api/v1/explain` |

### 4. Test the API

```bash
# Health check
curl https://YOUR_USERNAME-melanoma-api.hf.space/api/v1/health

# Predict
curl -X POST https://YOUR_USERNAME-melanoma-api.hf.space/api/v1/predict \
  -F "file=@path/to/dermoscopy_image.jpg"
```

---

## Optional: Pre-Test Locally

Build and run the Docker images locally to verify before pushing:

```bash
# Test Gradio image
docker build -t melanoma-gradio -f Dockerfile .
docker run --rm -p 7860:7860 melanoma-gradio
# Visit http://localhost:7860

# Test API image
docker build -t melanoma-api -f Dockerfile.hf_api .
docker run --rm -p 7860:7860 melanoma-api
# Visit http://localhost:7860/docs
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Build fails (OOM)** | HF Spaces has 16GB RAM — the full image (~2.5GB) fits. If it fails, use the slim `Dockerfile` without PyTorch. |
| **Model not loading** | Ensure `models/onnx/efficientnet_b0_fp32.onnx` and `models/checkpoints/best_model.pt` exist and are tracked with Git LFS. |
| **Grad-CAM not working** | Check that `timm` is installed and the checkpoint file exists. The Gradio app gracefully disables the tab if unavailable. |
| **Cold start delay** | HF Spaces sleeps after ~48h of inactivity. First request wakes it (~30s startup). |
| **Port already in use** | HF Spaces requires port **7860**. Both Dockerfiles are configured for this. |

---

## Resource Usage (Free Tier)

| Metric | Gradio Space | API Space |
|--------|-------------|-----------|
| Image size | ~2.5 GB | ~2.5 GB |
| RAM at idle | ~300 MB | ~400 MB |
| RAM at inference | ~1.5 GB | ~1.5 GB |
| Cold start | ~30 s | ~30 s |
| Requests/min | Unlimited* | Unlimited* |

*Subject to fair use — HF Spaces does not enforce hard rate limits on the free tier.
