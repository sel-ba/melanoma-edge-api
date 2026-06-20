from __future__ import annotations

import base64
import hashlib
from io import BytesIO
from pathlib import Path
import time
import json

import numpy as np
import torch
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image

from src.api.schemas.response import ExplainResponse, PredictionClass, PredictionResponse
from src.explainability.gradcam import GradCAM
from src.models.efficientnet import MelanomaClassifier
from src.preprocessing.augmentation import get_val_transforms

router = APIRouter(tags=["explain"])

CLASS_NAMES = {0: "benign", 1: "melanoma"}

_model = None
_gradcam = None
_val_transforms = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _find_last_conv_layer_name(model: torch.nn.Module) -> str | None:
    conv_layers = [
        name for name, module in model.named_modules() if isinstance(module, torch.nn.Conv2d)
    ]
    return conv_layers[-1] if conv_layers else None


def _resolve_checkpoint(project_root: Path) -> Path:
    run_summary_path = project_root / "reports" / "artifacts" / "training_full_run_summary.json"
    run_id = None
    if run_summary_path.exists():
        try:
            run_id = json.loads(run_summary_path.read_text(encoding="utf-8")).get("run_id")
        except json.JSONDecodeError:
            run_id = None
    checkpoint_candidates = []
    if run_id:
        mlflow_root = project_root / "mlflow"
        if mlflow_root.exists():
            for path in mlflow_root.rglob("best_model.pt"):
                if run_id in str(path):
                    checkpoint_candidates.append(path)
    checkpoint_candidates.extend([
        project_root / "models" / "checkpoints" / "best_model.pt",
        project_root / "notebooks" / "models" / "checkpoints" / "best_model.pt",
    ])
    checkpoint_path = next((path for path in checkpoint_candidates if path.exists()), None)
    if checkpoint_path is None:
        tried = ", ".join(str(path) for path in checkpoint_candidates)
        raise FileNotFoundError(f"Missing checkpoint. Tried: {tried}")
    return checkpoint_path


def _load_explain_assets(project_root: Path) -> tuple[MelanomaClassifier, GradCAM]:
    global _model, _gradcam, _val_transforms
    if _model is not None and _gradcam is not None:
        return _model, _gradcam

    model = MelanomaClassifier(model_name="efficientnet_b0", num_classes=2, pretrained=False)

    checkpoint_path = _resolve_checkpoint(project_root)
    checkpoint = torch.load(checkpoint_path, map_location=_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(_device)
    model.eval()

    target_layer = _find_last_conv_layer_name(model)
    if target_layer is None:
        raise RuntimeError("No conv layer found for Grad-CAM")

    _val_transforms = get_val_transforms()
    _model = model
    _gradcam = GradCAM(model, target_layer)
    return _model, _gradcam


def _encode_png(image: np.ndarray) -> str:
    buffer = BytesIO()
    Image.fromarray(image.astype(np.uint8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@router.post("/explain", response_model=ExplainResponse)
async def explain_prediction(request: Request, file: UploadFile = File(...)):
    start_time = time.perf_counter()

    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG or PNG.",
        )

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large. Maximum size is 10MB.")

    image_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
    project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

    try:
        model, gradcam = await run_in_threadpool(_load_explain_assets, project_root)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Explainability not ready: {exc}")

    original = np.array(Image.open(BytesIO(image_bytes)).convert("RGB"))
    input_tensor = _val_transforms(image=original)["image"].to(_device)

    logits = model(input_tensor.unsqueeze(0))
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
    predicted_class_idx = int(probs.argmax())
    melanoma_prob = float(probs[1])
    predicted_class = CLASS_NAMES[predicted_class_idx]

    heatmap, _, confidence = gradcam.generate(input_tensor)
    overlay = gradcam.overlay_on_image(original, heatmap, alpha=0.4)
    heatmap_b64 = _encode_png(overlay)

    uncertainty = 1.0 - abs(2 * melanoma_prob - 1.0)
    if uncertainty < 0.2:
        confidence_level = "high"
    elif uncertainty < 0.4:
        confidence_level = "medium"
    else:
        confidence_level = "low"

    engine = request.app.state.engine
    requires_review, review_reason = engine.is_high_confidence(melanoma_prob, uncertainty)
    latency_ms = (time.perf_counter() - start_time) * 1000

    prediction = PredictionResponse(
        predicted_class=predicted_class,
        melanoma_probability=melanoma_prob,
        all_probabilities=[
            PredictionClass(label="benign", probability=float(probs[0])),
            PredictionClass(label="melanoma", probability=float(probs[1])),
        ],
        uncertainty=float(uncertainty),
        confidence_level=confidence_level,
        requires_review=requires_review,
        review_reason=review_reason,
        is_calibrated=False,
        model_version="efficientnet-b0-fp32-onnx",
        latency_ms=round(latency_ms, 2),
        image_hash=image_hash,
    )

    return ExplainResponse(
        prediction=prediction,
        gradcam_heatmap_b64=heatmap_b64,
        attention_regions=[],
    )
