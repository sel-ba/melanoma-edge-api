"""Gradio frontend — Melanoma Detection AI."""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import numpy as np
import onnxruntime as ort
import plotly.graph_objects as go
from PIL import Image

# ── Constants ──────────────────────────────────────────────────────────────
CLASS_NAMES = {0: "Benign", 1: "Melanoma"}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SIZE = 224
MODEL_PATH = Path("models/onnx/efficientnet_b0_fp32.onnx")
CHECKPOINT_PATH = Path("models/checkpoints/best_model.pt")

# ── Colour palette ─────────────────────────────────────────────────────────
C_BENIGN    = "#059669"   # emerald-600
C_MELANOMA  = "#DC2626"   # red-600
C_WARNING   = "#D97706"   # amber-600
C_PRIMARY   = "#0D9488"   # teal-600
C_MUTED     = "#64748B"   # slate-500

# ── ONNX Engine ────────────────────────────────────────────────────────────
_session: ort.InferenceSession | None = None
_input_name: str = ""
_output_name: str = ""


def _get_session() -> ort.InferenceSession:
    global _session, _input_name, _output_name
    if _session is None:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        _session = ort.InferenceSession(
            str(MODEL_PATH), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        _input_name = _session.get_inputs()[0].name
        _output_name = _session.get_outputs()[0].name
    return _session


def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, 0).astype(np.float32)


def predict_onnx(image: Image.Image) -> dict:
    tensor = preprocess(image)
    sess = _get_session()
    logits = sess.run([_output_name], {_input_name: tensor})[0]
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    probs = probs[0]
    melanoma_prob = float(probs[1])
    predicted_idx = int(probs.argmax())
    uncertainty = float(1.0 - abs(2.0 * melanoma_prob - 1.0))
    return {
        "predicted": CLASS_NAMES[predicted_idx],
        "melanoma_prob": melanoma_prob,
        "benign_prob": float(probs[0]),
        "uncertainty": uncertainty,
        "confidence": "High" if uncertainty < 0.2 else ("Medium" if uncertainty < 0.4 else "Low"),
        "requires_review": uncertainty > 0.2 or (0.35 < melanoma_prob < 0.65),
    }


# ── GradCAM (requires PyTorch checkpoint) ───────────────────────────────────
_gradcam_model = None
_gradcam = None
_transforms = None
_device = None

GRADCAM_AVAILABLE = CHECKPOINT_PATH.exists()


def _load_gradcam():
    global _gradcam_model, _gradcam, _transforms, _device
    if _gradcam_model is not None:
        return
    import torch
    from src.models.efficientnet import MelanomaClassifier
    from src.explainability.gradcam import GradCAM
    from src.preprocessing.augmentation import get_val_transforms

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MelanomaClassifier(model_name="efficientnet_b0", num_classes=2, pretrained=False)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=_device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(_device)
    model.eval()
    _transforms = get_val_transforms()

    conv_layers = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Conv2d)]
    _gradcam = GradCAM(model, conv_layers[-1])
    _gradcam_model = model


def generate_gradcam(image: Image.Image) -> np.ndarray | None:
    if not GRADCAM_AVAILABLE:
        return None
    try:
        _load_gradcam()
        original = np.array(image.convert("RGB"))
        transformed = _transforms(image=original)["image"].to(_device)
        heatmap, _, _ = _gradcam.generate(transformed)
        return _gradcam.overlay_on_image(original, heatmap, alpha=0.4)
    except Exception:
        return None


# ── Gradio Handlers ─────────────────────────────────────────────────────────
def handle_predict(image: Image.Image | None) -> tuple:
    if image is None:
        return "", "", None, gr.update(visible=False, value="")
    result = predict_onnx(image)

    label = result["predicted"]
    color = C_MELANOMA if label == "Melanoma" else C_BENIGN
    melanoma_pct = result["melanoma_prob"] * 100

    summary = f"## <span style='color:{color}'>{label}</span>"
    details = (
        f"**Melanoma risk:** {melanoma_pct:.1f}%  \n"
        f"**Confidence:** {result['confidence']}  \n"
        f"**Uncertainty:** {result['uncertainty']:.3f}"
    )

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=melanoma_pct,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Risk Assessment", "font": {"size": 16, "color": "#334155"}},
        delta={"reference": 50, "increasing": {"color": C_MELANOMA}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94A3B8"},
            "bar": {"color": C_MELANOMA if melanoma_pct > 50 else C_BENIGN},
            "steps": [
                {"range": [0, 30], "color": "#D1FAE5"},
                {"range": [30, 70], "color": "#FEF3C7"},
                {"range": [70, 100], "color": "#FEE2E2"},
            ],
            "threshold": {
                "line": {"color": "#64748B", "width": 2},
                "thickness": 0.75,
                "value": 50,
            },
        },
    ))
    fig.update_layout(
        height=350, margin=dict(l=30, r=30, t=40, b=20),
        paper_bgcolor="#FFFFFF", font={"color": "#334155"},
    )

    if result["requires_review"]:
        review_msg = (
            "## Clinical Review Recommended\n\n"
            "This prediction falls near the decision boundary or has high uncertainty. "
            "A dermatologist should review this case."
        )
        return summary, details, fig, gr.update(visible=True, value=review_msg)
    else:
        return summary, details, fig, gr.update(visible=False, value="")


def handle_explain(image: Image.Image | None) -> tuple:
    if image is None:
        return None, ""
    result = predict_onnx(image)
    gradcam_img = generate_gradcam(image)

    if gradcam_img is None:
        text = "**Grad‑CAM not available.** The PyTorch checkpoint is required for heatmap generation."
    else:
        text = (
            f"**Prediction:** {result['predicted']} · "
            f"**Melanoma:** {result['melanoma_prob']:.1%} · "
            f"**Confidence:** {result['confidence']}\n\n"
            "*Red/yellow areas indicate regions that most influenced the prediction.*"
        )
    return gradcam_img, text


# ── Theme ──────────────────────────────────────────────────────────────────
_theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="*neutral_50",
    block_background_fill="white",
    block_border_width="1px",
    block_border_color="*neutral_200",
    block_radius="lg",
    button_primary_background_fill="*primary_500",
    button_primary_text_color="white",
)

# ── Custom CSS (minimal overrides only) ────────────────────────────────────
CUSTOM_CSS = """
.main-header { text-align: center; padding: 1.5rem 0 0.5rem; }
.main-header h1 {
    font-size: 1.85rem; font-weight: 700; margin-bottom: 0.25rem;
}
.main-header p { font-size: 0.95rem; opacity: 0.7; }
.review-warning {
    background: #FFF1F2; border: 1.5px solid #F43F5E; border-radius: 10px;
    padding: 1rem; margin-top: 0.75rem;
}
.review-warning, .review-warning * {
    color: #1E293B !important; font-weight: 500;
}
footer { display: none !important; }
"""

with gr.Blocks(title="Melanoma Detection") as demo:
    # ── Header ─────────────────────────────────────────────────────────────
    gr.HTML("""<div class="main-header">
        <h1>Melanoma Detection AI</h1>
        <p>AI-assisted dermatoscopic image analysis &mdash; For research use only</p>
    </div>""")

    with gr.Tabs():
        # ═══════════════════════════════════════════════════════════════════
        # Tab 1 — Predict
        # ═══════════════════════════════════════════════════════════════════
        with gr.TabItem("Predict"):
            with gr.Row():
                with gr.Column(scale=1):
                    predict_image = gr.Image(type="pil", label="Upload Dermoscopic Image", height=350)
                    predict_btn = gr.Button("Analyze", variant="primary", size="lg")

                with gr.Column(scale=1):
                    gauge_chart = gr.Plot(show_label=False)

            result_header = gr.Markdown("")
            result_details = gr.Markdown("")
            review_warning = gr.Markdown(visible=False, elem_classes=["review-warning"])

        # ═══════════════════════════════════════════════════════════════════
        # Tab 2 — Explain
        # ═══════════════════════════════════════════════════════════════════
        with gr.TabItem("Explain"):
            with gr.Row():
                with gr.Column(scale=1):
                    explain_image = gr.Image(type="pil", label="Upload Dermoscopic Image", height=350)
                    explain_btn = gr.Button("Generate Heatmap", variant="primary")

                with gr.Column(scale=1):
                    gradcam_output = gr.Image(label="Grad‑CAM Heatmap", height=350)
            explain_text = gr.Markdown("")

        # ═══════════════════════════════════════════════════════════════════
        # Tab 3 — Model Details
        # ═══════════════════════════════════════════════════════════════════
        with gr.TabItem("Model Details"):
            gr.Markdown("""
            ## Model Performance

            | Metric | Value |
            |---|---|
            | **ROC-AUC** | 0.91 |
            | **Sensitivity** (melanoma detection) | 80.6% (at clinical threshold 0.15) |
            | **Specificity** | 83.7% |
            | **Architecture** | EfficientNet‑B0 |
            | **Inference engine** | ONNX Runtime (FP32) |
            | **Model size** | 16.6 MB |
            | **Latency** (batch-1, CPU) | ~18 ms |
            | **Throughput** | ~55 FPS |

            ### Model Comparison

            | Model | Params | AUC | ONNX Size | Latency (p99) |
            |---|---|---|---|---|
            | **EfficientNet‑B0** | 4.3M | **0.91** | 16.6 MB (FP32) | ~18 ms |
            | MobileNetV3‑Small | 1.5M | 0.84 | 3.2 MB (FP32) | ~8 ms |
            | ResNet50 | 24.0M | 0.84 | 94 MB (FP32) | ~35 ms |

            ### INT8 Quantization Note

            EfficientNet‑B0 is not compatible with INT8 quantization at acceptable accuracy.
            All five quantization strategies tested produced **10–15% AUC drops** (from 0.91
            down to 0.76–0.81). This is caused by the SiLU activation function and
            depthwise separable convolutions, which are hostile to integer quantization.

            **Recommendation:** Deploy FP32 ONNX (16.6 MB). This size is practical for edge
            devices and preserves full diagnostic accuracy.

            ### Preprocessing Pipeline

            1. **Hair removal** — DullRazor algorithm removes occluding hair
            2. **Colour normalisation** — Reinhard stain normalisation for consistent colour
            3. **Augmentation** — Random flips, rotations, brightness/contrast jitter
            4. **Resize** — 224×224 pixels with Lanczos interpolation
            5. **Normalise** — ImageNet mean/std scaling
            """)

        # ═══════════════════════════════════════════════════════════════════
        # Tab 4 — About
        # ═══════════════════════════════════════════════════════════════════
        with gr.TabItem("About"):
            gr.Markdown("""
            ## About This Project

            This application demonstrates an end-to-end MLOps pipeline for melanoma
            detection from dermoscopic images. It was developed as a final-year
            engineering project at ENSIAS (École Nationale Supérieure d'Informatique
            et d'Analyse des Systèmes), Morocco.

            ### Technical Architecture

            **EfficientNet‑B0** (PyTorch) → **ONNX Runtime** (inference) → **FastAPI**
            (backend) → **Gradio** (frontend)

            The full pipeline includes data validation (Great Expectations), experiment
            tracking (MLflow), data versioning (DVC), patient-aware train/val/test splits,
            and comprehensive monitoring (Prometheus metrics, drift detection).

            ### Limitations

            - Trained exclusively on the HAM10000 dataset, which is biased toward
              lighter skin tones and European populations.
            - Not validated for clinical use — this is a research prototype.
            - Performance may degrade on images acquired under different conditions
              (lighting, camera, magnification).
            - Predictions near the decision boundary are flagged for review but
              should not replace professional judgement.

            ### Resources

            - **API documentation:** [selba-melanoma-api.hf.space/docs](https://selba-melanoma-api.hf.space/docs)
            """)

    # ── Event Wiring ───────────────────────────────────────────────────────
    predict_btn.click(
        fn=handle_predict,
        inputs=[predict_image],
        outputs=[result_header, result_details, gauge_chart, review_warning],
    )
    explain_btn.click(
        fn=handle_explain,
        inputs=[explain_image],
        outputs=[gradcam_output, explain_text],
    )

# ── Entrypoint ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, css=CUSTOM_CSS, theme=_theme)
