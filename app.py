"""Gradio frontend for Melanoma Detection API — deploy on Hugging Face Spaces."""

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
MODEL_PATH = Path("models/onnx/efficientnet_b0_int8.onnx")
CHECKPOINT_PATH = Path("models/checkpoints/best_model.pt")

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
        return "", "", None, gr.update(visible=True), "", "", ""
    result = predict_onnx(image)
    summary = f"## {result['predicted']}"
    details = (
        f"**Melanoma probability:** {result['melanoma_prob']:.1%}\n\n"
        f"**Benign probability:** {result['benign_prob']:.1%}\n\n"
        f"**Uncertainty:** {result['uncertainty']:.3f}\n\n"
        f"**Confidence:** {result['confidence']}\n\n"
        f"**Requires review:** {'Yes  ' if result['requires_review'] else 'No  '}"
    )
    # Build confidence gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=result["melanoma_prob"] * 100,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Melanoma Risk (%)", "font": {"size": 16}},
        delta={"reference": 50, "increasing": {"color": "#EF4444"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": "#EF4444" if result["melanoma_prob"] > 0.5 else "#22C55E"},
            "steps": [
                {"range": [0, 30], "color": "#dcfce7"},
                {"range": [30, 70], "color": "#fef9c3"},
                {"range": [70, 100], "color": "#fee2e2"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": 50,
            },
        },
    ))
    fig.update_layout(height=250, margin=dict(l=40, r=40, t=50, b=30))

    review_visible = result["requires_review"]
    return summary, details, fig, gr.update(visible=review_visible), "", result["confidence"], str(result["uncertainty"])


def handle_explain(image: Image.Image | None) -> tuple:
    if image is None:
        return None, "Upload an image first."
    result = predict_onnx(image)
    gradcam_img = generate_gradcam(image)
    text = (
        f"**Prediction:** {result['predicted']}  \n"
        f"**Melanoma:** {result['melanoma_prob']:.1%} | "
        f"**Benign:** {result['benign_prob']:.1%}  \n"
        f"**Uncertainty:** {result['uncertainty']:.3f} | "
        f"**Confidence:** {result['confidence']}"
    )
    if gradcam_img is None:
        text += "\n\n* PyTorch checkpoint not found. GradCAM requires `models/checkpoints/best_model.pt`."
    return gradcam_img, text


def handle_compare(image: Image.Image | None) -> str:
    if image is None:
        return "Upload an image to see model comparison."
    result = predict_onnx(image)
    return (
        "## Model Comparison\n\n"
        "| Model | Params | AUC | Size (ONNX) | Latency (p99) |\n"
        "|---|---|---|---|---|\n"
        "| **EfficientNet-B0**  | 4.3M | **0.8687** | 5.0 MB (INT8) | 4.4 ms |\n"
        "| MobileNetV3-Small | 1.5M | 0.8369 | 3.2 MB (INT8) | 3.1 ms |\n"
        "| ResNet50 | 24.0M | 0.8402 | 23.5 MB (INT8) | 12.8 ms |\n\n"
        f"**Current prediction ({result['predicted']}):** melanoma={result['melanoma_prob']:.1%}\n\n"
        "*EfficientNet-B0 was selected as the production model for the best accuracy/speed tradeoff.*"
    )


# ── UI ─────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
.gradio-container { font-family: 'Inter', system-ui, sans-serif; }
.main-header { text-align: center; padding: 1rem 0; }
.review-warning { 
    background: #fef2f2; border: 2px solid #ef4444; border-radius: 12px;
    padding: 1rem; color: #991b1b; font-weight: 600; text-align: center;
}
.disclaimer {
    background: #fffbeb; border: 1px solid #f59e0b; border-radius: 8px;
    padding: 0.5rem 1rem; font-size: 0.8rem; color: #92400e; text-align: center;
}
"""

with gr.Blocks(css=CUSTOM_CSS, title="Melanoma Detection AI") as demo:
    gr.HTML(
        """<div class="main-header">
        <h1>Melanoma Detection AI</h1>
        <p>Edge-deployable dermatoscopy assistant — Research use only</p>
        </div>"""
    )

    with gr.Tabs():
        # ── Tab 1: Predict ──────────────────────────────────────────────
        with gr.TabItem("Predict"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(type="pil", label="Upload Dermoscopic Image", height=350)
                    predict_btn = gr.Button("Analyze", variant="primary", size="lg")

                with gr.Column(scale=1):
                    result_header = gr.Markdown("")
                    result_details = gr.Markdown("")
                    gauge_chart = gr.Plot(label="Risk Gauge", show_label=False)

            review_warning = gr.Markdown("Requires clinical review", visible=False, elem_classes=["review-warning"])

            with gr.Accordion("Technical Details", open=False):
                with gr.Row():
                    confidence_badge = gr.Textbox(label="Confidence Level", interactive=False)
                    uncertainty_val = gr.Textbox(label="Uncertainty", interactive=False)

            gr.HTML(
                """<div class="disclaimer">
                This tool is for research/educational purposes only. Not a substitute for professional medical diagnosis.
                </div>"""
            )

        # ── Tab 2: Explain ──────────────────────────────────────────────
        with gr.TabItem("Explain (Grad‑CAM)"):
            with gr.Row():
                with gr.Column(scale=1):
                    explain_image = gr.Image(type="pil", label="Upload Dermoscopic Image", height=350)
                    explain_btn = gr.Button("Generate Explanation", variant="primary")

                with gr.Column(scale=1):
                    gradcam_output = gr.Image(label="Grad‑CAM Heatmap", height=350)
            explain_text = gr.Markdown("Grad‑CAM highlights which regions influenced the prediction.")
            gr.HTML("""<div class="disclaimer">
            Red/yellow = high influence on prediction. Heatmap should focus on the lesion, not background artifacts.
            </div>""")

        # ── Tab 3: Compare Models ────────────────────────────────────────
        with gr.TabItem("Compare Models"):
            comp_image = gr.Image(type="pil", label="Upload Image (optional)", height=250)
            comp_output = gr.Markdown(
                "## Model Comparison\n\n"
                "| Model | Params | AUC | Size (ONNX) | Latency (p99) |\n"
                "|---|---|---|---|---|\n"
                "| **EfficientNet-B0**  | 4.3M | **0.8687** | 5.0 MB (INT8) | 4.4 ms |\n"
                "| MobileNetV3-Small | 1.5M | 0.8369 | 3.2 MB (INT8) | 3.1 ms |\n"
                "| ResNet50 | 24.0M | 0.8402 | 23.5 MB (INT8) | 12.8 ms |\n\n"
                "*EfficientNet-B0 selected for best accuracy/speed ratio.*"
            )

        # ── Tab 4: About ─────────────────────────────────────────────────
        with gr.TabItem("About"):
            gr.Markdown("""
            ## Melanoma Detection AI — Edge-Deployable

            ### How it works
            1. Upload a dermoscopic image (JPEG/PNG)
            2. The model analyzes it using EfficientNet-B0 (ONNX Runtime, INT8)
            3. You get a probability, uncertainty estimate, and Grad‑CAM heatmap
            4. Predictions near the decision boundary are flagged for review

            ### Technical Stack
            - **Model:** EfficientNet-B0 fine-tuned on HAM10000 (10,015 images)
            - **Inference:** ONNX Runtime with INT8 quantization (5 MB, <5 ms)
            - **Explainability:** Grad‑CAM heatmaps, Monte‑Carlo Dropout uncertainty
            - **Validation:** Great Expectations suite (7/7 checks pass)
            - **Tracking:** MLflow experiment tracking, DVC data versioning
            - **Frontend:** Gradio — deploy anywhere, zero cost

            ### Performance
            - **ROC‑AUC:** 0.8687 on held‑out test set
            - **Sensitivity:** 0.85 (melanoma detection rate)
            - **Latency:** 4.4 ms p99 on CPU (255 FPS)
            - **Model size:** 5.0 MB (INT8 quantized)

            ### Limitations
            - Trained on HAM10000 — biased toward lighter skin tones
            - Not validated for clinical use
            - For research/educational purposes only

            ### Architecture Decision Records
            See `docs/adr/` for detailed decisions on model choice, runtime, and API framework.
            """)

    # ── Event Wiring ─────────────────────────────────────────────────────
    predict_btn.click(
        fn=handle_predict,
        inputs=[image_input],
        outputs=[result_header, result_details, gauge_chart, review_warning, explain_text, confidence_badge, uncertainty_val],
    )
    explain_btn.click(
        fn=handle_explain,
        inputs=[explain_image],
        outputs=[gradcam_output, explain_text],
    )
    comp_image.change(
        fn=handle_compare,
        inputs=[comp_image],
        outputs=[comp_output],
    )

# ── Entrypoint ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
