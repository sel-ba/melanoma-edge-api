"""Robust INT8 quantization comparison script.

Pre-loads calibration data as numpy arrays (no DataLoader workers, no hangs),
then tries multiple static-quantization strategies and evaluates each on the
test set. Writes a JSON summary to reports/artifacts/quantization_comparison.json.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import onnxruntime as ort
import pandas as pd
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quant_pre_process,
    quantize_static,
)
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score

PROJECT = Path(__file__).resolve().parents[1]
ONNX_DIR = PROJECT / "models" / "onnx"
FP32 = ONNX_DIR / "efficientnet_b0_fp32.onnx"
PRE = ONNX_DIR / "efficientnet_b0_fp32_preprocessed.onnx"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_tensor(image_path: Path, size: int = 224) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return np.transpose(arr, (2, 0, 1))[None].astype(np.float32)


def build_calibration_arrays(n: int = 300) -> list[np.ndarray]:
    """Pre-load n train images as raw numpy arrays — no DataLoader."""
    from src.data.dataset import build_image_index

    raw_dir = PROJECT / "data" / "raw"
    train_df = pd.read_csv(PROJECT / "data" / "splits" / "train_split.csv").head(n)
    index = build_image_index(raw_dir)
    tensors = []
    for image_id in train_df["image_id"]:
        p = index.get(image_id)
        if p and p.exists():
            tensors.append(load_tensor(p))
    print(f"Loaded {len(tensors)} calibration tensors")
    return tensors


class NumpyCalReader(CalibrationDataReader):
    def __init__(self, tensors: list[np.ndarray]):
        self.data = [{"input_image": t} for t in tensors]
        self.it = iter(self.data)

    def get_next(self):
        return next(self.it, None)

    def rewind(self):
        self.it = iter(self.data)


def build_test_arrays() -> tuple[np.ndarray, np.ndarray]:
    from src.data.dataset import build_image_index

    raw_dir = PROJECT / "data" / "raw"
    test_df = pd.read_csv(PROJECT / "data" / "splits" / "test_split.csv")
    index = build_image_index(raw_dir)
    X, y = [], []
    for _, row in test_df.iterrows():
        p = index.get(row["image_id"])
        if p and p.exists():
            X.append(load_tensor(p))
            y.append(int(row["dx"] == "mel"))
    return np.concatenate(X, axis=0), np.array(y)


def evaluate(path: Path, X: np.ndarray, y: np.ndarray) -> dict:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    # Batched eval
    probs = []
    bs = 64
    for i in range(0, len(X), bs):
        logits = sess.run(None, {iname: X[i : i + bs]})[0]
        exp = np.exp(logits - logits.max(1, keepdims=True))
        probs.extend((exp / exp.sum(1, keepdims=True))[:, 1])
    probs = np.array(probs)
    auc = roc_auc_score(y, probs)
    tn, fp, fn, tp = confusion_matrix(y, (probs >= 0.5).astype(int), labels=[0, 1]).ravel()
    return {
        "auc": round(float(auc), 4),
        "sensitivity": round(float(tp / (tp + fn)), 4),
        "specificity": round(float(tn / (tn + fp)), 4),
        "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
    }


def main() -> None:
    print("Step 1: pre-process FP32 model")
    if not PRE.exists():
        quant_pre_process(input_model_path=str(FP32), output_model_path=str(PRE))

    print("Step 2: build calibration data")
    cal_tensors = build_calibration_arrays(n=300)
    reader = NumpyCalReader(cal_tensors)

    variants = [
        ("qdq_pc_u8act_i8w", ONNX_DIR / "efficientnet_b0_int8_qdq_pc.onnx",
         dict(quant_format=QuantFormat.QDQ, per_channel=True,
              activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
              calibrate_method=CalibrationMethod.Entropy)),
        ("qdq_pc_u8act_u8w", ONNX_DIR / "efficientnet_b0_int8_qdq_pc_u8w.onnx",
         dict(quant_format=QuantFormat.QDQ, per_channel=True,
              activation_type=QuantType.QUInt8, weight_type=QuantType.QUInt8,
              calibrate_method=CalibrationMethod.Entropy)),
        ("qdq_pc_minmax", ONNX_DIR / "efficientnet_b0_int8_qdq_minmax.onnx",
         dict(quant_format=QuantFormat.QDQ, per_channel=True,
              activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
              calibrate_method=CalibrationMethod.MinMax)),
    ]

    print("Step 3: quantize variants")
    for name, out, kwargs in variants:
        if out.exists():
            print(f"  [{name}] exists, skipping")
            continue
        print(f"  [{name}] quantizing...")
        try:
            reader.rewind()
            quantize_static(str(PRE), str(out), reader, **kwargs)
            print(f"    -> {out.name} ({out.stat().st_size/1024/1024:.2f} MB)")
        except Exception as e:
            print(f"    FAILED: {e}")

    print("Step 4: build test arrays")
    X, y = build_test_arrays()
    print(f"  test set: {len(y)} images, melanoma rate={y.mean():.3f}")

    print("\nStep 5: evaluate")
    results = {}
    results["fp32"] = evaluate(FP32, X, y)
    print(f"  fp32               : {results['fp32']}")
    results["old_int8_broken"] = evaluate(ONNX_DIR / "efficientnet_b0_int8.onnx", X, y)
    print(f"  old_int8 (broken)  : {results['old_int8_broken']}")
    for name, out, _ in variants:
        if out.exists():
            results[name] = evaluate(out, X, y)
            print(f"  {name:20s}: {results[name]}")

    # Compute AUC drops
    fp_auc = results["fp32"]["auc"]
    for k, v in results.items():
        if isinstance(v, dict) and "auc" in v:
            v["auc_drop_vs_fp32"] = round(fp_auc - v["auc"], 4)

    out_json = PROJECT / "reports" / "artifacts" / "quantization_comparison.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_json}")

    # Pick best (lowest AUC drop among INT8 that load)
    best = None
    for k, v in results.items():
        if k.startswith("fp32") or "broken" in k:
            continue
        if isinstance(v, dict) and "auc_drop_vs_fp32" in v:
            if best is None or v["auc_drop_vs_fp32"] < best[1]:
                best = (k, v["auc_drop_vs_fp32"])
    if best:
        print(f"\nBest INT8 variant: {best[0]} (AUC drop {best[1]:.4f})")


if __name__ == "__main__":
    main()
