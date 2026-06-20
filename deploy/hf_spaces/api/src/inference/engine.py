from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml

CLASS_NAMES = {0: "benign", 1: "melanoma"}


class ONNXInferenceEngine:
    """ONNX Runtime inference engine with threshold checks."""

    def __init__(self, model_path: str, threshold_config_path: str) -> None:
        self.model_path = Path(model_path)
        self._load_thresholds(threshold_config_path)
        self._load_session()

    def _load_session(self) -> None:
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session_options.intra_op_num_threads = 4
        session_options.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _load_thresholds(self, config_path: str) -> None:
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        self.melanoma_threshold = float(config.get("melanoma_threshold", 0.5))
        self.decision_boundary_margin = float(config.get("decision_boundary_margin", 0.15))
        self.uncertainty_threshold = float(config.get("uncertainty_threshold", 0.2))
        self.low_confidence_threshold = float(config.get("low_confidence_threshold", 0.15))

    def predict(self, preprocessed_image: np.ndarray) -> dict:
        logits = self.session.run(
            [self.output_name],
            {self.input_name: preprocessed_image},
        )[0]
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs = probs[0]

        melanoma_prob = float(probs[1])
        predicted_class_idx = int(probs.argmax())
        predicted_class = CLASS_NAMES[predicted_class_idx]

        return {
            "probs": probs,
            "melanoma_probability": melanoma_prob,
            "predicted_class": predicted_class,
            "predicted_class_idx": predicted_class_idx,
        }

    def is_high_confidence(self, melanoma_prob: float, uncertainty: float) -> tuple[bool, str | None]:
        if abs(melanoma_prob - self.melanoma_threshold) < self.decision_boundary_margin:
            return True, "prediction_near_decision_boundary"
        if uncertainty > self.uncertainty_threshold:
            return True, "high_epistemic_uncertainty"
        if melanoma_prob < self.low_confidence_threshold:
            return False, None
        return False, None
