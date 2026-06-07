from __future__ import annotations

import numpy as np

from src.inference.preprocessor import ImagePreprocessor


class TestImagePreprocessor:
    """Tests for the ONNX inference preprocessor."""

    def test_output_shape(self, test_image_bytes: bytes) -> None:
        preprocessor = ImagePreprocessor(input_size=224)
        result = preprocessor.process_bytes(test_image_bytes)
        assert result["tensor"].shape == (1, 3, 224, 224)

    def test_output_dtype(self, test_image_bytes: bytes) -> None:
        preprocessor = ImagePreprocessor(input_size=224)
        result = preprocessor.process_bytes(test_image_bytes)
        assert result["tensor"].dtype == np.float32

    def test_small_image_resized(self, small_image_bytes: bytes) -> None:
        preprocessor = ImagePreprocessor(input_size=224)
        result = preprocessor.process_bytes(small_image_bytes)
        assert result["tensor"].shape == (1, 3, 224, 224)

    def test_large_image_resized(self, large_image_bytes: bytes) -> None:
        preprocessor = ImagePreprocessor(input_size=224)
        result = preprocessor.process_bytes(large_image_bytes)
        assert result["tensor"].shape == (1, 3, 224, 224)

    def test_values_are_normalized(self, test_image_bytes: bytes) -> None:
        """Output values should be normalized (not raw 0-255)."""
        preprocessor = ImagePreprocessor(input_size=224)
        result = preprocessor.process_bytes(test_image_bytes)
        tensor = result["tensor"]
        # After ImageNet normalization, values range roughly from -3 to +3
        assert tensor.max() < 10.0
        assert tensor.min() > -10.0
        # Raw pixels would be in 0-255 range
        assert not (tensor.min() >= 0.0 and tensor.max() > 1.5)
