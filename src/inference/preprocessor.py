from __future__ import annotations

from io import BytesIO
from typing import Iterable

import numpy as np
from PIL import Image


class ImagePreprocessor:
    """Preprocess input images for ONNX inference."""

    def __init__(
        self,
        input_size: int = 224,
        mean: Iterable[float] | None = None,
        std: Iterable[float] | None = None,
    ) -> None:
        self.input_size = input_size
        self.mean = np.array(list(mean or [0.485, 0.456, 0.406]), dtype=np.float32)
        self.std = np.array(list(std or [0.229, 0.224, 0.225]), dtype=np.float32)

    def process_bytes(self, image_bytes: bytes) -> dict:
        image = self._load_image(image_bytes)
        resized = image.resize((self.input_size, self.input_size), resample=Image.LANCZOS)
        array = np.asarray(resized).astype(np.float32) / 255.0
        array = (array - self.mean) / self.std
        array = np.transpose(array, (2, 0, 1))
        array = np.expand_dims(array, 0).astype(np.float32)
        return {"tensor": array}

    @staticmethod
    def _load_image(image_bytes: bytes) -> Image.Image:
        with Image.open(BytesIO(image_bytes)) as image:
            return image.convert("RGB")
