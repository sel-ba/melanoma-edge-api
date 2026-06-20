from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def remove_hair_dullrazor(image: np.ndarray) -> np.ndarray:
    """Remove dermoscopic hair artifacts using a DullRazor-style pipeline."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    hair_mask = cv2.dilate(hair_mask, kernel, iterations=1)
    return cv2.inpaint(image, hair_mask, inpaintRadius=6, flags=cv2.INPAINT_TELEA)


def batch_remove_hair(
    image_paths: list[Path | str],
    output_dir: str | Path,
    skip_if_low_hair: bool = True,
    hair_threshold: float = 0.02,
) -> None:
    """Process a batch of images and save the cleaned copies."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        image_path = Path(image_path)
        image = np.array(Image.open(image_path).convert("RGB"))

        if skip_if_low_hair:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17))
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
            hair_density = mask.sum() / (255 * mask.size)
            if hair_density < hair_threshold:
                Image.fromarray(image).save(output_path / image_path.name)
                continue

        cleaned = remove_hair_dullrazor(image)
        Image.fromarray(cleaned).save(output_path / image_path.name)
