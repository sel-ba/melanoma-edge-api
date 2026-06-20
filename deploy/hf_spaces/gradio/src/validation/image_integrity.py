from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image


def check_single_image(image_path: str) -> tuple[str, bool, str]:
    """Return the path, validity flag, and error message for one image."""
    try:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            if image.mode not in {"RGB", "RGBA"}:
                return image_path, False, f"Unexpected mode: {image.mode}"
            width, height = image.size
            if width < 100 or height < 100:
                return image_path, False, f"Image too small: {width}x{height}"
        return image_path, True, ""
    except Exception as exc:
        return image_path, False, str(exc)


def validate_image_directory(image_dir: str, max_workers: int = 8) -> dict[str, Any]:
    """Validate all JPG and PNG images under a directory."""
    image_paths = list(Path(image_dir).rglob("*.jpg")) + list(Path(image_dir).rglob("*.png"))

    corrupt: list[dict[str, str]] = []
    valid = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(check_single_image, str(path)): path for path in image_paths}
        for future in as_completed(futures):
            path, is_valid, error = future.result()
            if is_valid:
                valid += 1
            else:
                corrupt.append({"path": path, "error": error})

    total = len(image_paths)
    return {
        "total": total,
        "valid": valid,
        "corrupt": len(corrupt),
        "corrupt_paths": corrupt,
        "validity_rate": valid / total if total else 0,
    }
