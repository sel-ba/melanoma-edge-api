from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import imagehash
from PIL import Image


def find_visual_duplicates(image_dir: str, hash_size: int = 16, threshold: int = 5) -> dict[str, list[str]]:
    """Find visually similar images using perceptual hashes."""
    image_paths = list(Path(image_dir).rglob("*.jpg"))
    hashes: dict[str, imagehash.ImageHash] = {}

    for path in image_paths:
        try:
            with Image.open(path) as image:
                hashes[str(path)] = imagehash.phash(image, hash_size=hash_size)
        except Exception:
            continue

    duplicate_groups: dict[str, list[str]] = defaultdict(list)
    processed: set[str] = set()
    paths_list = list(hashes.keys())

    for index, path_a in enumerate(paths_list):
        if path_a in processed:
            continue
        group = [path_a]
        for path_b in paths_list[index + 1 :]:
            if path_b in processed:
                continue
            if hashes[path_a] - hashes[path_b] <= threshold:
                group.append(path_b)
                processed.add(path_b)
        if len(group) > 1:
            duplicate_groups[path_a] = group
        processed.add(path_a)

    return dict(duplicate_groups)
