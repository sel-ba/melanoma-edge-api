from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def build_image_index(raw_dir: Path) -> dict[str, Path]:
    image_dirs = sorted(
        path for path in raw_dir.iterdir()
        if path.is_dir() and path.name.startswith("HAM10000_images_part_")
    )
    index: dict[str, Path] = {}
    for image_dir in image_dirs:
        for image_path in image_dir.glob("*.jpg"):
            index[image_path.stem] = image_path
    return index


class Ham10000Dataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_index: dict[str, Path],
        transforms: Callable | None = None,
        image_id_col: str = "image_id",
        label_col: str = "label",
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_index = image_index
        self.transforms = transforms
        self.image_id_col = image_id_col
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.dataframe.iloc[idx]
        image_id = row[self.image_id_col]
        image_path = self.image_index.get(image_id)
        if image_path is None:
            raise KeyError(f"Image id not found in index: {image_id}")

        image = np.array(Image.open(image_path).convert("RGB"))
        if self.transforms is not None:
            transformed = self.transforms(image=image)
            image = transformed["image"]

        label = int(row[self.label_col])
        return {"image": image, "label": label, "image_id": image_id}
