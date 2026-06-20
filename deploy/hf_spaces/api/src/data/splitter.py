from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def create_patient_aware_splits(
    metadata_path: str | Path,
    output_dir: str | Path,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> dict:
    """Create train/val/test splits with no lesion_id overlap."""
    metadata_path = Path(metadata_path)
    df = pd.read_csv(metadata_path)

    if "lesion_id" not in df.columns:
        raise ValueError("Expected lesion_id column in metadata.")
    if "dx" not in df.columns:
        raise ValueError("Expected dx column in metadata.")

    df = df.copy()
    df["label"] = (df["dx"] == "mel").astype(int)

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=test_ratio,
        random_state=random_seed,
    )
    trainval_idx, test_idx = next(gss.split(df, groups=df["lesion_id"]))

    df_trainval = df.iloc[trainval_idx].copy()
    df_test = df.iloc[test_idx].copy()

    val_adjusted = val_ratio / (1 - test_ratio)
    gss2 = GroupShuffleSplit(
        n_splits=1,
        test_size=val_adjusted,
        random_state=random_seed,
    )
    train_idx, val_idx = next(gss2.split(df_trainval, groups=df_trainval["lesion_id"]))

    df_train = df_trainval.iloc[train_idx].copy()
    df_val = df_trainval.iloc[val_idx].copy()

    train_lesions = set(df_train["lesion_id"])
    val_lesions = set(df_val["lesion_id"])
    test_lesions = set(df_test["lesion_id"])

    if train_lesions & val_lesions:
        raise RuntimeError("Leakage detected between train and val splits.")
    if train_lesions & test_lesions:
        raise RuntimeError("Leakage detected between train and test splits.")
    if val_lesions & test_lesions:
        raise RuntimeError("Leakage detected between val and test splits.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(output_dir / "train_split.csv", index=False)
    df_val.to_csv(output_dir / "val_split.csv", index=False)
    df_test.to_csv(output_dir / "test_split.csv", index=False)

    return {
        "train_size": len(df_train),
        "val_size": len(df_val),
        "test_size": len(df_test),
        "train_melanoma_rate": float(df_train["label"].mean()),
        "val_melanoma_rate": float(df_val["label"].mean()),
        "test_melanoma_rate": float(df_test["label"].mean()),
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    metadata = project_root / "data" / "raw" / "HAM10000_metadata.csv"
    splits_dir = project_root / "data" / "splits"

    stats = create_patient_aware_splits(metadata, splits_dir)
    print(stats)
