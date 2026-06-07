from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import Ham10000Dataset, build_image_index
from src.models.efficientnet import MelanomaClassifier
from src.models.mobilenetv3 import MobileNetV3Classifier
from src.models.resnet_baseline import ResNetBaseline
from src.preprocessing.augmentation import get_train_transforms, get_val_transforms
from src.preprocessing.class_balancer import get_class_weights, get_weighted_sampler
from src.training.loss import FocalLoss
from src.training.trainer import MelanomaTrainer

logging.basicConfig(level=logging.INFO)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def ensure_label(df: pd.DataFrame) -> pd.DataFrame:
    if "label" in df.columns:
        return df
    df = df.copy()
    df["label"] = (df["dx"] == "mel").astype(int)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train melanoma classifier with MLflow tracking.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/efficientnet_b0.yaml",
        help="Path to YAML training config.",
    )
    parser.add_argument("--run-name", type=str, default=None, help="MLflow run name.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / args.config
    config = load_config(config_path)

    set_seed(int(config.get("seed", 42)))

    data_root = project_root / "data"
    raw_dir = data_root / "raw"
    train_df = pd.read_csv(data_root / "splits" / "train_split.csv")
    val_df = pd.read_csv(data_root / "splits" / "val_split.csv")
    train_df = ensure_label(train_df)
    val_df = ensure_label(val_df)

    image_index = build_image_index(raw_dir)
    train_transforms = get_train_transforms()
    val_transforms = get_val_transforms()

    train_dataset = Ham10000Dataset(train_df, image_index, transforms=train_transforms)
    val_dataset = Ham10000Dataset(val_df, image_index, transforms=val_transforms)

    labels = train_df["label"].astype(int).tolist()
    loss_cfg = config.get("loss", {})
    weights = get_class_weights(
        labels,
        strategy=loss_cfg.get("class_weight_strategy", "effective_num"),
        beta=float(loss_cfg.get("beta", 0.999)),
    )
    sampler = get_weighted_sampler(labels, weights)

    data_cfg = config.get("data", {})
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(data_cfg.get("batch_size", 32)),
        sampler=sampler,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(data_cfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
    )

    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name", "efficientnet_b0")
    nc = int(model_cfg.get("num_classes", 2))
    pretrained = bool(model_cfg.get("pretrained", True))
    do_rate = float(model_cfg.get("dropout_rate", 0.3))

    if model_name.startswith("resnet"):
        model = ResNetBaseline(
            model_name=model_name, num_classes=nc,
            pretrained=pretrained, dropout_rate=do_rate,
        )
    elif model_name.startswith("mobilenet"):
        model = MobileNetV3Classifier(
            model_name=model_name, num_classes=nc,
            pretrained=pretrained, dropout_rate=do_rate,
        )
    else:
        model = MelanomaClassifier(
            model_name=model_name, num_classes=nc,
            pretrained=pretrained, dropout_rate=do_rate,
        )

    criterion = FocalLoss(
        gamma=float(loss_cfg.get("gamma", 2.0)),
        alpha=weights,
    )

    trainer = MelanomaTrainer(config)
    run_id = trainer.train(model, train_loader, val_loader, criterion, run_name=args.run_name)
    print(f"MLflow run id: {run_id}")


if __name__ == "__main__":
    main()
