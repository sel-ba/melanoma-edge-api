from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

logger = logging.getLogger(__name__)


class MelanomaTrainer:
    """Training orchestrator with MLflow experiment tracking."""

    def __init__(self, config: dict) -> None:
        self.config = self._resolve_config(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self.device)

    def train(
        self,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        criterion,
        run_name: str | None = None,
    ):
        # Use local file-based tracking (consistent for CI and local dev)
        mlflow.set_tracking_uri("file:./mlflow")
        mlflow.set_experiment(self.config["experiment_name"])

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(self._flatten_dict(self.config))
            mlflow.log_param("device", str(self.device))
            mlflow.log_param("model_name", model.model_name)
            mlflow.log_param("total_params", sum(p.numel() for p in model.parameters()))
            mlflow.log_param(
                "trainable_params",
                sum(p.numel() for p in model.parameters() if p.requires_grad),
            )

            model = model.to(self.device)
            best_val_auc = 0.0
            patience_counter = 0

            logger.info("Phase 1: Training classification head (backbone frozen)")
            model.freeze_backbone()
            optimizer = self._get_optimizer(model, phase=1)
            scheduler = self._get_scheduler(optimizer, phase=1)

            for epoch in range(self.config["phase1_epochs"]):
                train_metrics = self._run_epoch(
                    model, train_loader, criterion, optimizer, training=True
                )
                val_metrics = self._run_epoch(
                    model, val_loader, criterion, optimizer=None, training=False
                )

                scheduler.step()
                self._log_epoch_metrics(epoch, train_metrics, val_metrics, prefix="p1")

                logger.info(
                    "Phase1 Epoch %s: train_loss=%.4f, val_auc=%.4f",
                    epoch + 1,
                    train_metrics["loss"],
                    val_metrics["auc"],
                )

            logger.info("Phase 2: Fine-tuning full model (backbone unfrozen)")
            model.unfreeze_backbone()
            optimizer = self._get_optimizer(model, phase=2)
            scheduler = self._get_scheduler(optimizer, phase=2)

            for epoch in range(self.config["phase2_epochs"]):
                train_metrics = self._run_epoch(
                    model, train_loader, criterion, optimizer, training=True
                )
                val_metrics = self._run_epoch(
                    model, val_loader, criterion, optimizer=None, training=False
                )

                scheduler.step()
                self._log_epoch_metrics(
                    self.config["phase1_epochs"] + epoch,
                    train_metrics,
                    val_metrics,
                    prefix="p2",
                )

                if val_metrics["auc"] > best_val_auc:
                    best_val_auc = val_metrics["auc"]
                    patience_counter = 0
                    checkpoint_path = Path("models/checkpoints/best_model.pt")
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_auc": best_val_auc,
                            "config": self.config,
                            "run_id": run.info.run_id,
                        },
                        checkpoint_path,
                    )
                    mlflow.log_artifact(str(checkpoint_path))
                    logger.info("New best val_auc: %.4f", best_val_auc)
                else:
                    patience_counter += 1

                if patience_counter >= self.config["patience"]:
                    logger.info("Early stopping triggered at epoch %s", epoch + 1)
                    break

            mlflow.log_metric("best_val_auc", best_val_auc)
            mlflow.pytorch.log_model(model, "pytorch_model")

            return run.info.run_id

    def _run_epoch(self, model, loader, criterion, optimizer, training: bool) -> dict:
        model.train(training)
        total_loss = 0.0
        all_labels = []
        all_probs = []

        with torch.set_grad_enabled(training):
            for batch in loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = model(images)
                loss = criterion(logits, labels)

                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                total_loss += loss.item()
                probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy())

        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_preds = (all_probs >= 0.5).astype(int)

        metrics = {
            "loss": total_loss / len(loader),
            "auc": roc_auc_score(all_labels, all_probs),
            "f1": f1_score(all_labels, all_preds, zero_division=0),
            "sensitivity": self._sensitivity(all_labels, all_preds),
            "specificity": self._specificity(all_labels, all_preds),
        }
        return metrics

    def _sensitivity(self, y_true, y_pred) -> float:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return tp / (tp + fn + 1e-8)

    def _specificity(self, y_true, y_pred) -> float:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return tn / (tn + fp + 1e-8)

    def _log_epoch_metrics(self, epoch, train_m, val_m, prefix: str) -> None:
        for key, value in train_m.items():
            mlflow.log_metric(f"{prefix}_train_{key}", value, step=epoch)
        for key, value in val_m.items():
            mlflow.log_metric(f"{prefix}_val_{key}", value, step=epoch)

    def _get_optimizer(self, model, phase: int):
        if phase == 1:
            return torch.optim.AdamW(
                model.classifier.parameters(),
                lr=self.config["phase1_lr"],
                weight_decay=self.config["weight_decay"],
            )
        return torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": self.config["phase2_backbone_lr"]},
                {"params": model.classifier.parameters(), "lr": self.config["phase2_head_lr"]},
            ],
            weight_decay=self.config["weight_decay"],
        )

    def _get_scheduler(self, optimizer, phase: int):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config[f"phase{phase}_epochs"],
            eta_min=1e-6,
        )

    def _flatten_dict(self, data, parent_key: str = "", sep: str = ".") -> dict:
        items = []
        for key, value in data.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, dict):
                items.extend(self._flatten_dict(value, new_key, sep=sep).items())
            else:
                items.append((new_key, value))
        return dict(items)

    def _resolve_config(self, config: dict) -> dict:
        training = config.get("training", {})
        resolved = {**config, **training}
        resolved.setdefault("model", config.get("model", {}))
        resolved.setdefault("data", config.get("data", {}))
        resolved.setdefault("loss", config.get("loss", {}))
        resolved.setdefault("training", training)
        return resolved
