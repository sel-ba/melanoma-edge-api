from __future__ import annotations

import torch
import torch.nn as nn
import timm


class ResNetBaseline(nn.Module):
    """ResNet50 baseline model for comparison."""

    def __init__(
        self,
        model_name: str = "resnet50",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        feature_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, num_layers: int = -1) -> None:
        params = list(self.backbone.parameters())
        if num_layers == -1:
            for param in params:
                param.requires_grad = True
        else:
            for param in params[-num_layers:]:
                param.requires_grad = True
