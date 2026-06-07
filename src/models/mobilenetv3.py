from __future__ import annotations

import torch
import torch.nn as nn
import timm


class MobileNetV3Classifier(nn.Module):
    """MobileNetV3-Small fine-tuned for melanoma detection on edge devices.

    Optimized for resource-constrained environments with only 2.5M parameters.
    Best suited for Raspberry Pi 4 and mobile inference.
    """

    def __init__(
        self,
        model_name: str = "mobilenetv3_small_100",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
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

        # timm reports num_features=576 but forward pass outputs 1024
        # for mobilenetv3_small_100 (due to internal conv_head).
        # Infer the real feature dimension with a dry run.
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feature_dim = self.backbone(dummy).shape[1]

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(128, num_classes),
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
