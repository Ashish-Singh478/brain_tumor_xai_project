"""Transfer-learning model factory for the six architectures in the paper."""

from __future__ import annotations

from typing import Final

import timm
from torch import Tensor, nn


MODEL_NAMES: Final = {
    "vgg19": "vgg19",
    "resnet50": "resnet50",
    "xception": "xception71",
    "mobilenetv2": "mobilenetv2_100",
    "inceptionv3": "inception_v3",
    "nasnetlarge": "nasnetalarge",
}


class BrainTumorClassifier(nn.Module):
    """ImageNet backbone followed by the paper's custom classification head."""

    def __init__(
        self,
        architecture: str,
        num_classes: int = 4,
        dropout: float = 0.4,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        key = architecture.lower()
        if key not in MODEL_NAMES:
            choices = ", ".join(sorted(MODEL_NAMES))
            raise ValueError(f"Unknown architecture '{architecture}'. Choose: {choices}")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the range [0, 1).")

        self.architecture = key
        self.backbone = timm.create_model(
            MODEL_NAMES[key],
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feature_count = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_count, num_classes),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.backbone(inputs))


def create_model(
    architecture: str,
    num_classes: int = 4,
    dropout: float = 0.4,
    pretrained: bool = True,
) -> BrainTumorClassifier:
    return BrainTumorClassifier(
        architecture=architecture,
        num_classes=num_classes,
        dropout=dropout,
        pretrained=pretrained,
    )
