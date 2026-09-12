import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)


class EmotionModel(nn.Module):

    def __init__(self, num_classes=6):
        super().__init__()

        self.backbone = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT
        )

        in_features = self.backbone.classifier[-1].in_features

        self.backbone.classifier[-1] = nn.Linear(
            in_features,
            num_classes
        )

    def forward(self, x):
        return self.backbone(x)