from __future__ import annotations

from torch import nn


class BitFitClassifier(nn.Module):
    """
    Frozen backbone where only bias parameters remain trainable, plus a linear head.

    For a ViT backbone this typically unfreezes LayerNorm biases and attention / MLP
    biases — a tiny fraction of the total parameter count.
    """

    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        for name, p in self.backbone.named_parameters():
            if "bias" in name:
                p.requires_grad = True
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))
