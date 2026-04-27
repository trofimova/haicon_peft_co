
from __future__ import annotations

from torch import nn


class BottleneckAdapter(nn.Module):
    def __init__(self, dim: int, bottleneck_dim: int = 32):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, dim)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))


class AdapterHeadClassifier(nn.Module):
    """
    Teaching-friendly stand-in for hidden-state adapters:
    frozen feature extractor -> trainable adapter -> classifier
    """
    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int, bottleneck_dim: int = 32):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.adapter = BottleneckAdapter(feature_dim, bottleneck_dim=bottleneck_dim)
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.adapter(feats)
        return self.head(feats)
