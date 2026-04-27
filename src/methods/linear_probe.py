
from torch import nn


class LinearProbeModel(nn.Module):
    """
    Generic wrapper: frozen backbone + trainable linear head.
    Assumes backbone(x) -> feature tensor of shape [B, D].
    """
    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)
