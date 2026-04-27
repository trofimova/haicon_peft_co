
from __future__ import annotations

import torch
from torch import nn


class VisualPrompt(nn.Module):
    """
    Minimal visual prompt module.
    Prepends/perturbs a learnable patch-like tensor in image space.

    This is intentionally simple for teaching, not a full VPT implementation.
    """
    def __init__(self, channels: int = 3, prompt_size: int = 16):
        super().__init__()
        self.prompt = nn.Parameter(torch.zeros(1, channels, prompt_size, prompt_size))
        nn.init.normal_(self.prompt, std=0.02)

    def forward(self, x):
        b, c, h, w = x.shape
        ph, pw = self.prompt.shape[-2:]
        x = x.clone()
        x[:, :, :ph, :pw] = x[:, :, :ph, :pw] + self.prompt
        return x


class PromptTunedClassifier(nn.Module):
    """
    Frozen backbone with a learnable image-space prompt and linear head.
    """
    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int, prompt_size: int = 16):
        super().__init__()
        self.prompt = VisualPrompt(prompt_size=prompt_size)
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = self.prompt(x)
        feats = self.backbone(x)
        return self.head(feats)
