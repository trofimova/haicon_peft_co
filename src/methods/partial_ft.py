from __future__ import annotations

from typing import List

from torch import nn


class PartialFineTuneClassifier(nn.Module):
    """
    Frozen backbone with selected submodules unfrozen, plus a trainable linear head.

    Parameters
    ----------
    modules_to_unfreeze : list of nn.Module
        Submodules of ``backbone`` to make trainable after the initial freeze.
        Pass references to the actual submodule objects, e.g.::

            last_block = list(backbone.transformer.layers)[-1]
            PartialFineTuneClassifier(backbone, D, K, modules_to_unfreeze=[last_block])
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int,
        modules_to_unfreeze: List[nn.Module] | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        if modules_to_unfreeze:
            for m in modules_to_unfreeze:
                for p in m.parameters():
                    p.requires_grad = True
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))
