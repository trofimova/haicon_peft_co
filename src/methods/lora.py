
from __future__ import annotations

from torch import nn


class LoRAClassifier(nn.Module):
    """
    Frozen backbone with LoRA applied to specified linear layers via the PEFT library,
    plus a trainable classification head.

    Parameters
    ----------
    target_modules : list[str]
        Names (or name suffixes) of the Linear layers inside the backbone to wrap with LoRA.
        For a ViT with TransformerEncoderLayer, use e.g. ``["linear1", "linear2"]``
        to target the FFN projections, or ``["out_proj"]`` for the attention output.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int,
        target_modules: list[str] | None = None,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if target_modules is None:
            target_modules = ["linear1", "linear2"]
        for p in backbone.parameters():
            p.requires_grad = False
        self.backbone = apply_lora(backbone, target_modules, rank, alpha, dropout)
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))


def apply_lora(
    model: nn.Module,
    target_modules: list[str],
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> nn.Module:
    """
    Wrap named Linear layers in `model` with LoRA via the PEFT library.

    Returns a PeftModel. Only lora_A / lora_B parameters are trainable.
    Requires ``peft`` to be installed (``pip install peft``).
    """
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )
    return get_peft_model(model, config)
