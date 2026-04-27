
from __future__ import annotations

from torch import nn
from peft import LoraConfig, get_peft_model


def apply_lora(
    model: nn.Module,
    target_modules: list[str],
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> nn.Module:
    """
    Wrap the named Linear layers in `model` with LoRA using the PEFT library.

    Returns a PeftModel. Only lora_A / lora_B parameters are trainable.
    The wrapped layers expose:
        layer.lora_A['default'].weight   — A matrix  (in  → rank)
        layer.lora_B['default'].weight   — B matrix  (rank → out)
        layer.scaling['default']         — alpha / rank
        layer.base_layer                 — original frozen Linear
    """
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )
    return get_peft_model(model, config)
