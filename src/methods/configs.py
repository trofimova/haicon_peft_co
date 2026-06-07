from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class LinearProbeConfig:
    pass


@dataclass
class BitFitConfig:
    pass


@dataclass
class VisualPromptConfig:
    num_prompt_tokens: int = 10


@dataclass
class LoRAConfig:
    rank: int = 8
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class AdapterConfig:
    bottleneck_dim: int = 32


@dataclass
class PartialFTConfig:
    n_blocks: int = 1   # how many last transformer blocks to unfreeze


CONFIGS: dict = {
    "linear_probe": LinearProbeConfig(),
    "bitfit":       BitFitConfig(),
    "visual_prompt": VisualPromptConfig(),
    "lora":         LoRAConfig(),
    "adapter":      AdapterConfig(),
    "partial_ft":   PartialFTConfig(),
}
