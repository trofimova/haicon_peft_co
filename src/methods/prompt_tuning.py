from __future__ import annotations

import torch
from torch import nn


class VisualPrompt(nn.Module):
    """
    Token-space visual prompt — shallow VPT (Jia et al. 2022).

    Maintains k learnable vectors in embedding space that are inserted between
    the CLS token and the first patch token before the transformer encoder runs.
    The image pixels and all ViT weights are completely unchanged.

    Sequence layout after insertion:
        [CLS] [p_1] ... [p_k] [patch_1] ... [patch_N]
    """
    def __init__(self, num_tokens: int = 10, feature_dim: int = 384):
        super().__init__()
        self.num_tokens = num_tokens
        self.tokens = nn.Parameter(torch.zeros(1, num_tokens, feature_dim))
        nn.init.normal_(self.tokens, std=0.02)

    def insert(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Insert prompt tokens between CLS and patch tokens."""
        B = embeddings.size(0)
        prompts = self.tokens.to(embeddings.device).expand(B, -1, -1)
        return torch.cat([embeddings[:, :1], prompts, embeddings[:, 1:]], dim=1)


class PromptTunedClassifier(nn.Module):
    """
    Frozen HF ViT with shallow token-space prompt and a linear head.

    Forward pass:
        1. ViT embedding layer  →  [CLS, patch_1 … patch_N]
        2. Insert k prompt tokens  →  [CLS, p_1 … p_k, patch_1 … patch_N]
        3. Frozen transformer encoder
        4. LayerNorm + extract CLS
        5. Linear head

    Requires the backbone to expose a `.vit` attribute (HFViTBackbone).
    """
    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int,
                 num_prompt_tokens: int = 10):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.prompt = VisualPrompt(num_tokens=num_prompt_tokens,
                                   feature_dim=feature_dim)
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vit = self.backbone.vit

        embeddings = vit.embeddings(x)              # [B, 1+N, D]
        embeddings = self.prompt.insert(embeddings) # [B, 1+k+N, D]
        hidden = embeddings
        for layer in vit.layers:
            out = layer(hidden)
            hidden = out[0] if isinstance(out, tuple) else out
        hidden = vit.layernorm(hidden)
        return self.head(hidden[:, 0])              # CLS token → logits
