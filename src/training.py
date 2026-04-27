
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch
from torch import nn
from tqdm.auto import tqdm


@dataclass
class History:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str = "cpu") -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
    }


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    epochs: int = 5,
    device: str = "cpu",
) -> History:
    criterion = nn.CrossEntropyLoss()
    history = History()
    model.to(device)

    for _ in tqdm(range(epochs), desc="training"):
        model.train()
        running_loss = 0.0
        n = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            n += x.size(0)

        history.train_loss.append(running_loss / max(n, 1))
        val_metrics = evaluate(model, val_loader, device=device)
        history.val_loss.append(val_metrics["loss"])
        history.val_acc.append(val_metrics["acc"])

    return history
