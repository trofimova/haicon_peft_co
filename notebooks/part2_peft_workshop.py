from __future__ import annotations

import copy
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from transformers import ViTModel

from src.data import IMAGENET_MEAN, IMAGENET_STD, make_dataset_loaders
from src.methods.bitfit import BitFitClassifier
from src.methods.linear_probe import LinearProbeModel
from src.methods.lora import LoRAClassifier
from src.methods.partial_ft import PartialFineTuneClassifier
from src.methods.prompt_tuning import PromptTunedClassifier
from src.training import count_trainable_parameters, evaluate, freeze_module, train_model


@dataclass
class WorkshopConfig:
    model_name: str = "WinKawaks/vit-small-patch16-224"
    image_size: int = 224

    dataset_name: str = "eurosat"
    medmnist_name: str = "pathmnist"
    data_root: str = "./data"
    data_subset_fraction: float | None = None
    num_workers: int = 0

    batch_size: int = 16
    seed: int = 42
    repeat_seeds: list[int] = field(default_factory=lambda: [42])
    shots_per_class: list[int] = field(default_factory=lambda: [1, 4, 8])
    eval_per_class: int = 10
    domain_shift_strength: float = 0.35
    target_adaptation_shots: int = 0

    methods_to_run: list[str] = field(
        default_factory=lambda: ["linear_probe", "lora", "visual_prompt", "full_ft"]
    )

    lora_rank: int = 8
    lora_target: list[str] = field(default_factory=lambda: ["query", "value"])
    num_prompt_tokens: int = 8
    partial_n_blocks: int = 1

    default_epochs: int = 3
    epochs_by_shot: dict[int, int] = field(default_factory=dict)
    epochs_by_method: dict[str, int] = field(default_factory=lambda: {"full_ft": 6})
    lr_by_method: dict[str, float] = field(
        default_factory=lambda: {
            "linear_probe": 5e-3,
            "bitfit": 1e-3,
            "lora": 3e-3,
            "visual_prompt": 3e-3,
            "partial_ft": 1e-4,
            "full_ft": 1e-5,
        }
    )
    weight_decay: float = 1e-2
    parameter_budget: int = 250_000
    output_root: str = "outputs/part3"


class HFViTBackbone(nn.Module):
    """Thin wrapper around a HuggingFace ViTModel that returns the CLS token."""

    def __init__(self, model_name: str):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        self.feature_dim = self.vit.config.hidden_size

    def forward(self, x):
        return self.vit(pixel_values=x).last_hidden_state[:, 0]


class FullFineTuneClassifier(nn.Module):
    """Train the whole backbone plus a classification head."""

    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))


class ShiftedDataset(Dataset):
    """Dataset wrapper that applies the controlled target-domain distortion."""

    def __init__(self, dataset: Dataset, strength: float, seed: int, image_size: int):
        self.dataset = dataset
        self.strength = strength
        self.seed = seed
        self.image_size = image_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x_shifted = apply_domain_shift(
            x.unsqueeze(0),
            strength=self.strength,
            seed=self.seed + idx,
            image_size=self.image_size,
        ).squeeze(0)
        return x_shifted, y


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def apply_domain_shift(x, strength=0.35, seed=123, image_size=224):
    if strength <= 0:
        return x.clone()

    g = torch.Generator().manual_seed(seed)
    shifted = x.clone()

    scales = torch.tensor([
        1.0 + 0.70 * strength,
        1.0 - 0.45 * strength,
        1.0 + 0.20 * strength,
    ]).view(1, 3, 1, 1)
    bias = torch.tensor([0.12 * strength, -0.08 * strength, 0.04 * strength]).view(
        1, 3, 1, 1
    )
    shifted = shifted * scales + bias

    pixels = max(1, int(image_size * 0.04 * strength))
    shifted = shifted.roll(shifts=pixels, dims=-1)

    noise = torch.randn(shifted.shape, generator=g) * (0.25 * strength)
    return shifted + noise


def get_label(dataset: Dataset, idx: int) -> int:
    if isinstance(dataset, Subset):
        return get_label(dataset.dataset, int(dataset.indices[idx]))
    if hasattr(dataset, "targets"):
        return int(dataset.targets[idx])
    if hasattr(dataset, "labels"):
        label = torch.as_tensor(dataset.labels[idx]).reshape(-1)[0]
        return int(label.item())
    if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "labels"):
        label = torch.as_tensor(dataset.dataset.labels[idx]).reshape(-1)[0]
        return int(label.item())

    _, label = dataset[idx]
    return int(torch.as_tensor(label).reshape(-1)[0].item())


def unwrap_dataset(dataset: Dataset) -> Dataset:
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    if hasattr(dataset, "dataset"):
        return unwrap_dataset(dataset.dataset)
    return dataset


def get_class_names(dataset: Dataset, num_classes: int) -> list[str]:
    base_dataset = unwrap_dataset(dataset)
    if hasattr(base_dataset, "classes"):
        return list(base_dataset.classes)
    if hasattr(base_dataset, "dataset"):
        nested = unwrap_dataset(base_dataset.dataset)
        if hasattr(nested, "classes"):
            return list(nested.classes)
    return [str(i) for i in range(num_classes)]


def denormalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=x.dtype, device=x.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=x.dtype, device=x.device).view(3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def get_sample_images(
    config: WorkshopConfig | None = None,
    split: str = "train",
    examples_per_class: int = 1,
    max_classes: int | None = None,
    denormalize: bool = True,
) -> list[tuple[torch.Tensor, str]]:
    """Return a small class-balanced image sample for notebook visualization."""
    config = config or WorkshopConfig()
    bundle = make_dataset_loaders(
        config.dataset_name,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        root=config.data_root,
        image_size=config.image_size,
        subset_fraction=config.data_subset_fraction,
        seed=config.seed,
        medmnist_name=config.medmnist_name,
    )

    datasets_by_split = {
        "train": bundle.train_dataset,
        "val": bundle.val_dataset,
        "test": bundle.test_dataset,
    }
    if split not in datasets_by_split:
        raise ValueError("split must be one of: train, val, test")

    dataset = datasets_by_split[split]
    class_names = get_class_names(dataset, bundle.num_classes)
    class_limit = bundle.num_classes if max_classes is None else min(max_classes, bundle.num_classes)
    counts = defaultdict(int)
    examples = []

    for idx in range(len(dataset)):
        label_idx = get_label(dataset, idx)
        if label_idx >= class_limit or counts[label_idx] >= examples_per_class:
            continue

        image, _ = dataset[idx]
        if denormalize:
            image = denormalize_imagenet(image)
        examples.append((image.cpu(), class_names[label_idx]))
        counts[label_idx] += 1

        if len(examples) >= class_limit * examples_per_class:
            break

    return examples


def class_index_map_from_dataset(dataset: Dataset, num_classes: int):
    by_class = defaultdict(list)
    for idx in range(len(dataset)):
        label = get_label(dataset, idx)
        if 0 <= label < num_classes:
            by_class[label].append(idx)
    return by_class


def take_balanced_indices(by_class, shots_per_class, num_classes, offset=0, seed=None):
    chosen = []
    for class_idx in range(num_classes):
        class_indices = by_class[class_idx]
        if seed is None:
            ordered = class_indices
        else:
            g = torch.Generator().manual_seed(seed + class_idx)
            perm = torch.randperm(len(class_indices), generator=g).tolist()
            ordered = [class_indices[i] for i in perm]
        chosen.extend(ordered[offset : offset + shots_per_class])
    return chosen


def load_experiment_data(config: WorkshopConfig):
    bundle = make_dataset_loaders(
        config.dataset_name,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        root=config.data_root,
        image_size=config.image_size,
        subset_fraction=config.data_subset_fraction,
        seed=config.seed,
        medmnist_name=config.medmnist_name,
    )
    by_class = class_index_map_from_dataset(bundle.train_dataset, bundle.num_classes)
    required_per_class = max(config.shots_per_class) + config.target_adaptation_shots
    counts = {c: len(by_class[c]) for c in range(bundle.num_classes)}

    missing = [c for c, n in counts.items() if n < required_per_class]
    if missing:
        raise ValueError(
            f"Not enough training examples for classes {missing}. "
            "Increase data size or reduce shots."
        )

    return {
        "num_classes": bundle.num_classes,
        "train_dataset": bundle.train_dataset,
        "val_dataset": bundle.val_dataset,
        "test_dataset": bundle.test_dataset,
        "by_class": by_class,
        "counts": counts,
    }


def make_few_shot_loaders(data: dict[str, Any], shots_per_class: int, repeat_seed: int, config: WorkshopConfig):
    train_idx = take_balanced_indices(
        data["by_class"],
        shots_per_class,
        data["num_classes"],
        offset=0,
        seed=repeat_seed,
    )
    train_dataset = Subset(data["train_dataset"], train_idx)

    if config.target_adaptation_shots > 0:
        target_train_idx = take_balanced_indices(
            data["by_class"],
            config.target_adaptation_shots,
            data["num_classes"],
            offset=shots_per_class,
            seed=repeat_seed,
        )
        target_train = ShiftedDataset(
            Subset(data["train_dataset"], target_train_idx),
            strength=config.domain_shift_strength,
            seed=repeat_seed + 10_000,
            image_size=config.image_size,
        )
        train_dataset = ConcatDataset([train_dataset, target_train])

    eval_by_class = class_index_map_from_dataset(data["val_dataset"], data["num_classes"])
    eval_shots = min(config.eval_per_class, min(len(v) for v in eval_by_class.values()))
    eval_idx = take_balanced_indices(
        eval_by_class,
        eval_shots,
        data["num_classes"],
        offset=0,
        seed=repeat_seed,
    )
    source_eval = Subset(data["val_dataset"], eval_idx)
    target_eval = ShiftedDataset(
        source_eval,
        strength=config.domain_shift_strength,
        seed=repeat_seed + 20_000,
        image_size=config.image_size,
    )

    train_generator = torch.Generator().manual_seed(repeat_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        generator=train_generator,
    )
    source_loader = DataLoader(
        source_eval,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    target_loader = DataLoader(
        target_eval,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, source_loader, target_loader


def build_model(method: str, backbone: nn.Module, feature_dim: int, num_classes: int, config: WorkshopConfig):
    bb = copy.deepcopy(backbone)

    if method == "linear_probe":
        return LinearProbeModel(bb, feature_dim, num_classes)
    if method == "bitfit":
        return BitFitClassifier(bb, feature_dim, num_classes)
    if method == "lora":
        return LoRAClassifier(
            bb,
            feature_dim,
            num_classes,
            target_modules=config.lora_target,
            rank=config.lora_rank,
        )
    if method == "visual_prompt":
        return PromptTunedClassifier(
            bb,
            feature_dim,
            num_classes,
            num_prompt_tokens=config.num_prompt_tokens,
        )
    if method == "partial_ft":
        blocks = list(bb.vit.encoder.layer)
        n_blocks = max(0, min(config.partial_n_blocks, len(blocks)))
        return PartialFineTuneClassifier(
            bb,
            feature_dim,
            num_classes,
            modules_to_unfreeze=blocks[-n_blocks:] if n_blocks else [],
        )
    if method == "full_ft":
        return FullFineTuneClassifier(bb, feature_dim, num_classes)

    raise ValueError(
        "Unknown method. Choose from: linear_probe, bitfit, lora, "
        "visual_prompt, partial_ft, full_ft"
    )


def optimizer_for(method: str, model: nn.Module, config: WorkshopConfig):
    lr = config.lr_by_method.get(method, 3e-3)
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable, lr=lr, weight_decay=config.weight_decay)


def epochs_for(method: str, shots_per_class: int, config: WorkshopConfig) -> int:
    if method in config.epochs_by_method:
        return config.epochs_by_method[method]
    if shots_per_class in config.epochs_by_shot:
        return config.epochs_by_shot[shots_per_class]
    return config.default_epochs


def run_single_experiment(
    method: str,
    shots_per_class: int,
    repeat_seed: int,
    backbone: nn.Module,
    feature_dim: int,
    data: dict[str, Any],
    device: str,
    config: WorkshopConfig,
):
    train_loader, source_loader, target_loader = make_few_shot_loaders(
        data, shots_per_class, repeat_seed, config
    )
    set_seed(repeat_seed)
    model = build_model(method, backbone, feature_dim, data["num_classes"], config).to(device)
    trainable = count_trainable_parameters(model)
    total = sum(p.numel() for p in model.parameters())
    optimizer = optimizer_for(method, model, config)
    epochs = epochs_for(method, shots_per_class, config)

    print()
    print(
        f"{method} | shots/class={shots_per_class} | seed={repeat_seed} | "
        f"trainable={trainable:,} | epochs={epochs}"
    )
    history = train_model(model, train_loader, source_loader, optimizer, epochs=epochs, device=device)
    train_metrics = {"loss": history.train_loss[-1], "acc": history.train_acc[-1]}
    source_metrics = evaluate(model, source_loader, device=device)
    target_metrics = evaluate(model, target_loader, device=device)

    result = {
        "method": method,
        "shots_per_class": shots_per_class,
        "repeat_seed": repeat_seed,
        "train_examples": len(train_loader.dataset),
        "epochs": epochs,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": 100 * trainable / total,
        "train_loss": train_metrics["loss"],
        "train_acc": train_metrics["acc"],
        "source_loss": source_metrics["loss"],
        "source_acc": source_metrics["acc"],
        "target_loss": target_metrics["loss"],
        "target_acc": target_metrics["acc"],
        "generalization_gap": train_metrics["acc"] - source_metrics["acc"],
        "domain_gap": source_metrics["acc"] - target_metrics["acc"],
        "history": history,
    }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def make_output_dir(config: WorkshopConfig) -> Path:
    dataset_tag = config.dataset_name
    if config.dataset_name == "medmnist":
        dataset_tag = f"medmnist-{config.medmnist_name}"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(config.output_root) / f"{timestamp}_{dataset_tag}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def history_to_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        history = result["history"]
        for epoch, (train_loss, train_acc, val_loss, val_acc) in enumerate(
            zip(history.train_loss, history.train_acc, history.val_loss, history.val_acc),
            start=1,
        ):
            rows.append({
                "method": result["method"],
                "shots_per_class": result["shots_per_class"],
                "repeat_seed": result["repeat_seed"],
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            })
    return rows


def save_results(results: list[dict[str, Any]], config: WorkshopConfig, run_dir: Path, device: str, num_classes: int):
    df = pd.DataFrame([{k: v for k, v in row.items() if k != "history"} for row in results])
    history_df = pd.DataFrame(history_to_rows(results))
    cfg = asdict(config)
    cfg["device"] = device
    cfg["num_classes"] = num_classes

    df.to_csv(run_dir / "results.csv", index=False)
    df.to_json(run_dir / "results.json", orient="records", indent=2)
    history_df.to_csv(run_dir / "history.csv", index=False)
    with (run_dir / "config.json").open("w") as f:
        json.dump(cfg, f, indent=2)
    return df, history_df


def run_experiment_grid(config: WorkshopConfig):
    set_seed(config.seed)
    device = get_device()

    print("device:", device)
    print("dataset:", config.dataset_name)
    print("methods:", config.methods_to_run)
    print("shots/class:", config.shots_per_class)
    print("repeat seeds:", config.repeat_seeds)

    backbone = HFViTBackbone(config.model_name).to(device)
    freeze_module(backbone)
    feature_dim = backbone.feature_dim
    print(f"feature dim: {feature_dim}")

    data = load_experiment_data(config)
    print("classes:", data["num_classes"])
    print("train examples/class:", data["counts"])

    run_dir = make_output_dir(config)
    print("output dir:", run_dir)

    results = []
    for repeat_seed in config.repeat_seeds:
        for shots in config.shots_per_class:
            for method in config.methods_to_run:
                results.append(
                    run_single_experiment(
                        method,
                        shots,
                        repeat_seed,
                        backbone,
                        feature_dim,
                        data,
                        device,
                        config,
                    )
                )

    df, history_df = save_results(results, config, run_dir, device, data["num_classes"])
    return run_dir, df, history_df
