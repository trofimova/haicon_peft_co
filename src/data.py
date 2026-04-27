
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    num_classes: int
    input_shape: Tuple[int, ...]


def make_cifar10_loaders(
    batch_size: int = 64,
    val_fraction: float = 0.1,
    num_workers: int = 2,
    root: str = "./data",
    image_size: int = 224,
    subset_fraction: float | None = None,
) -> DataBundle:
    """
    Create CIFAR-10 loaders for quick PEFT-style experiments.
    Resize to ViT-friendly resolution by default.
    """
    # ImageNet mean/std — matches ViT-B/16 (and most torchvision) pretraining
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    train_ds_full = datasets.CIFAR10(root=root, train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(root=root, train=False, download=True, transform=test_tf)

    if subset_fraction is not None and 0 < subset_fraction < 1:
        subset_n = max(1, int(len(train_ds_full) * subset_fraction))
        train_ds_full, _ = random_split(
            train_ds_full,
            [subset_n, len(train_ds_full) - subset_n],
            generator=torch.Generator().manual_seed(42),
        )

    val_n = int(len(train_ds_full) * val_fraction)
    train_n = len(train_ds_full) - val_n

    train_ds, val_ds = random_split(
        train_ds_full,
        [train_n, val_n],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=10,
        input_shape=(3, image_size, image_size),
    )
