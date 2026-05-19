
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    num_classes: int
    input_shape: Tuple[int, ...]
    train_dataset: Dataset | None = None
    val_dataset: Dataset | None = None
    test_dataset: Dataset | None = None


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _subset_fraction(dataset: Dataset, subset_fraction: float | None, seed: int) -> Dataset:
    if subset_fraction is None or not (0 < subset_fraction < 1):
        return dataset

    subset_n = max(1, int(len(dataset) * subset_fraction))
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:subset_n]
    return Subset(dataset, indices.tolist())


def _split_dataset(dataset: Dataset, fractions: tuple[float, ...], seed: int) -> list[Dataset]:
    if not torch.isclose(torch.tensor(sum(fractions)), torch.tensor(1.0)):
        raise ValueError(f"Split fractions must sum to 1.0, got {fractions}")

    lengths = [int(len(dataset) * frac) for frac in fractions[:-1]]
    lengths.append(len(dataset) - sum(lengths))
    return list(random_split(dataset, lengths, generator=torch.Generator().manual_seed(seed)))


def _make_bundle(
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    num_classes: int,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> DataBundle:
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        input_shape=(3, image_size, image_size),
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
    )


def make_cifar10_loaders(
    batch_size: int = 64,
    val_fraction: float = 0.1,
    num_workers: int = 2,
    root: str = "./data",
    image_size: int = 224,
    subset_fraction: float | None = None,
    seed: int = 42,
) -> DataBundle:
    """
    Create CIFAR-10 loaders for quick PEFT-style experiments.
    Resize to ViT-friendly resolution by default.
    """
    train_tf = imagenet_transform(image_size)
    test_tf = imagenet_transform(image_size)

    train_ds_full = datasets.CIFAR10(root=root, train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(root=root, train=False, download=True, transform=test_tf)
    train_ds_full = _subset_fraction(train_ds_full, subset_fraction, seed)

    val_n = int(len(train_ds_full) * val_fraction)
    train_n = len(train_ds_full) - val_n

    train_ds, val_ds = random_split(
        train_ds_full,
        [train_n, val_n],
        generator=torch.Generator().manual_seed(seed),
    )

    return _make_bundle(train_ds, val_ds, test_ds, 10, image_size, batch_size, num_workers)


def make_eurosat_loaders(
    batch_size: int = 64,
    val_fraction: float = 0.1,
    test_fraction: float = 0.2,
    num_workers: int = 2,
    root: str = "./data",
    image_size: int = 224,
    subset_fraction: float | None = None,
    seed: int = 42,
) -> DataBundle:
    """
    Create EuroSAT loaders.

    EuroSAT ships as one dataset in torchvision, so we create deterministic
    train/val/test splits locally after downloading the archive.
    """
    if not (0 < val_fraction < 1) or not (0 < test_fraction < 1):
        raise ValueError("val_fraction and test_fraction must be between 0 and 1")
    if val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be smaller than 1")

    tfm = imagenet_transform(image_size)
    dataset = datasets.EuroSAT(root=root, download=True, transform=tfm)
    dataset = _subset_fraction(dataset, subset_fraction, seed)

    train_fraction = 1.0 - val_fraction - test_fraction
    train_ds, val_ds, test_ds = _split_dataset(
        dataset, (train_fraction, val_fraction, test_fraction), seed
    )
    return _make_bundle(train_ds, val_ds, test_ds, 10, image_size, batch_size, num_workers)


class _MedMNISTSingleLabel(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        label_tensor = torch.as_tensor(label)
        return image, int(label_tensor.reshape(-1)[0].item())


def make_medmnist_loaders(
    batch_size: int = 64,
    dataset_name: str = "pathmnist",
    num_workers: int = 2,
    root: str = "./data",
    image_size: int = 224,
    subset_fraction: float | None = None,
    seed: int = 42,
) -> DataBundle:
    """
    Create loaders for a single-label MedMNIST dataset.

    Good workshop choices include ``pathmnist``, ``dermamnist``,
    ``octmnist``, and ``organamnist``. Multi-label datasets such as
    ``chestmnist`` need a different loss and are intentionally not handled here.
    """
    try:
        import medmnist
        from medmnist import INFO
    except ImportError as exc:
        raise ImportError(
            "MedMNIST support requires `pip install medmnist`."
        ) from exc

    dataset_name = dataset_name.lower()
    if dataset_name not in INFO:
        known = ", ".join(sorted(INFO))
        raise ValueError(f"Unknown MedMNIST dataset '{dataset_name}'. Known names: {known}")

    info = INFO[dataset_name]
    if info.get("task") == "multi-label, binary-class":
        raise ValueError(
            f"{dataset_name} is multi-label. Choose a single-label MedMNIST dataset "
            "or adapt the training loss."
        )

    dataset_cls = getattr(medmnist, info["python_class"])
    tfm = imagenet_transform(image_size)
    train_ds = _MedMNISTSingleLabel(
        dataset_cls(split="train", transform=tfm, download=True, root=root, as_rgb=True)
    )
    val_ds = _MedMNISTSingleLabel(
        dataset_cls(split="val", transform=tfm, download=True, root=root, as_rgb=True)
    )
    test_ds = _MedMNISTSingleLabel(
        dataset_cls(split="test", transform=tfm, download=True, root=root, as_rgb=True)
    )

    train_ds = _subset_fraction(train_ds, subset_fraction, seed)
    num_classes = len(info["label"])
    return _make_bundle(
        train_ds, val_ds, test_ds, num_classes, image_size, batch_size, num_workers
    )


def make_dataset_loaders(
    dataset_name: str,
    batch_size: int = 64,
    num_workers: int = 2,
    root: str = "./data",
    image_size: int = 224,
    subset_fraction: float | None = None,
    seed: int = 42,
    medmnist_name: str = "pathmnist",
) -> DataBundle:
    dataset_name = dataset_name.lower()
    if dataset_name == "cifar10":
        return make_cifar10_loaders(
            batch_size=batch_size,
            num_workers=num_workers,
            root=root,
            image_size=image_size,
            subset_fraction=subset_fraction,
            seed=seed,
        )
    if dataset_name == "eurosat":
        return make_eurosat_loaders(
            batch_size=batch_size,
            num_workers=num_workers,
            root=root,
            image_size=image_size,
            subset_fraction=subset_fraction,
            seed=seed,
        )
    if dataset_name == "medmnist":
        return make_medmnist_loaders(
            batch_size=batch_size,
            dataset_name=medmnist_name,
            num_workers=num_workers,
            root=root,
            image_size=image_size,
            subset_fraction=subset_fraction,
            seed=seed,
        )

    raise ValueError("dataset_name must be one of: cifar10, eurosat, medmnist")
