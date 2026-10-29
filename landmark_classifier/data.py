"""Dataset, transforms and dataloaders for the landmark classifier.

Expected layout::

    data/
      train/
        Brooklyn_Bridge/
        Eiffel_Tower/
        ...
      test/
        Brooklyn_Bridge/
        ...

``torchvision.datasets.ImageFolder`` sorts class names alphabetically, so the
index assigned to each class is stable as long as the folder names are stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def _to_rgb(image: Image.Image) -> Image.Image:
    """ImageFolder may return grayscale/RGBA images; the backbone needs RGB."""
    return image.convert("RGB")


def build_train_transform(
    image_size: tuple[int, int] | list[int],
    mean: Sequence[float],
    std: Sequence[float],
) -> transforms.Compose:
    """Training transform: augmentation followed by model-specific normalization."""
    return transforms.Compose(
        [
            transforms.Lambda(_to_rgb),
            transforms.RandomResizedCrop(
                size=(int(image_size[0]), int(image_size[1])),
                scale=(0.7, 1.0),
                ratio=(0.75, 1.33),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=tuple(mean), std=tuple(std)),
        ]
    )


def build_eval_transform(
    image_size: tuple[int, int] | list[int],
    mean: Sequence[float],
    std: Sequence[float],
) -> transforms.Compose:
    """Deterministic evaluation transform (resize shorter side, center crop)."""
    height, width = int(image_size[0]), int(image_size[1])
    # timm's standard recipe is Resize(256) + CenterCrop(224). Scale the resize
    # size when a backbone uses a different input resolution.
    resize_size = int(round(max(height, width) * 256 / 224))
    return transforms.Compose(
        [
            transforms.Lambda(_to_rgb),
            transforms.Resize(
                resize_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.CenterCrop((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=tuple(mean), std=tuple(std)),
        ]
    )


@dataclass
class DataLoaders:
    """Everything the training script needs to know about the data split."""

    train: DataLoader
    val: DataLoader
    test: DataLoader
    class_names: list[str]
    class_to_idx: dict[str, int]
    class_counts: dict[str, int]
    train_size: int
    val_size: int
    test_size: int
    test_paths: list[str]


def build_data_loaders(
    data_dir: str | Path,
    image_size: tuple[int, int] | list[int],
    mean: Sequence[float],
    std: Sequence[float],
    batch_size: int = 64,
    val_fraction: float = 0.25,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoaders:
    """Build stratified train/validation loaders plus the held-out test loader."""
    data_dir = Path(data_dir)
    train_root = data_dir / "train"
    test_root = data_dir / "test"

    if not train_root.is_dir():
        raise FileNotFoundError(
            f"Training directory not found: {train_root}\n"
            "Expected data/train/<class_name>/<images>."
        )
    if not test_root.is_dir():
        raise FileNotFoundError(
            f"Test directory not found: {test_root}\n"
            "Expected data/test/<class_name>/<images>."
        )
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between 0 and 1.")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0.")

    train_base = datasets.ImageFolder(train_root)
    test_base = datasets.ImageFolder(test_root)

    if train_base.classes != test_base.classes:
        raise ValueError(
            "Train and test class folders do not match.\n"
            f"Train: {train_base.classes}\n"
            f"Test:  {test_base.classes}"
        )

    class_names = list(train_base.classes)
    targets = np.asarray(train_base.targets, dtype=np.int64)
    class_counts_array = np.bincount(targets, minlength=len(class_names))

    if class_counts_array.min() < 2:
        raise ValueError(
            "Every class needs at least two training images so that a "
            "stratified train/validation split is possible."
        )

    indices = np.arange(len(train_base))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_fraction,
        stratify=targets,
        random_state=seed,
        shuffle=True,
    )

    train_dataset = Subset(
        datasets.ImageFolder(train_root, transform=build_train_transform(image_size, mean, std)),
        train_indices,
    )
    val_dataset = Subset(
        datasets.ImageFolder(train_root, transform=build_eval_transform(image_size, mean, std)),
        val_indices,
    )
    test_dataset = datasets.ImageFolder(
        test_root,
        transform=build_eval_transform(image_size, mean, std),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        common["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **common,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, **common)

    return DataLoaders(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        class_names=class_names,
        class_to_idx=dict(train_base.class_to_idx),
        class_counts={
            name: int(class_counts_array[index])
            for index, name in enumerate(class_names)
        },
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        test_size=len(test_dataset),
        test_paths=[str(path) for path, _ in test_base.samples],
    )


def build_test_loader(
    data_dir: str | Path,
    image_size: tuple[int, int] | list[int],
    mean: Sequence[float],
    std: Sequence[float],
    batch_size: int = 64,
    num_workers: int = 0,
    class_names: Sequence[str] | None = None,
) -> tuple[DataLoader, list[str], list[str]]:
    """Build a test loader from a checkpoint's preprocessing configuration."""
    test_root = Path(data_dir) / "test"
    if not test_root.is_dir():
        raise FileNotFoundError(
            f"Test directory not found: {test_root}\n"
            "Expected data/test/<class_name>/<images>."
        )

    dataset = datasets.ImageFolder(
        test_root,
        transform=build_eval_transform(image_size, mean, std),
    )

    if class_names is not None and list(dataset.classes) != list(class_names):
        raise ValueError(
            "The checkpoint was trained with different classes than this test set.\n"
            f"Checkpoint: {list(class_names)}\n"
            f"Dataset:    {list(dataset.classes)}"
        )

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        common["persistent_workers"] = True

    loader = DataLoader(dataset, shuffle=False, **common)
    paths = [str(path) for path, _ in dataset.samples]
    return loader, list(dataset.classes), paths
