"""PyTorch dataset and DataLoader construction for the Brain Tumor MRI dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from .preprocessing import PreprocessingConfig, preprocess_image, read_image


CLASS_NAMES = ("glioma", "meningioma", "no_tumor", "pituitary")
CLASS_DIRECTORY_ALIASES = {
    "glioma": ("glioma",),
    "meningioma": ("meningioma",),
    "no_tumor": ("no_tumor", "notumor", "no-tumor"),
    "pituitary": ("pituitary",),
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class DataConfig:
    root: Path
    image_size: int = 200
    validation_fraction: float = 0.20
    batch_size: int = 32
    num_workers: int = 0
    seed: int = 42
    normalize_for_imagenet: bool = True
    preprocessing: PreprocessingConfig = PreprocessingConfig()


def _augmentation_transform() -> transforms.Compose:
    """Match the paper: +/-10 degrees, 5% shifts, and 50% horizontal flips."""
    return transforms.Compose(
        [
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    )


def _tensor_transform(
    normalize_for_imagenet: bool, augment: bool
) -> transforms.Compose:
    operations: list[Callable] = []
    if augment:
        operations.extend([transforms.ToPILImage(), _augmentation_transform()])
    operations.append(transforms.ToTensor())
    if normalize_for_imagenet:
        operations.append(
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        )
    return transforms.Compose(operations)


class BrainTumorDataset(Dataset[tuple[Tensor, int]]):
    """Loads class-folder images and applies deterministic preprocessing."""

    def __init__(
        self,
        root: str | Path,
        image_size: int = 200,
        augment: bool = False,
        normalize_for_imagenet: bool = True,
        preprocessing: PreprocessingConfig | None = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.root}")
        self.image_size = image_size
        self.preprocessing = preprocessing or PreprocessingConfig()
        self.transform = _tensor_transform(normalize_for_imagenet, augment)
        self.samples = self._discover_samples()
        if not self.samples:
            raise FileNotFoundError(f"No supported images found below: {self.root}")

    def _discover_samples(self) -> list[tuple[Path, int]]:
        class_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}
        samples: list[tuple[Path, int]] = []
        for class_name, class_index in class_to_index.items():
            for directory_name in CLASS_DIRECTORY_ALIASES[class_name]:
                class_dir = self.root / directory_name
                if not class_dir.is_dir():
                    continue
                for path in sorted(class_dir.rglob("*")):
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                        samples.append((path, class_index))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        path, label = self.samples[index]
        image = preprocess_image(
            read_image(path),
            image_size=self.image_size,
            config=self.preprocessing,
        )
        return self.transform(image), label


def build_dataloaders(config: DataConfig) -> dict[str, DataLoader]:
    """Build train/validation/test loaders from ``train`` and ``test`` folders."""
    if not 0 < config.validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    train_root = config.root / "Training"
    if not train_root.is_dir():
        train_root = config.root / "train"
    test_root = config.root / "Testing"
    if not test_root.is_dir():
        test_root = config.root / "test"

    full_train = BrainTumorDataset(
        train_root,
        image_size=config.image_size,
        augment=False,
        normalize_for_imagenet=config.normalize_for_imagenet,
        preprocessing=config.preprocessing,
    )
    validation_size = round(len(full_train) * config.validation_fraction)
    training_size = len(full_train) - validation_size
    generator = torch.Generator().manual_seed(config.seed)
    train_subset, validation_subset = random_split(
        full_train, [training_size, validation_size], generator=generator
    )

    train_dataset = BrainTumorDataset(
        train_root,
        image_size=config.image_size,
        augment=True,
        normalize_for_imagenet=config.normalize_for_imagenet,
        preprocessing=config.preprocessing,
    )
    train_dataset.samples = [full_train.samples[index] for index in train_subset.indices]
    validation_dataset = BrainTumorDataset(
        train_root,
        image_size=config.image_size,
        augment=False,
        normalize_for_imagenet=config.normalize_for_imagenet,
        preprocessing=config.preprocessing,
    )
    validation_dataset.samples = [
        full_train.samples[index] for index in validation_subset.indices
    ]
    test_dataset = BrainTumorDataset(
        test_root,
        image_size=config.image_size,
        augment=False,
        normalize_for_imagenet=config.normalize_for_imagenet,
        preprocessing=config.preprocessing,
    )

    loader_args = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return {
        "train": DataLoader(train_dataset, shuffle=True, **loader_args),
        "validation": DataLoader(validation_dataset, shuffle=False, **loader_args),
        "test": DataLoader(test_dataset, shuffle=False, **loader_args),
    }
