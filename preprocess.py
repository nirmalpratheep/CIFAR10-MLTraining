import torch
from torchvision import datasets
from albumentations import (
    Compose, HorizontalFlip, ShiftScaleRotate, CoarseDropout,
    Normalize, ColorJitter, PadIfNeeded, RandomCrop
)
from albumentations.pytorch import ToTensorV2
import numpy as np


# CIFAR-10 statistics (RGB)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _coarse_dropout_fill_value_from_mean(mean_rgb: tuple[float, float, float]) -> tuple[int, int, int]:
    """Convert mean RGB (0–1) to 0–255 scale for CoarseDropout fill color."""
    return tuple(int(m * 255.0) for m in mean_rgb)


class AlbumentationsAdapter:
    """Adapter to make Albumentations transforms compatible with torchvision datasets."""
    def __init__(self, transform: Compose):
        self.transform = transform

    def __call__(self, img):
        img_np = np.array(img)
        augmented = self.transform(image=img_np)
        return augmented["image"]


def get_transforms(_: str | None = None):
    fill_value = _coarse_dropout_fill_value_from_mean(CIFAR10_MEAN)

    train_transforms = Compose([
        PadIfNeeded(min_height=36, min_width=36, border_mode=0, p=1.0),
        RandomCrop(height=32, width=32, p=1.0),
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.3),
        CoarseDropout(
            num_holes_range=(1, 1),
            hole_height_range=(8, 8),
            hole_width_range=(8, 8),
            fill=fill_value,
            p=0.4,
        ),
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=0.4),
        Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ToTensorV2(),
    ])

    test_transforms = Compose([
        Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ToTensorV2(),
    ])

    return AlbumentationsAdapter(train_transforms), AlbumentationsAdapter(test_transforms)


def get_datasets(data_dir: str = "./data", model_name: str | None = None):
    """Return CIFAR-10 train/test datasets with Albumentations transforms."""
    train_transforms, test_transforms = get_transforms(model_name)

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transforms
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transforms
    )

    return train_dataset, test_dataset


def get_data_loaders(
    batch_size: int = 128,
    data_dir: str = "./data",
    num_workers: int = 2,
    pin_memory: bool = True,
    shuffle_train: bool = True,
    model_name: str | None = None,
):
    """Return CIFAR-10 train/test dataloaders with on-the-fly Albumentations."""
    train_dataset, test_dataset = get_datasets(data_dir=data_dir, model_name=model_name)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader
