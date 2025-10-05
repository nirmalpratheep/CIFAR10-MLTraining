import torch
from torchvision import datasets
from albumentations import Compose, HorizontalFlip, ShiftScaleRotate, CoarseDropout, Normalize, ColorJitter
from albumentations import PadIfNeeded, RandomCrop
from albumentations.pytorch import ToTensorV2
import numpy as np
import os
from typing import Optional, Tuple, Any


# CIFAR-10 statistics (RGB)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _coarse_dropout_fill_value_from_mean(mean_rgb: tuple[float, float, float]) -> tuple[int, int, int]:
    return tuple(int(m * 255.0) for m in mean_rgb)


class AlbumentationsAdapter:
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
        fill=_coarse_dropout_fill_value_from_mean(CIFAR10_MEAN),
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


class CachedTransformDataset(torch.utils.data.Dataset):
    """Wrap a torchvision dataset and cache transformed samples to disk.

    Saves a dict {"image": tensor, "target": int} as .pt per index.
    """

    def __init__(
        self,
        base_dataset: torch.utils.data.Dataset,
        split_name: str,
        cache_dir: str,
        enable_cache: bool = False,
        cache_namespace: Optional[str] = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.split_name = split_name
        self.enable_cache = enable_cache
        # namespace allows changing folder when transforms change
        self.cache_root = os.path.join(cache_dir, split_name, cache_namespace or "default")
        if self.enable_cache:
            os.makedirs(self.cache_root, exist_ok=True)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _cache_path(self, index: int) -> str:
        return os.path.join(self.cache_root, f"{index}.pt")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        if self.enable_cache:
            path = self._cache_path(index)
            if os.path.exists(path):
                obj = torch.load(path)
                return obj["image"], obj["target"]

        image, target = self.base_dataset[index]
        if self.enable_cache:
            torch.save({"image": image, "target": target}, self._cache_path(index))
        return image, target


def get_datasets(
    data_dir: str = "./data",
    model_name: str | None = None,
    cache_transforms: bool = False,
    cache_dir: str = "./cache",
    cache_namespace: Optional[str] = None,
):
    train_transforms, test_transforms = get_transforms(model_name)
    train_base = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transforms)
    test_base = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transforms)

    train_dataset = CachedTransformDataset(
        train_base, split_name="train", cache_dir=cache_dir, enable_cache=cache_transforms, cache_namespace=cache_namespace
    )
    test_dataset = CachedTransformDataset(
        test_base, split_name="test", cache_dir=cache_dir, enable_cache=cache_transforms, cache_namespace=cache_namespace
    )
    return train_dataset, test_dataset


def get_data_loaders(
    batch_size: int = 128,
    data_dir: str = "./data",
    num_workers: int = 2,  # Restored to 2 for better performance
    pin_memory: bool = True,  # Restored for GPU training
    shuffle_train: bool = True,
    model_name: str | None = None,
    cache_transforms: bool = False,  # Default to False, enable with --cache_transforms
    cache_dir: str = "./cache",
    cache_namespace: Optional[str] = None,
):
    train_dataset, test_dataset = get_datasets(
        data_dir=data_dir,
        model_name=model_name,
        cache_transforms=cache_transforms,
        cache_dir=cache_dir,
        cache_namespace=cache_namespace,
    )

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


