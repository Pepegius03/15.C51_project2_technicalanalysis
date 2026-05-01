"""
PyTorch Dataset classes for loading pre-generated images.

ClassificationDataset: returns (image_tensor, label_binary)  — candlestick, GAF
RegressionDataset:     returns (image_tensor, returns_tensor) — line

Transforms applied:
  - Resize to 64×64 (images should already be 64×64, kept as safety)
  - ToTensor
  - Normalize with ImageNet stats for pretrained models,
    or dataset-computed stats for scratch models (pass normalize_stats explicitly)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Optional

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

SIZE = 64


def _build_transform(mean: list[float], std: list[float]) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((SIZE, SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def imagenet_transform() -> transforms.Compose:
    return _build_transform(_IMAGENET_MEAN, _IMAGENET_STD)


def compute_dataset_stats(manifest: pd.DataFrame, split: str = "train") -> tuple[list, list]:
    """Compute per-channel mean/std over the training split images."""
    rows = manifest[manifest["split"] == split]
    pixel_sums = np.zeros(3)
    pixel_sq_sums = np.zeros(3)
    count = 0

    for path in rows["image_path"]:
        img = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
        pixel_sums += img.reshape(-1, 3).sum(axis=0)
        pixel_sq_sums += (img ** 2).reshape(-1, 3).sum(axis=0)
        count += img.shape[0] * img.shape[1]

    mean = (pixel_sums / count).tolist()
    std = (np.sqrt(pixel_sq_sums / count - (pixel_sums / count) ** 2)).tolist()
    return mean, std


class ClassificationDataset(Dataset):
    """For candlestick and GAF images. Label = label_binary."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        split: str,
        pretrained: bool = True,
        normalize_stats: Optional[tuple[list, list]] = None,
    ):
        self.df = manifest[manifest["split"] == split].reset_index(drop=True)
        if pretrained:
            self.transform = imagenet_transform()
        else:
            mean, std = normalize_stats if normalize_stats else (_IMAGENET_MEAN, _IMAGENET_STD)
            self.transform = _build_transform(mean, std)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        x = self.transform(img)
        y = int(row["label_binary"])
        return x, y


class RegressionDataset(Dataset):
    """For line plot images. Target = log-returns tensor shape (5,)."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        split: str,
        pretrained: bool = True,
        normalize_stats: Optional[tuple[list, list]] = None,
    ):
        self.df = manifest[manifest["split"] == split].reset_index(drop=True)
        self.return_cols = [f"r{i}" for i in range(1, 6)]
        if pretrained:
            self.transform = imagenet_transform()
        else:
            mean, std = normalize_stats if normalize_stats else (_IMAGENET_MEAN, _IMAGENET_STD)
            self.transform = _build_transform(mean, std)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        x = self.transform(img)
        y = torch.tensor(row[self.return_cols].values.astype(np.float32))
        return x, y
