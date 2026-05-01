"""
PyTorch Dataset classes for loading pre-generated images.

ClassificationDataset: returns (image_tensor, label_binary)  — candlestick, GAF
RegressionDataset:     returns (image_tensor, returns_tensor) — line

No torchvision dependency — transforms implemented with PIL + numpy + torch.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SIZE = 64


def _to_tensor(img: Image.Image, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    img = img.resize((SIZE, SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0   # (H, W, 3)
    arr = (arr - mean) / std                          # normalize
    return torch.from_numpy(arr.transpose(2, 0, 1))  # (3, H, W)


def compute_dataset_stats(manifest: pd.DataFrame, split: str = "train") -> tuple[list, list]:
    """Compute per-channel mean/std over the training split images."""
    rows = manifest[manifest["split"] == split]
    pixel_sums = np.zeros(3, dtype=np.float64)
    pixel_sq_sums = np.zeros(3, dtype=np.float64)
    count = 0

    for path in rows["image_path"]:
        img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
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
        if normalize_stats:
            self.mean = np.array(normalize_stats[0], dtype=np.float32)
            self.std  = np.array(normalize_stats[1], dtype=np.float32)
        else:
            self.mean = _IMAGENET_MEAN
            self.std  = _IMAGENET_STD

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        x = _to_tensor(img, self.mean, self.std)
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
        if normalize_stats:
            self.mean = np.array(normalize_stats[0], dtype=np.float32)
            self.std  = np.array(normalize_stats[1], dtype=np.float32)
        else:
            self.mean = _IMAGENET_MEAN
            self.std  = _IMAGENET_STD

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        x = _to_tensor(img, self.mean, self.std)
        y = torch.tensor(row[self.return_cols].values.astype(np.float32))
        return x, y
