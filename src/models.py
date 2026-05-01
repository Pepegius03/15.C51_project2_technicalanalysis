"""
Six model configurations:

  #  Image         Strategy    Head              Loss
  1  line          scratch     regression (5)    MSE
  2  line          pretrained  regression (5)    MSE
  3  candlestick   scratch     classification    CrossEntropy
  4  candlestick   pretrained  classification    CrossEntropy
  5  gaf           scratch     classification    CrossEntropy
  6  gaf           pretrained  classification    CrossEntropy

Pretrained: ResNet-18 with final fc replaced.
Scratch: 4 conv blocks [16→32→64→128] + BN + ReLU + MaxPool,
         then Linear(128*4*4, 256) → Linear(256, n_out).
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

N_CLASS = 2
N_REG = 5


# ── Scratch CNN ───────────────────────────────────────────────────────────────

class _ConvBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )


class ScratchCNN(nn.Module):
    """Input: (B, 3, 64, 64). After 4 MaxPool(2): spatial = 4×4."""

    def __init__(self, n_out: int):
        super().__init__()
        self.features = nn.Sequential(
            _ConvBlock(3, 16),
            _ConvBlock(16, 32),
            _ConvBlock(32, 64),
            _ConvBlock(64, 128),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ── Pretrained ResNet-18 ───────────────────────────────────────────────────────

class PretrainedResNet(nn.Module):
    def __init__(self, n_out: int):
        super().__init__()
        base = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = base.fc.in_features
        base.fc = nn.Linear(in_features, n_out)
        self.model = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_model(config_id: int) -> tuple[nn.Module, nn.Module, float]:
    """
    Returns (model, loss_fn, lr) for a given config id 1–6.
    """
    configs = {
        1: ("line",        "scratch",    "regression"),
        2: ("line",        "pretrained", "regression"),
        3: ("candlestick", "scratch",    "classification"),
        4: ("candlestick", "pretrained", "classification"),
        5: ("gaf",         "scratch",    "classification"),
        6: ("gaf",         "pretrained", "classification"),
    }
    if config_id not in configs:
        raise ValueError(f"config_id must be 1–6, got {config_id}")

    _, strategy, task = configs[config_id]

    if task == "regression":
        n_out = N_REG
        loss_fn = nn.MSELoss()
        lr = 1e-4
    else:
        n_out = N_CLASS
        loss_fn = nn.CrossEntropyLoss()
        lr = 1e-4 if strategy == "pretrained" else 1e-3

    if strategy == "pretrained":
        model = PretrainedResNet(n_out)
    else:
        model = ScratchCNN(n_out)

    return model, loss_fn, lr


def config_meta(config_id: int) -> dict:
    configs = {
        1: dict(image="line",        strategy="scratch",    task="regression"),
        2: dict(image="line",        strategy="pretrained", task="regression"),
        3: dict(image="candlestick", strategy="scratch",    task="classification"),
        4: dict(image="candlestick", strategy="pretrained", task="classification"),
        5: dict(image="gaf",         strategy="scratch",    task="classification"),
        6: dict(image="gaf",         strategy="pretrained", task="classification"),
    }
    return configs[config_id]
