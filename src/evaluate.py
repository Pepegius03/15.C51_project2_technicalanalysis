"""
Evaluation metrics for classification and regression model configurations.

Classification metrics: accuracy, AUC-ROC
Regression metrics:     directional accuracy, MSE, Sharpe ratio of predicted signal
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score


@torch.no_grad()
def eval_classification(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Returns dict with keys: accuracy, auc_roc, n_samples.
    """
    model.eval()
    all_probs = []
    all_labels = []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = (probs >= 0.5).astype(int)

    accuracy = (preds == labels).mean()
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")

    return {"accuracy": float(accuracy), "auc_roc": float(auc), "n_samples": len(labels)}


@torch.no_grad()
def eval_regression(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Returns dict with keys:
      directional_accuracy, mse, sharpe_ratio, n_samples.

    Target is the 5-day cumulative return (scalar per sample).
    Sharpe: annualised, treating each sample as one holding period (~5 days).
    """
    model.eval()
    all_preds = []
    all_targets = []

    for x, y in loader:
        x = x.to(device)
        preds = model(x).squeeze(1).cpu().numpy()   # (N,)
        all_preds.append(preds)
        all_targets.append(y.numpy())

    preds   = np.concatenate(all_preds)    # (N,)
    targets = np.concatenate(all_targets)  # (N,)

    dir_acc = float(((preds > 0) == (targets > 0)).mean())
    mse     = float(np.mean((preds - targets) ** 2))

    # Windows have stride=3 and a 5-day forward period, so consecutive samples share 2
    # days of returns. Subsample every ceil(5/3)=2 steps to get non-overlapping returns
    # before computing Sharpe, avoiding inflated t-stats from correlated observations.
    step = 2
    positions  = np.sign(preds[::step])
    daily_pnl  = positions * targets[::step]
    sharpe     = float(daily_pnl.mean() / (daily_pnl.std() + 1e-8) * np.sqrt(252 / 5))

    return {
        "directional_accuracy": float(dir_acc),
        "mse": mse,
        "sharpe_ratio": sharpe,
        "n_samples": len(preds),
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task: str,
) -> dict:
    if task == "classification":
        return eval_classification(model, loader, device)
    elif task == "regression":
        return eval_regression(model, loader, device)
    else:
        raise ValueError(f"Unknown task: {task}")
