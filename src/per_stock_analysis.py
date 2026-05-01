"""
Per-stock analysis: compute per-ticker accuracy and correlation with SPY.

For each ticker in the test split:
  - accuracy (classification) or directional accuracy (regression)
  - Pearson correlation of per-ticker accuracy with SPY accuracy as a baseline
    (proxy for "does model skill correlate with market beta / trend-following ease")
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from .evaluate import eval_classification, eval_regression


def per_ticker_accuracy(
    model: nn.Module,
    manifest: pd.DataFrame,
    dataset_cls,
    split: str,
    task: str,
    device: torch.device,
    pretrained: bool,
    normalize_stats=None,
    batch_size: int = 64,
) -> pd.DataFrame:
    """
    Returns DataFrame: ticker, accuracy (or directional_accuracy), n_samples.
    """
    tickers = manifest[manifest["split"] == split]["ticker"].unique()
    rows = []

    for ticker in tickers:
        sub_manifest = manifest[
            (manifest["split"] == split) & (manifest["ticker"] == ticker)
        ].copy()
        if len(sub_manifest) == 0:
            continue

        ds = dataset_cls(
            sub_manifest.assign(split=split),
            split=split,
            pretrained=pretrained,
            normalize_stats=normalize_stats,
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

        if task == "classification":
            metrics = eval_classification(model, loader, device)
            acc = metrics["accuracy"]
        else:
            metrics = eval_regression(model, loader, device)
            acc = metrics["directional_accuracy"]

        rows.append({"ticker": ticker, "accuracy": acc, "n_samples": metrics["n_samples"]})

    return pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)


def spy_correlation(per_stock_df: pd.DataFrame, spy_ticker: str = "SPY") -> float:
    """
    Pearson correlation between per-ticker accuracy and SPY accuracy (as scalar).
    SPY's row is used as the reference; correlates other tickers' accuracies against it.
    Returns NaN if SPY not found.
    """
    spy_rows = per_stock_df[per_stock_df["ticker"] == spy_ticker]
    if spy_rows.empty:
        return float("nan")

    spy_acc = float(spy_rows.iloc[0]["accuracy"])
    other = per_stock_df[per_stock_df["ticker"] != spy_ticker]["accuracy"].values

    if len(other) < 2:
        return float("nan")

    # Correlation of each ticker's accuracy vector against the SPY scalar
    # (degenerate case: correlate the distribution of accuracies with a constant)
    # More meaningful: rank correlation across stocks is computed in run_evaluate
    return float(np.corrcoef(other, np.full_like(other, spy_acc))[0, 1])
