"""
Sliding window slicing, normalization, and label computation.

Window spec:
  - 60 consecutive trading days per stock, stride = 3
  - Skip any window with NaN in OHLCV or no forward close available
  - Z-score normalization of close prices, per window
  - Raw OHLCV kept for candlestick image generation

Labels:
  Binary  (candlestick + GAF): 1 if close[T+5] > close[T], else 0
  Regression (line):           log-returns [r1..r5], ri = log(close[T+i]/close[T+i-1])
                                direction = 1 if sum(ri) > 0 else 0

Time split: Train ≤ 2018, Val 2019–2020, Test 2021–2024
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

WINDOW_LEN = 60
STRIDE = 3
FORWARD_DAYS = 5

TRAIN_END = "2018-12-31"
VAL_START = "2019-01-01"
VAL_END = "2020-12-31"
TEST_START = "2021-01-01"


@dataclass
class WindowRecord:
    ticker: str
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    split: str                       # "train" | "val" | "test"

    # Raw OHLCV array shape (60, 5) — open/high/low/close/volume
    ohlcv: np.ndarray

    # Z-score normalized close, shape (60,)
    close_norm: np.ndarray

    # Labels
    label_binary: int                # 0 or 1
    returns: np.ndarray              # shape (5,) log-returns
    label_direction: int             # derived from sum(returns)


def _zscore(arr: np.ndarray) -> np.ndarray:
    mu, sigma = arr.mean(), arr.std()
    if sigma < 1e-8:
        return np.zeros_like(arr)
    return (arr - mu) / sigma


def _assign_split(ts: pd.Timestamp) -> str:
    if ts <= pd.Timestamp(TRAIN_END):
        return "train"
    if ts <= pd.Timestamp(VAL_END):
        return "val"
    return "test"


def extract_windows(
    ticker: str,
    df: pd.DataFrame,
) -> list[WindowRecord]:
    """
    df must have columns [open, high, low, close, volume] and a sorted DatetimeIndex.
    Returns list of WindowRecord objects.
    """
    closes = df["close"].values
    n = len(df)
    dates = df.index

    records: list[WindowRecord] = []

    for start_i in range(0, n - WINDOW_LEN - FORWARD_DAYS, STRIDE):
        end_i = start_i + WINDOW_LEN - 1  # inclusive last window index
        fwd_i = end_i + FORWARD_DAYS      # T+5 index

        if fwd_i >= n:
            break

        ohlcv = df.iloc[start_i : end_i + 1][["open", "high", "low", "close", "volume"]].values

        # Skip windows with any NaN
        if np.isnan(ohlcv).any():
            continue

        fwd_closes = closes[end_i : fwd_i + 1]  # T, T+1, ..., T+5
        if np.isnan(fwd_closes).any():
            continue

        close_arr = ohlcv[:, 3]  # close column
        close_norm = _zscore(close_arr)

        close_T = closes[end_i]
        label_binary = 1 if closes[fwd_i] > close_T else 0

        # Log-returns r1..r5
        returns = np.array([
            np.log(fwd_closes[i + 1] / fwd_closes[i])
            for i in range(FORWARD_DAYS)
        ], dtype=np.float32)
        label_direction = 1 if returns.sum() > 0 else 0

        window_end_ts = dates[end_i]
        split = _assign_split(window_end_ts)

        records.append(WindowRecord(
            ticker=ticker,
            window_start=dates[start_i],
            window_end=window_end_ts,
            split=split,
            ohlcv=ohlcv.astype(np.float32),
            close_norm=close_norm.astype(np.float32),
            label_binary=label_binary,
            returns=returns,
            label_direction=label_direction,
        ))

    return records


def build_manifest(windows: list[WindowRecord]) -> pd.DataFrame:
    rows = []
    for w in windows:
        row = {
            "ticker": w.ticker,
            "window_start": w.window_start,
            "window_end": w.window_end,
            "split": w.split,
            "label_binary": w.label_binary,
            "label_direction": w.label_direction,
        }
        for i, r in enumerate(w.returns, 1):
            row[f"r{i}"] = float(r)
        rows.append(row)
    return pd.DataFrame(rows)
