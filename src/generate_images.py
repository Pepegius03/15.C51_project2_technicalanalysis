"""
Generate 64×64 RGB PNG images for each window record.
Three image types: line, candlestick, GAF.
All images: no axes, no titles, no whitespace padding.
Named: {ticker}_{window_end_YYYYMMDD}_{split}.png
"""

import io
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf
from PIL import Image
from pathlib import Path
from typing import Literal

# GAF
from pyts.image import GramianAngularField

from .windows import WindowRecord

SIZE = 64  # output image px
# image_size must be ≤ n_timestamps (60); resize to SIZE=64 after transform
_GAF = GramianAngularField(image_size=60, method="summation")

ImageType = Literal["line", "candlestick", "gaf"]


def _img_name(w: WindowRecord) -> str:
    date_str = w.window_end.strftime("%Y%m%d")
    return f"{w.ticker}_{date_str}_{w.split}.png"


# ── Line plot ──────────────────────────────────────────────────────────────────

def _render_line(close_norm: np.ndarray) -> Image.Image:
    """Z-score normalized close, last 5 days shaded grey (empty framing)."""
    fig, ax = plt.subplots(figsize=(1, 1), dpi=SIZE)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.axis("off")

    n = len(close_norm)
    x = np.arange(n)

    ax.plot(x[:-5], close_norm[:-5], color="black", linewidth=0.6, solid_capstyle="round")
    # Grey shaded zone for last 5 days (future zone, no data plotted there)
    ax.axvspan(n - 5 - 0.5, n - 0.5, alpha=0.15, color="grey", linewidth=0)

    y_min, y_max = close_norm.min(), close_norm.max()
    pad = max((y_max - y_min) * 0.05, 0.1)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(y_min - pad, y_max + pad)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=SIZE, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
    return img


# ── Candlestick ────────────────────────────────────────────────────────────────

_CANDLE_STYLE = mpf.make_mpf_style(
    base_mpf_style="classic",
    rc={"axes.labelsize": 0, "xtick.labelsize": 0, "ytick.labelsize": 0},
)


def _render_candlestick(ohlcv: np.ndarray, dates: pd.DatetimeIndex) -> Image.Image:
    df = pd.DataFrame(
        ohlcv,
        index=dates,
        columns=["Open", "High", "Low", "Close", "Volume"],
    )

    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mpf.plot(
            df,
            type="candle",
            style=_CANDLE_STYLE,
            axisoff=True,
            tight_layout=True,
            volume=False,
            figsize=(1, 1),
            savefig=dict(fname=buf, format="png", dpi=SIZE, bbox_inches="tight", pad_inches=0),
        )
    buf.seek(0)
    img = Image.open(buf).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
    return img


# ── GAF ───────────────────────────────────────────────────────────────────────

def _render_gaf(close_norm: np.ndarray) -> Image.Image:
    # pyts expects shape (n_samples, n_timestamps)
    x = close_norm.reshape(1, -1)
    gaf_arr = _GAF.fit_transform(x)[0]  # shape (SIZE, SIZE), values in [-1, 1]

    # Map [-1, 1] → [0, 255]
    arr_uint8 = ((gaf_arr + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    # Grayscale → RGB
    img_gray = Image.fromarray(arr_uint8, mode="L").resize((SIZE, SIZE), Image.LANCZOS)
    img = img_gray.convert("RGB")
    return img


# ── Dispatcher ────────────────────────────────────────────────────────────────

def render_image(w: WindowRecord, img_type: ImageType) -> Image.Image:
    if img_type == "line":
        return _render_line(w.close_norm)
    elif img_type == "candlestick":
        # Reconstruct a DatetimeIndex for the window (business-day freq as proxy)
        idx = pd.bdate_range(end=w.window_end, periods=60)
        return _render_candlestick(w.ohlcv, idx)
    elif img_type == "gaf":
        return _render_gaf(w.close_norm)
    else:
        raise ValueError(f"Unknown image type: {img_type}")


def generate_all(
    windows: list[WindowRecord],
    output_root: Path,
    img_type: ImageType,
) -> pd.DataFrame:
    """
    Render and save all images of a given type.
    Returns manifest DataFrame: [ticker, window_end, split, label_binary,
                                  label_direction, r1..r5, image_path].
    """
    out_dir = output_root / img_type
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for w in windows:
        img_path = out_dir / _img_name(w)
        if not img_path.exists():
            img = render_image(w, img_type)
            img.save(img_path)

        row = {
            "ticker": w.ticker,
            "window_start": w.window_start,
            "window_end": w.window_end,
            "split": w.split,
            "label_binary": w.label_binary,
            "label_direction": w.label_direction,
            "image_path": str(img_path),
        }
        for i, r in enumerate(w.returns, 1):
            row[f"r{i}"] = float(r)
        rows.append(row)

    return pd.DataFrame(rows)
