"""
Parse bbg_raw_data.csv into per-stock DataFrames.

CSV layout:
  Row 0 : ticker row  — col 0 = ÿ (BOM artifact), remaining: "AAPL US Equity" × 5, etc.
  Row 1 : field row   — col 0 = "DATES", remaining: "px_open(...)", "px_high(...)", …
  Row 2+: data rows   — col 0 = date "M/D/YYYY", rest = float values or #N/A

Each ticker occupies exactly 5 consecutive columns in order: open, high, low, close, volume.
File encoding: latin-1.
"""

import numpy as np
import pandas as pd
from pathlib import Path

_FIELDS = ["open", "high", "low", "close", "volume"]
_NA_STR = "#N/A"
_ENCODING = "latin-1"

DATA_PATH = Path(__file__).parent.parent / "bbg_raw_data.csv"


def load(csv_path: Path = DATA_PATH) -> dict[str, pd.DataFrame]:
    """
    Returns dict: short_ticker (e.g. 'AAPL') → DataFrame
    columns [open, high, low, close, volume], DatetimeIndex of trading days.
    """
    raw = pd.read_csv(csv_path, header=None, dtype=str, encoding=_ENCODING, low_memory=False)

    # Row 0: ticker names ("AAPL US Equity" × 5 per ticker); col 0 is BOM artifact
    # Row 1: field definitions; col 0 = "DATES"
    # Row 2+: data
    ticker_row = raw.iloc[0].tolist()
    data = raw.iloc[2:].reset_index(drop=True)

    # Parse dates from first column
    dates = pd.to_datetime(data.iloc[:, 0], format="%m/%d/%Y", errors="coerce")

    # Group non-date columns into 5-column blocks per ticker
    # ticker_row[1:] has the raw ticker names; same name appears 5 consecutive times
    asset_cols = list(range(1, len(ticker_row)))

    result: dict[str, pd.DataFrame] = {}
    i = 0
    while i + 4 < len(asset_cols):
        col_idxs = [asset_cols[i + j] for j in range(5)]
        ticker_full = str(ticker_row[col_idxs[0]]).strip()
        if not ticker_full or ticker_full.lower() in ("nan", "ÿ", ""):
            i += 5
            continue

        arrays = {}
        for field_idx, field_name in enumerate(_FIELDS):
            col_data = data.iloc[:, col_idxs[field_idx]]
            numeric = pd.to_numeric(col_data.replace(_NA_STR, np.nan), errors="coerce")
            arrays[field_name] = numeric.values

        df = pd.DataFrame(arrays, index=dates)
        df.index.name = "date"
        df = df[df.index.notna()]
        df = df.dropna(how="all")
        df = df.sort_index()

        short = ticker_full.split()[0]
        result[short] = df
        i += 5

    return result
