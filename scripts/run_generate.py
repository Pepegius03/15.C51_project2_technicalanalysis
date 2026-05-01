"""
Entry point: generate all images and write manifest CSVs.

Usage:
    python -m scripts.run_generate [--workers N]

Generates images/{line,candlestick,gaf}/ and manifests/{line,candlestick,gaf}_manifest.csv
"""

import argparse
import sys
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocess import load
from src.windows import extract_windows, WindowRecord
from src.generate_images import generate_all, ImageType


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "bbg_raw_data.csv"))
    parser.add_argument("--images-dir", default=str(ROOT / "images"))
    parser.add_argument("--manifests-dir", default=str(ROOT / "manifests"))
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    manifests_dir = Path(args.manifests_dir)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and parsing CSV…")
    stocks = load(Path(args.data))
    print(f"  {len(stocks)} tickers loaded")

    print("Extracting sliding windows…")
    all_windows: list[WindowRecord] = []
    for ticker, df in tqdm(stocks.items(), desc="tickers"):
        all_windows.extend(extract_windows(ticker, df))
    print(f"  {len(all_windows):,} total windows")

    for img_type in ("line", "candlestick", "gaf"):
        print(f"\nGenerating {img_type} images…")
        manifest = generate_all(all_windows, images_dir, img_type)  # type: ignore[arg-type]
        out_csv = manifests_dir / f"{img_type}_manifest.csv"
        manifest.to_csv(out_csv, index=False)
        print(f"  Saved {len(manifest):,} entries → {out_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
