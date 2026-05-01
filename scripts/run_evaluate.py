"""
Entry point: evaluate all 6 trained model configurations on the test split.

Usage:
    python -m scripts.run_evaluate [--split test] [--batch-size 64]

Outputs:
  - Summary table printed to stdout
  - Per-stock accuracy CSVs saved to results/per_stock_config{N}.csv
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from torch.utils.data import DataLoader

from src.models import build_model, config_meta
from src.dataset import ClassificationDataset, RegressionDataset, compute_dataset_stats
from src.evaluate import evaluate_model
from src.per_stock_analysis import per_ticker_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--manifests-dir", default=str(ROOT / "manifests"))
    parser.add_argument("--checkpoints-dir", default=str(ROOT / "checkpoints"))
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for cfg_id in range(1, 7):
        meta = config_meta(cfg_id)
        img_type = meta["image"]
        task = meta["task"]
        pretrained = meta["strategy"] == "pretrained"

        manifest_path = Path(args.manifests_dir) / f"{img_type}_manifest.csv"
        if not manifest_path.exists():
            print(f"[config {cfg_id}] manifest missing, skipping")
            continue

        ckpt_path = Path(args.checkpoints_dir) / f"config{cfg_id}_{img_type}_{meta['strategy']}.pt"
        if not ckpt_path.exists():
            print(f"[config {cfg_id}] checkpoint missing, skipping")
            continue

        manifest = pd.read_csv(manifest_path)
        DatasetCls = RegressionDataset if task == "regression" else ClassificationDataset

        norm_stats = None
        if not pretrained:
            norm_stats = compute_dataset_stats(manifest, split="train")

        test_ds = DatasetCls(manifest, args.split, pretrained=pretrained, normalize_stats=norm_stats)
        loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, persistent_workers=True)

        model, _, _ = build_model(cfg_id)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model = model.to(device)
        model.eval()

        metrics = evaluate_model(model, loader, device, task)
        metrics["config"] = cfg_id
        metrics["image"] = img_type
        metrics["strategy"] = meta["strategy"]
        metrics["task"] = task
        summary_rows.append(metrics)
        print(f"[config {cfg_id}] {metrics}")

        # Per-stock accuracy
        per_stock_df = per_ticker_accuracy(
            model, manifest, DatasetCls, args.split, task, device, pretrained, norm_stats,
            batch_size=args.batch_size,
        )
        per_stock_path = results_dir / f"per_stock_config{cfg_id}.csv"
        per_stock_df.to_csv(per_stock_path, index=False)
        print(f"  Per-stock saved: {per_stock_path}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = results_dir / f"summary_{args.split}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved: {summary_path}")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
