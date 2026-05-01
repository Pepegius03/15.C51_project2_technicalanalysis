"""
Entry point: train one model configuration.

Usage:
    python -m scripts.run_train --config 3 [--epochs 30] [--batch-size 64] [--no-cuda]

Config IDs:
  1  line        scratch     regression
  2  line        pretrained  regression
  3  candlestick scratch     classification
  4  candlestick pretrained  classification
  5  gaf         scratch     classification
  6  gaf         pretrained  classification
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
from src.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=int, required=True, choices=range(1, 7))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--manifests-dir", default=str(ROOT / "manifests"))
    parser.add_argument("--checkpoints-dir", default=str(ROOT / "checkpoints"))
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    meta = config_meta(args.config)
    img_type = meta["image"]
    task = meta["task"]
    pretrained = meta["strategy"] == "pretrained"

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    print(f"Config {args.config}: {img_type} | {meta['strategy']} | {task}  →  {device}")

    manifest_path = Path(args.manifests_dir) / f"{img_type}_manifest.csv"
    if not manifest_path.exists():
        sys.exit(f"Manifest not found: {manifest_path}\nRun scripts/run_generate.py first.")

    manifest = pd.read_csv(manifest_path)

    DatasetCls = RegressionDataset if task == "regression" else ClassificationDataset

    # Compute dataset stats for scratch models
    norm_stats = None
    if not pretrained:
        print("Computing dataset normalization statistics…")
        norm_stats = compute_dataset_stats(manifest, split="train")

    train_ds = DatasetCls(manifest, "train", pretrained=pretrained, normalize_stats=norm_stats)
    val_ds   = DatasetCls(manifest, "val",   pretrained=pretrained, normalize_stats=norm_stats)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    print(f"  train={len(train_ds):,}  val={len(val_ds):,}")

    model, loss_fn, lr = build_model(args.config)
    model = model.to(device)

    ckpt_dir = Path(args.checkpoints_dir)
    ckpt_path = ckpt_dir / f"config{args.config}_{img_type}_{meta['strategy']}.pt"

    history = train(
        model, train_loader, val_loader, loss_fn, lr,
        task=task, device=device,
        epochs=args.epochs, patience=args.patience,
        checkpoint_path=ckpt_path,
    )

    print(f"\nBest epoch: {history['best_epoch']}  best val loss: {min(history['val_loss']):.5f}")
    print(f"Checkpoint saved: {ckpt_path}")


if __name__ == "__main__":
    main()
