"""
Visualise model predictions on a random test sample.
Automatically handles classification (configs 3-6) and regression (configs 1-2).

Usage:
    python scripts/peek.py --config 1
    python scripts/peek.py --config 3 --n 20
"""

import argparse
import math
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from models import build_model, config_meta
from dataset import _IMAGENET_MEAN, _IMAGENET_STD, SIZE

CHECKPOINT_DIR = ROOT / "checkpoints"
MANIFEST_DIR   = ROOT / "manifests"
OUTPUT_DIR     = ROOT / "results" / "peek"

MANIFEST_MAP = {
    "line":        "line_manifest.csv",
    "candlestick": "candlestick_manifest.csv",
    "gaf":         "gaf_manifest.csv",
}


def load_image(path, mean, std):
    img = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(((arr - mean) / std).transpose(2, 0, 1))
    return img, tensor


def peek_regression(model, sample, mean, std, config_id, strategy):
    n_cols = 4
    n_rows = math.ceil(len(sample) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(n_cols * 5, n_rows * 2.8))
    axes = np.array(axes).reshape(n_rows, n_cols * 2)
    correct = 0

    with torch.no_grad():
        for i, (_, row) in enumerate(sample.iterrows()):
            r, c = divmod(i, n_cols)
            ax_img = axes[r, c * 2]
            ax_bar = axes[r, c * 2 + 1]

            pil_img, tensor = load_image(row["image_path"], mean, std)
            pred = float(model(tensor.unsqueeze(0)).squeeze())
            true_rets = np.array([row[f"r{j}"] for j in range(1, 6)], dtype=np.float32)
            true = float(np.prod(1 + true_rets) - 1)
            hit  = (pred > 0) == (true > 0)
            correct += hit

            ax_img.imshow(pil_img)
            ax_img.axis("off")
            ax_img.set_title(f"{row['ticker']}  {row['window_end']}", fontsize=7, pad=2)

            ax_bar.bar(["pred", "actual"], [pred * 100, true * 100],
                       color=["tomato", "steelblue"], width=0.5)
            ax_bar.axhline(0, color="gray", lw=0.8)
            ax_bar.set_ylabel("%", fontsize=6)
            ax_bar.tick_params(labelsize=6)
            ax_bar.set_title("✓" if hit else "✗", fontsize=9,
                              color="green" if hit else "red", pad=2)

    for i in range(len(sample), n_rows * n_cols):
        r, c = divmod(i, n_cols)
        axes[r, c * 2].axis("off")
        axes[r, c * 2 + 1].axis("off")

    acc = correct / len(sample)
    fig.suptitle(
        f"Config {config_id}: line | {strategy} | regression  —  "
        f"directional acc: {acc:.0%} ({correct}/{len(sample)})",
        fontsize=10,
    )
    return fig


def peek_classification(model, sample, mean, std, config_id, img_type, strategy):
    n_cols = 4
    n_rows = math.ceil(len(sample) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3.4))
    axes = np.array(axes).flatten()
    correct = 0

    with torch.no_grad():
        for i, (_, row) in enumerate(sample.iterrows()):
            pil_img, tensor = load_image(row["image_path"], mean, std)
            probs = torch.softmax(model(tensor.unsqueeze(0)), dim=1)[0]
            pred  = int(probs.argmax())
            conf  = float(probs[pred])
            true  = int(row["label_binary"])
            hit   = pred == true
            correct += hit

            ax = axes[i]
            ax.imshow(pil_img)
            ax.axis("off")
            pred_lbl = "UP" if pred == 1 else "DOWN"
            true_lbl = "UP" if true == 1 else "DOWN"
            ax.set_title(
                f"{row['ticker']}  {row['window_end']}\npred={pred_lbl} ({conf:.0%})  true={true_lbl}",
                fontsize=7, color="green" if hit else "red", pad=3,
            )

    for ax in axes[len(sample):]:
        ax.axis("off")

    acc = correct / len(sample)
    fig.suptitle(
        f"Config {config_id}: {img_type} | {strategy} | classification  —  "
        f"acc: {acc:.0%} ({correct}/{len(sample)})",
        fontsize=10, y=1.01,
    )
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=int, required=True, choices=range(1, 7))
    parser.add_argument("--n",      type=int, default=16)
    args = parser.parse_args()

    meta     = config_meta(args.config)
    task     = meta["task"]
    img_type = meta["image"]
    strategy = meta["strategy"]

    ckpt_path = CHECKPOINT_DIR / f"config{args.config}_{img_type}_{strategy}.pt"
    if not ckpt_path.exists():
        sys.exit(f"Checkpoint not found: {ckpt_path}")

    manifest = pd.read_csv(MANIFEST_DIR / MANIFEST_MAP[img_type])
    test_df  = manifest[manifest["split"] == "test"].reset_index(drop=True)
    sample   = test_df.sample(n=min(args.n, len(test_df)), random_state=42).reset_index(drop=True)

    model, _, _ = build_model(args.config)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.eval()

    mean = _IMAGENET_MEAN if strategy == "pretrained" else np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std  = _IMAGENET_STD  if strategy == "pretrained" else np.array([0.5, 0.5, 0.5], dtype=np.float32)

    if task == "regression":
        fig = peek_regression(model, sample, mean, std, args.config, strategy)
    else:
        fig = peek_classification(model, sample, mean, std, args.config, img_type, strategy)

    plt.tight_layout()
    out_path = OUTPUT_DIR / f"config{args.config}_{img_type}_{strategy}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
