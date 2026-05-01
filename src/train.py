"""
Generic training loop shared by all 6 model configurations.
Supports both classification and regression tasks.
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    task: str,
) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device) if task == "regression" else y.to(device, dtype=torch.long)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    task: str,
) -> float:
    model.eval()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device) if task == "regression" else y.to(device, dtype=torch.long)
        out = model(x)
        loss = loss_fn(out, y)
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    lr: float,
    task: str,
    device: torch.device,
    epochs: int = 30,
    patience: int = 5,
    checkpoint_path: Path | None = None,
) -> dict:
    """
    Train with Adam + ReduceLROnPlateau + early stopping.
    Returns history dict with train_loss, val_loss lists and best_epoch.
    Saves best checkpoint if checkpoint_path provided.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    history = {"train_loss": [], "val_loss": [], "best_epoch": 0}
    best_val = float("inf")
    no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, task)
        vl_loss = eval_epoch(model, val_loader, loss_fn, device, task)
        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}  train={tr_loss:.5f}  val={vl_loss:.5f}  ({elapsed:.1f}s)")

        if vl_loss < best_val:
            best_val = vl_loss
            history["best_epoch"] = epoch
            no_improve = 0
            if checkpoint_path:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # Reload best weights
    if checkpoint_path and checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    return history
