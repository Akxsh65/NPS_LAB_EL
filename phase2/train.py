import argparse
import csv
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import build_model


# Allow importing SavedDataset from phase1 without duplicating code.
PHASE1_DIR = Path(__file__).resolve().parents[1] / "phase1"
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))

from dataset import SavedDataset  # noqa: E402


@dataclass
class TrainConfig:
    model: str
    train_pt: str
    val_pt: str
    out_dir: str
    batch_size: int = 1024
    num_workers: int = 4
    pin_memory: bool = True
    epochs: int = 60
    patience: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-2
    t_max: int = 60
    min_lr: float = 1e-6
    amp_dtype: str = "bf16"
    label_smoothing: float = 0.05
    max_grad_norm: float = 1.0
    monitor_metric: str = "val_acc"
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_amp_dtype(dtype_name: str) -> torch.dtype:
    dtype_name = dtype_name.lower()
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    raise ValueError("amp_dtype must be either 'bf16' or 'fp16'")


def build_loaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    train_ds = SavedDataset(cfg.train_pt)
    val_ds = SavedDataset(cfg.val_pt)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.num_workers > 0,
    )
    return train_loader, val_loader


def infer_num_classes(train_pt_path: str) -> int:
    data = torch.load(train_pt_path, map_location="cpu")
    y = data["y"]
    return int(y.max().item()) + 1


def compute_class_weights(train_pt_path: str, num_classes: int) -> torch.Tensor:
    data = torch.load(train_pt_path, map_location="cpu")
    y = data["y"].long()
    counts = torch.bincount(y, minlength=num_classes).float()
    weights = counts.sum() / (counts.clamp_min(1.0) * num_classes)
    return weights


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    amp_dtype: torch.dtype,
    max_grad_norm: float,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)

        if device.type == "cuda":
            scaler.scale(loss).backward()
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in tqdm(loader, desc="val", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast(device_type="cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return running_loss / max(total, 1), correct / max(total, 1)


def save_metrics_csv(csv_path: str, history: list[Dict[str, float]]) -> None:
    if not history:
        return
    fieldnames = list(history[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def run_training(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    device = get_device()
    amp_dtype = get_amp_dtype(cfg.amp_dtype)
    num_classes = infer_num_classes(cfg.train_pt)

    train_loader, val_loader = build_loaders(cfg)
    model = build_model(cfg.model, num_classes=num_classes).to(device)

    class_weights = compute_class_weights(cfg.train_pt, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.t_max, eta_min=cfg.min_lr)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    best_val_acc = -1.0
    epochs_without_improve = 0
    history = []

    ckpt_acc_path = os.path.join(cfg.out_dir, f"{cfg.model}_best_acc.pt")
    ckpt_loss_path = os.path.join(cfg.out_dir, f"{cfg.model}_best_loss.pt")
    log_json_path = os.path.join(cfg.out_dir, f"{cfg.model}_config.json")
    log_csv_path = os.path.join(cfg.out_dir, f"{cfg.model}_history.csv")

    print(f"Device: {device}")
    print(f"Model: {cfg.model}, classes: {num_classes}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(
        f"Monitor: {cfg.monitor_metric} | label_smoothing={cfg.label_smoothing} "
        f"| max_grad_norm={cfg.max_grad_norm}"
    )

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, amp_dtype, cfg.max_grad_norm
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, amp_dtype)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | lr={lr:.2e} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        improved_acc = val_acc > best_val_acc
        improved_loss = val_loss < best_val_loss

        if improved_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_name": cfg.model,
                    "num_classes": num_classes,
                    "state_dict": model.state_dict(),
                    "best_val_loss": val_loss,
                    "best_val_acc": best_val_acc,
                    "epoch": epoch,
                },
                ckpt_acc_path,
            )
            print(f"  Saved new best accuracy checkpoint -> {ckpt_acc_path}")

        if improved_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_name": cfg.model,
                    "num_classes": num_classes,
                    "state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_val_acc": val_acc,
                    "epoch": epoch,
                },
                ckpt_loss_path,
            )
            print(f"  Saved new best loss checkpoint -> {ckpt_loss_path}")

        monitor_improved = improved_acc if cfg.monitor_metric == "val_acc" else improved_loss
        if monitor_improved:
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= cfg.patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    restore_path = ckpt_acc_path if cfg.monitor_metric == "val_acc" else ckpt_loss_path
    if os.path.exists(restore_path):
        best_ckpt = torch.load(restore_path, map_location=device)
        model.load_state_dict(best_ckpt["state_dict"])
        print(
            f"Restored best checkpoint from epoch {best_ckpt.get('epoch', 'unknown')} "
            f"(val_loss={best_ckpt.get('best_val_loss', float('nan')):.4f}, "
            f"val_acc={best_ckpt.get('best_val_acc', float('nan')):.4f})."
        )

    save_metrics_csv(log_csv_path, history)
    with open(log_json_path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best val acc : {best_val_acc:.4f}")
    print(f"Best loss ckpt: {ckpt_loss_path}")
    print(f"Best acc  ckpt: {ckpt_acc_path}")
    print(f"History log: {log_csv_path}")
    print(f"Config log : {log_json_path}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Phase 2 training entrypoint")
    parser.add_argument("--model", type=str, required=True, choices=["cnn_bilstm", "transformer"])
    parser.add_argument("--train-pt", type=str, default="../phase1/artifacts/train_tensors.pt")
    parser.add_argument("--val-pt", type=str, default="../phase1/artifacts/val_tensors.pt")
    parser.add_argument("--out-dir", type=str, default="./artifacts")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.set_defaults(pin_memory=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--t-max", type=int, default=60)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--monitor-metric", type=str, default="val_acc", choices=["val_acc", "val_loss"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return TrainConfig(
        model=args.model,
        train_pt=args.train_pt,
        val_pt=args.val_pt,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        t_max=args.t_max,
        min_lr=args.min_lr,
        amp_dtype=args.amp_dtype,
        label_smoothing=args.label_smoothing,
        max_grad_norm=args.max_grad_norm,
        monitor_metric=args.monitor_metric,
        seed=args.seed,
    )


if __name__ == "__main__":
    config = parse_args()
    run_training(config)
