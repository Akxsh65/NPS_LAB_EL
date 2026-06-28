"""
One-shot clean TEST evaluation to finalize Phase 2 (before Phase 3).

Loads architecture hyperparameters from the training config JSON next to the
checkpoint (required for transformer_masked d_model=160, etc.).

Usage (from repo root or phase4/):
  python eval_clean_test.py \\
    --checkpoint ../phase2/artifacts/refine/architecture/run_masked_d160/transformer_masked_best_acc.pt \\
    --config ../phase2/artifacts/refine/architecture/run_masked_d160/transformer_masked_config.json \\
    --test-pt ../phase1/artifacts/test_tensors.pt \\
    --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

PHASE1_DIR = Path(__file__).resolve().parents[1] / "phase1"
PHASE2_DIR = Path(__file__).resolve().parents[1] / "phase2"
for p in (PHASE1_DIR, PHASE2_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from models import build_model  # noqa: E402


def load_model_from_checkpoint(checkpoint_path: str, config_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    model_name = ckpt.get("model_name", cfg.get("model", "transformer_masked"))
    num_classes = int(ckpt["num_classes"])

    model = build_model(
        model_name,
        num_classes=num_classes,
        d_model=int(cfg.get("d_model", 128)),
        nhead=int(cfg.get("nhead", 8)),
        num_layers=int(cfg.get("num_layers", 4)),
        ff_dim=int(cfg.get("ff_dim", 256)),
        dropout=float(cfg.get("dropout", 0.2)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt, cfg


@torch.no_grad()
def evaluate_test(
    model: torch.nn.Module,
    test_pt: str,
    device: torch.device,
    batch_size: int = 1024,
    num_workers: int = 4,
) -> dict:
    data = torch.load(test_pt, map_location="cpu", weights_only=False)
    X, y = data["X"], data["y"].long()

    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_preds, all_labels = [], []
    for xb, yb in tqdm(loader, desc="test", leave=False):
        xb = xb.to(device, non_blocking=True)
        preds = model(xb).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(yb.numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    n_classes = int(y_true.max()) + 1
    chance = 1.0 / n_classes

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "num_samples": int(len(y_true)),
        "num_classes": n_classes,
        "random_chance_accuracy": chance,
        "classification_report": classification_report(
            y_true, y_pred, zero_division=0, digits=4
        ),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean test-set eval (Phase 2 finalize)")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to *_best_acc.pt from Phase 2",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Training config JSON (default: <checkpoint_dir>/<model>_config.json)",
    )
    parser.add_argument(
        "--test-pt",
        default=str(PHASE1_DIR / "artifacts" / "test_tensors.pt"),
    )
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "results" / "phase2_finalize"),
    )
    parser.add_argument("--save-report", action="store_true", help="Write per-class report .txt")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if args.config:
        config_path = Path(args.config).resolve()
    else:
        model_stem = ckpt_path.stem.replace("_best_acc", "").replace("_best_loss", "")
        config_path = ckpt_path.parent / f"{model_stem}_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Config not found: {config_path}\n"
            "Pass --config explicitly (e.g. transformer_masked_config.json)."
        )

    test_pt = Path(args.test_pt).resolve()
    if not test_pt.is_file():
        raise FileNotFoundError(f"Test tensors not found: {test_pt}")

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Config: {config_path}")
    print(f"Test: {test_pt}")

    model, ckpt, cfg = load_model_from_checkpoint(str(ckpt_path), str(config_path), device)
    metrics = evaluate_test(
        model, str(test_pt), device, args.batch_size, args.num_workers
    )

    val_acc = ckpt.get("best_val_acc")
    val_ep = ckpt.get("epoch")

    os.makedirs(args.out_dir, exist_ok=True)
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(ckpt_path),
        "config": str(config_path),
        "test_pt": str(test_pt),
        "device": str(device),
        "model_name": ckpt.get("model_name"),
        "d_model": cfg.get("d_model"),
        "best_val_acc_at_train": val_acc,
        "best_val_epoch_at_train": val_ep,
        "test_accuracy": metrics["accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "test_weighted_f1": metrics["weighted_f1"],
        "num_samples": metrics["num_samples"],
        "num_classes": metrics["num_classes"],
        "random_chance_accuracy": metrics["random_chance_accuracy"],
    }
    if val_acc is not None:
        summary["val_test_acc_gap"] = float(val_acc) - metrics["accuracy"]

    out_json = os.path.join(args.out_dir, "clean_test_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 55)
    print("PHASE 2 — CLEAN TEST RESULTS (held-out week)")
    print("=" * 55)
    if val_acc is not None:
        print(f"  Val acc @ train (best ckpt): {val_acc:.4f} ({val_acc*100:.2f}%)  epoch={val_ep}")
    print(f"  Test accuracy:               {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Test macro F1:               {metrics['macro_f1']:.4f}")
    print(f"  Test weighted F1:            {metrics['weighted_f1']:.4f}")
    print(f"  Random chance (1/C):         {metrics['random_chance_accuracy']:.4f}")
    if val_acc is not None:
        gap = summary["val_test_acc_gap"]
        print(f"  Val − test acc gap:          {gap:.4f} ({gap*100:.2f} pp)")
    print(f"  Samples / classes:           {metrics['num_samples']} / {metrics['num_classes']}")
    print(f"  Saved: {out_json}")

    if args.save_report:
        report_path = os.path.join(args.out_dir, "clean_test_classification_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(metrics["classification_report"])
        print(f"  Per-class report: {report_path}")

    print("=" * 55)
    print("If val–test gap is modest (~2–5 pp), proceed to Phase 3.")


if __name__ == "__main__":
    main()
