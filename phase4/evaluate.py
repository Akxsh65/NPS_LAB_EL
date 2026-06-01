"""
Phase 4 — Evaluate frozen Transformer on test / obfuscated tensors.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

PHASE1_DIR = Path(__file__).resolve().parents[1] / "phase1"
PHASE2_DIR = Path(__file__).resolve().parents[1] / "phase2"
for p in (PHASE1_DIR, PHASE2_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dataset import SavedDataset  # noqa: E402
from models import build_model  # noqa: E402

from config import (  # noqa: E402
    BATCH_SIZE,
    DEFAULT_CHECKPOINT,
    FALLBACK_CHECKPOINT,
    LABEL_ENCODER,
    MODEL_NAME,
    NUM_WORKERS,
    PHASE4_RESULTS,
)


def resolve_checkpoint(path: Optional[str] = None) -> str:
    if path and os.path.isfile(path):
        return path
    if os.path.isfile(DEFAULT_CHECKPOINT):
        return DEFAULT_CHECKPOINT
    if os.path.isfile(FALLBACK_CHECKPOINT):
        return FALLBACK_CHECKPOINT
    raise FileNotFoundError(
        "No checkpoint found. Expected one of:\n"
        f"  {DEFAULT_CHECKPOINT}\n"
        f"  {FALLBACK_CHECKPOINT}\n"
        "Pass --checkpoint explicitly."
    )


def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_name = ckpt.get("model_name", MODEL_NAME)
    num_classes = int(ckpt["num_classes"])
    model = build_model(model_name, num_classes=num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict(
    model: nn.Module,
    pt_path: str,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
) -> Dict:
    data = torch.load(pt_path, map_location="cpu")
    X, y = data["X"], data["y"].long()

    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    all_preds: List[int] = []
    all_labels: List[int] = []

    for xb, yb in tqdm(loader, desc=f"eval {os.path.basename(pt_path)}", leave=False):
        xb = xb.to(device)
        logits = model(xb)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(yb.numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    return {
        "dataset": pt_path,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "num_samples": int(len(y_true)),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def save_confusion_matrix(y_true, y_pred, out_path: str, class_names=None) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, cmap="Blues", square=False)
    plt.title("Confusion matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def evaluate_one(
    pt_path: str,
    checkpoint: Optional[str] = None,
    device: Optional[str] = None,
    save_cm: bool = False,
    out_dir: str = PHASE4_RESULTS,
) -> Dict:
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_path = resolve_checkpoint(checkpoint)
    model = load_model(ckpt_path, device_t)

    result = predict(model, pt_path, device_t)
    result["checkpoint"] = ckpt_path

    if save_cm:
        os.makedirs(out_dir, exist_ok=True)
        base = Path(pt_path).stem
        cm_path = os.path.join(out_dir, f"confusion_{base}.png")
        save_confusion_matrix(result.pop("y_true"), result.pop("y_pred"), cm_path)
        result["confusion_matrix"] = cm_path
    else:
        result.pop("y_true")
        result.pop("y_pred")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one tensor file")
    parser.add_argument("--pt", required=True, help="Path to .pt with keys X, y")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-cm", action="store_true")
    parser.add_argument("--out-dir", default=PHASE4_RESULTS)
    args = parser.parse_args()

    metrics = evaluate_one(
        args.pt,
        checkpoint=args.checkpoint,
        device=args.device,
        save_cm=args.save_cm,
        out_dir=args.out_dir,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
