"""
Phase 4 — Evaluate frozen classifier on test / obfuscated tensors.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

PHASE1_DIR = Path(__file__).resolve().parents[1] / "phase1"
PHASE2_DIR = Path(__file__).resolve().parents[1] / "phase2"
for p in (PHASE2_DIR, PHASE1_DIR):
    if str(p) not in sys.path:
        sys.path.append(str(p))

from models import build_model  # noqa: E402

from config import (  # noqa: E402
    BATCH_SIZE,
    CNN_BILSTM_ALT_CHECKPOINT,
    CNN_BILSTM_ALT_CONFIG,
    CNN_BILSTM_CHECKPOINT,
    CNN_BILSTM_CONFIG,
    DEFAULT_CHECKPOINT,
    DEFAULT_MODEL_CONFIG,
    FALLBACK_CHECKPOINT,
    LABEL_ENCODER,
    MASKED_D160_CHECKPOINT,
    MASKED_D160_CONFIG,
    MODEL_NAME,
    NUM_WORKERS,
    PHASE4_RESULTS,
    PREDICTIONS_DIR,
)


def resolve_checkpoint(path: Optional[str] = None, attack_model: str = "transformer") -> str:
    def _ckpt_model_name(p: str) -> str:
        meta = torch.load(p, map_location="cpu", weights_only=False)
        return str(meta.get("model_name", "")).lower()

    if path and os.path.isfile(path):
        name = _ckpt_model_name(path)
        if attack_model == "cnn_bilstm":
            if name and "bilstm" not in name and "cnn" not in name:
                print(
                    f"WARNING: --checkpoint {path} is model '{name}', not cnn_bilstm; "
                    "using default BiLSTM checkpoint instead."
                )
            else:
                return path
        elif name and "bilstm" in name:
            print(
                f"WARNING: --checkpoint {path} is cnn_bilstm but attack_model={attack_model}; "
                "using default Transformer checkpoint instead."
            )
        else:
            return path
    if attack_model == "cnn_bilstm":
        for candidate in (CNN_BILSTM_CHECKPOINT, CNN_BILSTM_ALT_CHECKPOINT):
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(
            "No CNN-BiLSTM checkpoint found. Expected one of:\n"
            f"  {CNN_BILSTM_CHECKPOINT}\n"
            f"  {CNN_BILSTM_ALT_CHECKPOINT}\n"
            "Pass --checkpoint explicitly."
        )
    for candidate in (
        DEFAULT_CHECKPOINT,
        MASKED_D160_CHECKPOINT,
        FALLBACK_CHECKPOINT,
    ):
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "No checkpoint found. Expected one of:\n"
        f"  {DEFAULT_CHECKPOINT}\n"
        f"  {MASKED_D160_CHECKPOINT}\n"
        f"  {FALLBACK_CHECKPOINT}\n"
        "Pass --checkpoint explicitly."
    )


def infer_d_model_from_checkpoint(ckpt: dict) -> Optional[int]:
    """Read d_model from input_proj.weight in saved state_dict."""
    sd = ckpt.get("state_dict", ckpt)
    w = sd.get("input_proj.weight")
    if w is not None:
        return int(w.shape[0])
    return None


def _config_candidates_for_checkpoint(ckpt: Path, model_name: str) -> list[Path]:
    stem = ckpt.stem.replace("_best_acc", "").replace("_best_loss", "").replace(
        "_production", "_masked"
    )
    names = [f"{stem}_config.json"]
    if model_name == "cnn_bilstm":
        names.append("cnn_bilstm_config.json")
    elif "masked" in model_name or "production" in ckpt.stem:
        names.append("transformer_masked_config.json")
    else:
        names.extend(["transformer_masked_config.json", "transformer_config.json"])

    candidates: list[Path] = []
    if model_name != "cnn_bilstm":
        # Prefer refined masked-d160 config before generic d128 transformer_config.json
        candidates.extend([Path(MASKED_D160_CONFIG), Path(DEFAULT_MODEL_CONFIG)])
    for n in names:
        candidates.append(ckpt.parent / n)
    if model_name == "cnn_bilstm":
        candidates.extend([Path(CNN_BILSTM_CONFIG), Path(CNN_BILSTM_ALT_CONFIG)])

    seen: set[str] = set()
    unique: list[Path] = []
    for c in candidates:
        key = str(c.resolve()) if c.is_absolute() else str(c)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def _config_d_model(cfg_path: Path) -> Optional[int]:
    try:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        if "d_model" in cfg:
            return int(cfg["d_model"])
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    return None


def resolve_model_config(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    attack_model: Optional[str] = None,
) -> Optional[str]:
    if config_path and os.path.isfile(config_path):
        return os.path.abspath(config_path)

    ckpt = Path(checkpoint_path)
    ckpt_meta = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_name = str(ckpt_meta.get("model_name", attack_model or MODEL_NAME)).lower()
    expected_d = infer_d_model_from_checkpoint(ckpt_meta)

    candidates = _config_candidates_for_checkpoint(ckpt, model_name)
    fallback: Optional[str] = None

    for c in candidates:
        if not c.is_file():
            continue
        resolved = str(c.resolve())
        if fallback is None:
            fallback = resolved
        if expected_d is not None:
            cfg_d = _config_d_model(c)
            if cfg_d is not None and cfg_d != expected_d:
                continue
        return resolved

    if fallback:
        return fallback

    if model_name == "cnn_bilstm":
        return None

    raise FileNotFoundError(
        f"No training config JSON found for checkpoint {checkpoint_path}.\n"
        f"Pass --config (e.g. {MASKED_D160_CONFIG})."
    )


def load_model(
    checkpoint_path: str,
    device: torch.device,
    config_path: Optional[str] = None,
    attack_model: Optional[str] = None,
) -> Tuple[nn.Module, dict, dict]:
    """Load checkpoint + architecture hyperparameters from training config JSON."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = str(ckpt.get("model_name", attack_model or MODEL_NAME)).lower()
    cfg_path = resolve_model_config(checkpoint_path, config_path, attack_model=model_name)
    cfg: dict = {}
    if cfg_path:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)

    num_classes = int(ckpt["num_classes"])

    model = build_model(
        ckpt.get("model_name", cfg.get("model", model_name)),
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


def apply_channel_mask(X: torch.Tensor, active_channels: Optional[List[int]]) -> torch.Tensor:
    """Zero out channels not in active_channels (IPT=0, DIR=1, SIZE=2)."""
    if not active_channels or active_channels == [0, 1, 2]:
        return X
    out = torch.zeros_like(X)
    for ch in active_channels:
        out[:, ch, :] = X[:, ch, :]
    return out


def save_predictions_npz(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    experiment: str,
    out_dir: str,
    extra: Optional[dict] = None,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{experiment}.npz")
    payload = {"y_true": y_true, "y_pred": y_pred}
    if extra:
        payload.update(extra)
    np.savez_compressed(path, **payload)
    return path


def load_predictions_npz(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {"y_true": data["y_true"], "y_pred": data["y_pred"]}


@torch.no_grad()
def predict(
    model: nn.Module,
    pt_path: str,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    active_channels: Optional[List[int]] = None,
) -> Dict:
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    X, y = data["X"], data["y"].long()

    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_preds: List[int] = []
    all_labels: List[int] = []

    for xb, yb in tqdm(loader, desc=f"eval {os.path.basename(pt_path)}", leave=False):
        xb = apply_channel_mask(xb, active_channels)
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(yb.numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    n_classes = int(y_true.max()) + 1 if len(y_true) else 1

    return {
        "dataset": pt_path,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "num_samples": int(len(y_true)),
        "num_classes": n_classes,
        "random_chance_accuracy": 1.0 / max(n_classes, 1),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def load_class_names(label_encoder_path: Optional[str] = None) -> Optional[List[str]]:
    """Load app names from Phase 1 label_encoder.pkl."""
    path = label_encoder_path or LABEL_ENCODER
    if not os.path.isfile(path):
        return None
    import joblib

    le = joblib.load(path)
    return [str(c) for c in le.classes_]


def save_per_class_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    experiment: str,
    out_dir: str,
    label_encoder_path: Optional[str] = None,
) -> Dict[str, str]:
    """
    Write classification_report.txt and per-class CSV (precision/recall/F1/support).
    """
    os.makedirs(out_dir, exist_ok=True)
    class_names = load_class_names(label_encoder_path)
    labels = list(range(int(max(y_true.max(), y_pred.max())) + 1))

    report_txt = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names if class_names and len(class_names) == len(labels) else None,
        zero_division=0,
        digits=4,
    )
    txt_path = os.path.join(out_dir, f"{experiment}_classification_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Experiment: {experiment}\n")
        f.write(f"Samples: {len(y_true)}\n\n")
        f.write(report_txt)

    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    rows = []
    for i, lab in enumerate(labels):
        name = class_names[lab] if class_names and lab < len(class_names) else str(lab)
        rows.append(
            {
                "class_id": lab,
                "class_name": name,
                "support": int(sup[i]),
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f1[i]),
            }
        )
    csv_path = os.path.join(out_dir, f"{experiment}_per_class.csv")
    import pandas as pd

    pd.DataFrame(rows).sort_values("f1", ascending=True).to_csv(csv_path, index=False)

    return {"classification_report": txt_path, "per_class_csv": csv_path}


def top_class_indices_by_support(
    y_true: np.ndarray,
    n_top: int = 20,
) -> np.ndarray:
    """Class ids with highest test support (for readable CM inset)."""
    labels, counts = np.unique(y_true, return_counts=True)
    order = np.argsort(-counts)
    return labels[order[: min(n_top, len(labels))]]


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion matrix",
    top_n_inset: int = 20,
) -> str:
    """
    Full 64x64 heatmap (no tick labels) + inset of top-N classes by support with names.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    cm = confusion_matrix(y_true, y_pred)
    top_idx = top_class_indices_by_support(y_true, n_top=top_n_inset)
    cm_sub = cm[np.ix_(top_idx, top_idx)]

    def _names(idxs: np.ndarray) -> List[str]:
        if class_names and len(class_names) >= int(idxs.max()) + 1:
            return [str(class_names[int(i)])[:24] for i in idxs]
        return [str(int(i)) for i in idxs]

    sub_names = _names(top_idx)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", square=False, ax=ax, cbar=True, xticklabels=False, yticklabels=False)
    ax.set_title(title)
    ax.set_ylabel("True class")
    ax.set_xlabel("Predicted class")

    inset = inset_axes(ax, width="42%", height="42%", loc="upper right", borderpad=2)
    sns.heatmap(
        cm_sub,
        cmap="Blues",
        square=True,
        ax=inset,
        cbar=False,
        xticklabels=sub_names,
        yticklabels=sub_names,
    )
    inset.set_title(f"Top {len(top_idx)} classes (by support)", fontsize=8)
    inset.tick_params(axis="both", labelsize=6, rotation=45)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def evaluate_one(
    pt_path: str,
    checkpoint: Optional[str] = None,
    config: Optional[str] = None,
    device: Optional[str] = None,
    experiment_name: Optional[str] = None,
    attack_model: str = "transformer",
    active_channels: Optional[List[int]] = None,
    save_cm: bool = False,
    save_report: bool = False,
    save_predictions: bool = False,
    predictions_dir: Optional[str] = None,
    report_dir: Optional[str] = None,
    out_dir: str = PHASE4_RESULTS,
    batch_size: int = BATCH_SIZE,
) -> Dict:
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_path = resolve_checkpoint(checkpoint, attack_model=attack_model)
    cfg_path = resolve_model_config(ckpt_path, config, attack_model=attack_model)
    model, ckpt, cfg = load_model(ckpt_path, device_t, cfg_path, attack_model=attack_model)

    result = predict(
        model,
        pt_path,
        device_t,
        batch_size=batch_size,
        active_channels=active_channels,
    )
    result["checkpoint"] = ckpt_path
    result["config"] = cfg_path
    result["model_name"] = ckpt.get("model_name", cfg.get("model"))
    result["d_model"] = int(cfg.get("d_model", 128)) if cfg else None
    result["attack_model"] = attack_model
    if active_channels is not None:
        result["active_channels"] = active_channels

    y_true = result.pop("y_true")
    y_pred = result.pop("y_pred")
    experiment = experiment_name or Path(pt_path).stem

    if save_predictions:
        pdir = predictions_dir or PREDICTIONS_DIR
        extra = {"attack_model": attack_model}
        if active_channels is not None:
            extra["active_channels"] = np.array(active_channels, dtype=np.int8)
        result["predictions_npz"] = save_predictions_npz(
            y_true, y_pred, experiment, pdir, extra=extra
        )

    if save_cm:
        os.makedirs(out_dir, exist_ok=True)
        cm_path = os.path.join(out_dir, f"confusion_{experiment}.png")
        names = load_class_names()
        title = f"Confusion matrix — {experiment.replace('obfuscated_', '')}"
        save_confusion_matrix(y_true, y_pred, cm_path, class_names=names, title=title)
        result["confusion_matrix"] = cm_path

    if save_report:
        rdir = report_dir or os.path.join(out_dir, "reports")
        result["per_class_reports"] = save_per_class_report(
            y_true, y_pred, experiment, rdir
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one tensor file")
    parser.add_argument("--pt", required=True, help="Path to .pt with keys X, y")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None, help="Training config JSON (d_model, etc.)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--save-cm", action="store_true")
    parser.add_argument("--save-report", action="store_true")
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--out-dir", default=PHASE4_RESULTS)
    args = parser.parse_args()

    metrics = evaluate_one(
        args.pt,
        checkpoint=args.checkpoint,
        config=args.config,
        device=args.device,
        save_cm=args.save_cm,
        save_report=args.save_report,
        report_dir=args.report_dir,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
