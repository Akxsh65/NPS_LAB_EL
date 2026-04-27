# feature_engineering.py
# ============================================================
# Phase 1.2 — All normalization logic lives here.
#
# Three channels, three strategies:
#   Ch 0  IPT   → log1p  then Z-score  (stats from TRAIN only)
#   Ch 1  DIR   → already ±1; 0 reserved for padding
#   Ch 2  SIZE  → Min-Max to [0, 1]    (theoretical bounds 0–MTU)
#
# IMPORTANT: fit_ipt_scaler() must be called on train_df ONLY.
# The returned (mean, std) are then passed to every other split.
# ============================================================

import numpy as np
import joblib
import os
from tqdm import tqdm
from config import SEQ_LEN, MTU, IPT_CLIP_MAX, LOG_EPS, ARTIFACTS_DIR


# ── Scaler persistence ───────────────────────────────────────────────────────

def fit_ipt_scaler(train_df) -> tuple[float, float]:
    """
    Computes log1p mean and std from the TRAINING split ONLY.
    Saves result to artifacts/ipt_scaler.pkl so Phase 3 can reuse it.

    NEVER call this on val_df or test_df — that would leak future data.
    """
    print("Fitting IPT scaler on training data ...")
    all_log_ipt = []

    for seq in tqdm(train_df["PPI"], desc="  Scanning IPT sequences"):
        arr = np.asarray(seq[0], dtype=np.float64)
        arr = np.clip(arr, 0.0, IPT_CLIP_MAX)
        all_log_ipt.append(np.log1p(arr))

    all_log_ipt = np.concatenate(all_log_ipt)
    mean = float(all_log_ipt.mean())
    std  = float(all_log_ipt.std())

    scaler = {"mean": mean, "std": std}
    path   = os.path.join(ARTIFACTS_DIR, "ipt_scaler.pkl")
    joblib.dump(scaler, path)
    print(f"  ✓ IPT scaler saved  →  mean={mean:.4f}  std={std:.4f}")
    print(f"  ✓ Saved to {path}\n")
    return mean, std


def load_ipt_scaler() -> tuple[float, float]:
    """Loads the previously fitted IPT scaler from disk."""
    path = os.path.join(ARTIFACTS_DIR, "ipt_scaler.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"IPT scaler not found at {path}.\n"
            "Run feature_engineering.fit_ipt_scaler(train_df) first."
        )
    scaler = joblib.load(path)
    return scaler["mean"], scaler["std"]


# ── Per-sequence transforms ──────────────────────────────────────────────────

def transform_ipt(seq, mean: float, std: float) -> np.ndarray:
    """
    log1p → Z-score normalisation.
    Padding positions (beyond actual sequence length) remain 0.0.
    """
    arr    = np.asarray(seq, dtype=np.float64)
    length = min(len(arr), SEQ_LEN)
    out    = np.zeros(SEQ_LEN, dtype=np.float32)

    clipped  = np.clip(arr[:length], 0.0, IPT_CLIP_MAX)
    log_ipt  = np.log1p(clipped)
    z_scored = (log_ipt - mean) / (std + LOG_EPS)

    out[:length] = z_scored.astype(np.float32)
    return out


def transform_dir(seq) -> np.ndarray:
    """
    Client→server = +1, server→client = -1, padding = 0.
    The dataset already uses ±1 encoding; we just truncate/pad.
    0 is strictly reserved for padding so attention mechanisms can
    distinguish 'no packet' from 'packet with direction'.
    """
    arr    = np.asarray(seq, dtype=np.float32)
    length = min(len(arr), SEQ_LEN)
    out    = np.zeros(SEQ_LEN, dtype=np.float32)
    out[:length] = arr[:length]
    return out


def transform_size(seq) -> np.ndarray:
    """
    Min-Max normalisation: size / MTU → [0, 1].
    Theoretical bounds (0 bytes, 1500 bytes) are used so the scaler
    parameters are fixed and never need re-fitting across splits.
    """
    arr    = np.asarray(seq, dtype=np.float32)
    length = min(len(arr), SEQ_LEN)
    out    = np.zeros(SEQ_LEN, dtype=np.float32)
    out[:length] = np.clip(arr[:length], 0.0, MTU) / MTU
    return out


# ── Label encoding ───────────────────────────────────────────────────────────

def fit_label_encoder(train_df, min_samples: int = 200):
    """
    Fits a sklearn LabelEncoder on the training APP column.
    Drops classes with fewer than min_samples to avoid class-imbalance
    issues in macro-averaged metrics.

    Returns
    -------
    le         : fitted LabelEncoder
    valid_apps : set of app names retained after filtering
    """
    from sklearn.preprocessing import LabelEncoder

    counts     = train_df["APP"].value_counts()
    valid_apps = set(counts[counts >= min_samples].index.tolist())
    dropped    = set(counts[counts < min_samples].index.tolist())

    if dropped:
        print(f"  ⚠ Dropping {len(dropped)} rare classes "
              f"(< {min_samples} samples): {dropped}")

    le = LabelEncoder()
    le.fit(sorted(valid_apps))   # sorted for deterministic class ordering

    path = os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")
    joblib.dump(le, path)
    print(f"  ✓ Label encoder fitted: {len(le.classes_)} classes")
    print(f"  ✓ Saved to {path}\n")
    return le, valid_apps


def load_label_encoder():
    """Loads the previously fitted LabelEncoder from disk."""
    path = os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Label encoder not found at {path}.\n"
            "Run feature_engineering.fit_label_encoder(train_df) first."
        )
    return joblib.load(path)


if __name__ == "__main__":
    # Quick smoke-test with synthetic data
    fake_ipt  = [0, 12, 340, 5000, 1, 0]
    fake_dir  = [1, -1, 1, 1, -1, -1]
    fake_size = [100, 1450, 300, 800, 60, 1500]

    mean, std = 3.5, 1.2   # synthetic stand-ins

    ipt_out  = transform_ipt(fake_ipt, mean, std)
    dir_out  = transform_dir(fake_dir)
    size_out = transform_size(fake_size)

    assert ipt_out.shape  == (SEQ_LEN,), f"Bad IPT shape:  {ipt_out.shape}"
    assert dir_out.shape  == (SEQ_LEN,), f"Bad DIR shape:  {dir_out.shape}"
    assert size_out.shape == (SEQ_LEN,), f"Bad SIZE shape: {size_out.shape}"
    assert size_out.max() <= 1.0 and size_out.min() >= 0.0, "Size out of [0,1]"
    assert set(dir_out[:6]).issubset({-1.0, 0.0, 1.0}), "Bad dir values"

    print("✅  feature_engineering.py smoke-test passed.")
