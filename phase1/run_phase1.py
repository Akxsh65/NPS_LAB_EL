# run_phase1.py
# ============================================================
# MASTER SCRIPT — Run this to complete Phase 1 end-to-end.
#
# Usage:
#   python run_phase1.py
#
# What it does (in order):
#   1. Downloads / loads CESNET-QUIC22 (W-2022-44 + W-2022-45)
#   2. Fits IPT scaler and LabelEncoder on TRAIN split only
#   3. Builds (N, 3, 30) tensors for all three splits
#   4. Saves tensors + artifacts to disk
#   5. Runs the full validation suite (7 checks)
#   6. Prints a transfer checklist for the A100 system
#
# Expected runtime on a modern CPU with 16 GB RAM: ~45–90 minutes
# (mostly spent on the first download; subsequent runs use cache)
# ============================================================

import torch
import random
import numpy as np

from config import SEED, MIN_CLASS_SAMPLES
from dataset_loader      import load_raw_dataframes
from feature_engineering import (
    fit_ipt_scaler, fit_label_encoder, load_ipt_scaler, load_label_encoder
)
from dataset             import build_loaders, save_datasets
from validate_pipeline   import run_all_checks


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    set_seed(SEED)

    print("╔══════════════════════════════════════════════════════╗")
    print("║   PHASE 1: Data Engineering  (CPU-only / Windows)   ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # ── Step 1: Download / load raw DataFrames ───────────────
    print("STEP 1 ── Data ingestion")
    print("-" * 55)
    train_df, val_df, test_df = load_raw_dataframes()

    # ── Step 2: Fit scalers (TRAIN ONLY) ─────────────────────
    print("STEP 2 ── Fitting scalers and label encoder")
    print("-" * 55)
    ipt_mean, ipt_std = fit_ipt_scaler(train_df)
    label_encoder, valid_apps = fit_label_encoder(train_df, MIN_CLASS_SAMPLES)

    # ── Step 3: Build PyTorch DataLoaders ────────────────────
    print("STEP 3 ── Building PyTorch DataLoaders")
    print("-" * 55)
    train_loader, val_loader, test_loader = build_loaders(
        train_df, val_df, test_df,
        label_encoder, ipt_mean, ipt_std, valid_apps
    )

    # ── Step 4: Save tensors to disk ─────────────────────────
    print("\nSTEP 4 ── Saving tensors to ./artifacts/")
    print("-" * 55)
    # Re-build datasets to access .X and .y directly for saving
    from dataset import TrafficDataset
    train_ds = TrafficDataset(train_df, label_encoder,
                              ipt_mean, ipt_std, valid_apps, "Train")
    val_ds   = TrafficDataset(val_df,   label_encoder,
                              ipt_mean, ipt_std, valid_apps, "Val  ")
    test_ds  = TrafficDataset(test_df,  label_encoder,
                              ipt_mean, ipt_std, valid_apps, "Test ")
    save_datasets(train_ds, val_ds, test_ds)

    # ── Step 5: Validation suite ─────────────────────────────
    print("\nSTEP 5 ── Running validation suite")
    print("-" * 55)
    passed = run_all_checks(
        train_loader, val_loader, test_loader,
        label_encoder, ipt_mean, ipt_std, train_df
    )

    # ── Step 6: Transfer checklist ───────────────────────────
    if passed:
        _print_transfer_checklist()


def _print_transfer_checklist():
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║         A100 TRANSFER CHECKLIST                      ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  Copy these files/folders to the A100 system:        ║")
    print("║                                                       ║")
    print("║  ./artifacts/                                         ║")
    print("║    ├── train_tensors.pt      ← training data         ║")
    print("║    ├── val_tensors.pt        ← validation data       ║")
    print("║    ├── test_tensors.pt       ← test (FROZEN)         ║")
    print("║    ├── ipt_scaler.pkl        ← IPT mean + std        ║")
    print("║    └── label_encoder.pkl     ← class index mapping   ║")
    print("║                                                       ║")
    print("║  ./dataset.py                ← SavedDataset class    ║")
    print("║  ./config.py                 ← shared constants      ║")
    print("║  ./feature_engineering.py    ← obfuscator reuse      ║")
    print("║                                                       ║")
    print("║  On the A100, load data with:                        ║")
    print("║    from dataset import SavedDataset                   ║")
    print("║    ds = SavedDataset('./artifacts/train_tensors.pt')  ║")
    print("║    loader = DataLoader(ds, batch_size=1024,           ║")
    print("║                        pin_memory=True,               ║")
    print("║                        num_workers=4)                 ║")
    print("║                                                       ║")
    print("║  IMPORTANT: do NOT change ipt_scaler.pkl values.     ║")
    print("║  Phase 3 obfuscator must use the SAME scaler.        ║")
    print("╚══════════════════════════════════════════════════════╝\n")


if __name__ == "__main__":
    main()
