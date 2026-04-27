# dataset.py
# ============================================================
# Phase 1.3 — PyTorch Dataset class and DataLoader factory.
#
# TrafficDataset:
#   Materialises the entire split into a float32 tensor of shape
#   (N, 3, 30) in RAM during __init__ so that __getitem__ is
#   just a single tensor index — zero per-batch overhead.
#
# build_loaders():
#   Returns (train_loader, val_loader, test_loader) configured
#   for CPU-only Windows usage (num_workers=0, pin_memory=False).
# ============================================================

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import (
    SEQ_LEN, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, ARTIFACTS_DIR
)
from feature_engineering import transform_ipt, transform_dir, transform_size


class TrafficDataset(Dataset):
    """
    Converts a raw CESNET-QUIC22 DataFrame into a (N, 3, 30) float32
    tensor that lives entirely in CPU RAM.

    Tensor channel layout:
        dim 0  →  IPT   (log1p Z-scored)
        dim 1  →  DIR   (±1 with 0 padding)
        dim 2  →  SIZE  (Min-Max [0,1])

    Parameters
    ----------
    df           : raw DataFrame from cesnet-datazoo
    label_encoder: fitted sklearn LabelEncoder
    ipt_mean     : float — computed on TRAIN split only
    ipt_std      : float — computed on TRAIN split only
    valid_apps   : set   — classes retained after rare-class filtering
    split_name   : str   — human label for tqdm progress bar
    """

    def __init__(self, df, label_encoder, ipt_mean: float,
                 ipt_std: float, valid_apps: set, split_name: str = ""):

        # Drop rows whose app class was filtered out
        df = df[df["APP"].isin(valid_apps)].reset_index(drop=True)
        n  = len(df)

        print(f"  Building {split_name} tensors ({n:,} flows) ...")

        X = np.zeros((n, 3, SEQ_LEN), dtype=np.float32)
        y = np.empty(n, dtype=np.int64)

        for i in tqdm(range(n), desc=f"  {split_name}", leave=False):
            row = df.iloc[i]
            ppi = row["PPI"]                                     # single (3,30) array
            X[i, 0] = transform_ipt(ppi[0], ipt_mean, ipt_std)  # row 0 = IPT
            X[i, 1] = transform_dir(ppi[1])                     # row 1 = DIR
            X[i, 2] = transform_size(ppi[2])
            y[i]    = label_encoder.transform([row["APP"]])[0]

        self.X = torch.from_numpy(X)   # (N, 3, 30) — float32
        self.y = torch.from_numpy(y)   # (N,)        — int64
        self.n = n

        print(f"  ✓ {split_name}: X={tuple(self.X.shape)}  y={tuple(self.y.shape)}")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_loaders(train_df, val_df, test_df,
                  label_encoder, ipt_mean: float, ipt_std: float,
                  valid_apps: set):
    """
    Builds and returns the three DataLoaders.

    Windows CPU settings enforced here:
      num_workers = 0   (Windows multiprocessing spawn limitation)
      pin_memory  = False (no GPU DMA path on CPU-only)

    The test_loader is returned but should NOT be used until Phase 4.
    """
    print("\n── Building PyTorch Datasets ──────────────────────────")

    train_ds = TrafficDataset(train_df, label_encoder,
                              ipt_mean, ipt_std, valid_apps, "Train")
    val_ds   = TrafficDataset(val_df,   label_encoder,
                              ipt_mean, ipt_std, valid_apps, "Val  ")
    test_ds  = TrafficDataset(test_df,  label_encoder,
                              ipt_mean, ipt_std, valid_apps, "Test ")

    loader_kwargs = dict(
        batch_size  = BATCH_SIZE,
        num_workers = NUM_WORKERS,   # 0 on Windows
        pin_memory  = PIN_MEMORY,    # False on CPU
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    print("\n── DataLoader summary ─────────────────────────────────")
    print(f"  Train batches : {len(train_loader):,}  "
          f"({len(train_ds):,} samples)")
    print(f"  Val   batches : {len(val_loader):,}  "
          f"({len(val_ds):,} samples)")
    print(f"  Test  batches : {len(test_loader):,}  "
          f"({len(test_ds):,} samples)  ← FROZEN until Phase 4")

    return train_loader, val_loader, test_loader


def save_datasets(train_ds: TrafficDataset,
                  val_ds:   TrafficDataset,
                  test_ds:  TrafficDataset):
    """
    Saves the materialised tensors to disk as .pt files.
    On the A100 system you can load these directly — no re-processing needed.
    """
    torch.save({"X": train_ds.X, "y": train_ds.y},
               os.path.join(ARTIFACTS_DIR, "train_tensors.pt"))
    torch.save({"X": val_ds.X,   "y": val_ds.y},
               os.path.join(ARTIFACTS_DIR, "val_tensors.pt"))
    torch.save({"X": test_ds.X,  "y": test_ds.y},
               os.path.join(ARTIFACTS_DIR, "test_tensors.pt"))
    print("\n  ✓ Tensors saved to ./artifacts/")
    print("    train_tensors.pt  |  val_tensors.pt  |  test_tensors.pt")


class SavedDataset(Dataset):
    """
    Lightweight Dataset that loads pre-saved .pt tensors.
    Use this on the A100 machine instead of re-running the full pipeline.

    Usage
    -----
    ds = SavedDataset("./artifacts/train_tensors.pt")
    loader = DataLoader(ds, batch_size=1024, shuffle=True, pin_memory=True)
    """
    def __init__(self, pt_path: str):
        data   = torch.load(pt_path, map_location="cpu")
        self.X = data["X"]
        self.y = data["y"]

    def __len__(self):              return len(self.y)
    def __getitem__(self, idx):     return self.X[idx], self.y[idx]
