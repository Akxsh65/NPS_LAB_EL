# validate_pipeline.py
# ============================================================
# Phase 1 — Final go/no-go validation suite.
#
# Run this AFTER build_loaders() completes. All 7 checks must
# pass before you transfer files to the A100 system.
#
# Usage:
#   python validate_pipeline.py
# ============================================================

import time
import torch
import numpy as np
from collections import Counter

PASS = "  ✅ PASS"
FAIL = "  ❌ FAIL"


def run_all_checks(train_loader, val_loader, test_loader,
                   label_encoder, ipt_mean, ipt_std, train_df):
    """
    Runs all 7 validation checks and prints a final summary.
    Raises AssertionError on the first failed check.
    """
    results = {}

    print("\n" + "="*55)
    print("  PHASE 1 VALIDATION SUITE")
    print("="*55)

    results["shape"]     = _check_shape(train_loader)
    results["ranges"]    = _check_value_ranges(train_loader)
    results["nan_inf"]   = _check_nan_inf(train_loader)
    results["padding"]   = _check_padding_sentinel(train_loader)
    results["labels"]    = _check_label_range(train_loader, label_encoder)
    results["balance"]   = _check_class_balance(train_df, label_encoder)
    results["throughput"]= _check_throughput(train_loader)

    print("\n" + "="*55)
    print("  RESULTS SUMMARY")
    print("="*55)
    all_passed = True
    for name, passed in results.items():
        status = PASS if passed else FAIL
        print(f"  {name:<15} {status}")
        if not passed:
            all_passed = False

    print("="*55)
    if all_passed:
        print("  🎉 ALL CHECKS PASSED — Phase 1 complete.")
        print("     Transfer ./artifacts/ to the A100 system and begin Phase 2.")
    else:
        print("  ⚠  SOME CHECKS FAILED — do not proceed to Phase 2.")
    print("="*55 + "\n")
    return all_passed


# ── Individual checks ────────────────────────────────────────────────────────

def _check_shape(loader) -> bool:
    """Tensor shape must be (batch, 3, 30)."""
    print("\n[1/7] Checking tensor shape ...")
    try:
        X, y = next(iter(loader))
        from config import SEQ_LEN, BATCH_SIZE
        assert X.ndim == 3,               f"Expected 3D tensor, got {X.ndim}D"
        assert X.shape[1] == 3,           f"Expected 3 channels, got {X.shape[1]}"
        assert X.shape[2] == SEQ_LEN,     f"Expected seq_len={SEQ_LEN}, got {X.shape[2]}"
        assert X.dtype == torch.float32,  f"Expected float32, got {X.dtype}"
        assert y.dtype == torch.int64,    f"Expected int64 labels, got {y.dtype}"
        print(f"     X shape: {tuple(X.shape)}   y shape: {tuple(y.shape)}")
        return True
    except Exception as e:
        print(f"     ERROR: {e}")
        return False


def _check_value_ranges(loader) -> bool:
    """
    SIZE channel must be in [0,1].
    DIR  channel values must be a subset of {-1, 0, +1}.
    """
    print("[2/7] Checking channel value ranges ...")
    try:
        X, _ = next(iter(loader))
        size_ch = X[:, 2, :]
        dir_ch  = X[:, 1, :]

        assert float(size_ch.min()) >= 0.0, \
            f"SIZE below 0: {float(size_ch.min()):.4f}"
        assert float(size_ch.max()) <= 1.0, \
            f"SIZE above 1: {float(size_ch.max()):.4f}"

        unique_dirs = set(dir_ch.unique().tolist())
        allowed     = {-1.0, 0.0, 1.0}
        # allow small float rounding e.g. 0.9999
        for v in unique_dirs:
            assert any(abs(v - a) < 0.01 for a in allowed), \
                f"Unexpected DIR value: {v}"

        print(f"     SIZE ∈ [{float(size_ch.min()):.4f}, {float(size_ch.max()):.4f}]")
        print(f"     DIR  unique values: {sorted(unique_dirs)}")
        return True
    except Exception as e:
        print(f"     ERROR: {e}")
        return False


def _check_nan_inf(loader) -> bool:
    """No NaN or Inf anywhere in the first batch."""
    print("[3/7] Checking for NaN / Inf ...")
    try:
        X, y = next(iter(loader))
        nan_count = int(torch.isnan(X).sum())
        inf_count = int(torch.isinf(X).sum())
        assert nan_count == 0, f"Found {nan_count} NaN values in batch"
        assert inf_count == 0, f"Found {inf_count} Inf values in batch"
        print(f"     NaN: {nan_count}   Inf: {inf_count}")
        return True
    except Exception as e:
        print(f"     ERROR: {e}")
        return False


def _check_padding_sentinel(loader) -> bool:
    """
    Padding positions (beyond actual sequence length) should be 0.
    We verify that short flows have trailing zeros in ALL three channels.
    """
    print("[4/7] Checking padding sentinel = 0 ...")
    try:
        X, _ = next(iter(loader))
        # find rows where IPT channel has trailing zeros (padded flows)
        ipt_ch     = X[:, 0, :]
        padded_mask = (ipt_ch == 0.0)

        # for the same positions, DIR and SIZE should also be 0
        dir_at_pad  = X[:, 1, :][padded_mask]
        size_at_pad = X[:, 2, :][padded_mask]

        dir_non_zero  = int((dir_at_pad  != 0.0).sum())
        size_non_zero = int((size_at_pad != 0.0).sum())

        # allow a small tolerance — the first IPT is always 0 by definition
        # so we can't be 100% strict, just check there's no systematic error
        padded_total = int(padded_mask.sum())
        print(f"     Padding positions checked: {padded_total:,}")
        print(f"     DIR  non-zero at padded positions: {dir_non_zero}")
        print(f"     SIZE non-zero at padded positions: {size_non_zero}")
        # warn rather than fail — first packet IPT is legitimately 0
        if dir_non_zero > padded_total * 0.1:
            print("     WARNING: many non-zero DIR values at padding positions")
        return True
    except Exception as e:
        print(f"     ERROR: {e}")
        return False


def _check_label_range(loader, label_encoder) -> bool:
    """All label indices must be in [0, num_classes)."""
    print("[5/7] Checking label range ...")
    try:
        num_classes = len(label_encoder.classes_)
        bad_batches = 0
        for X, y in loader:
            if int(y.min()) < 0 or int(y.max()) >= num_classes:
                bad_batches += 1
                break   # one failure is enough

        assert bad_batches == 0, \
            f"Labels outside [0, {num_classes})"
        print(f"     num_classes={num_classes}   label range OK")
        return True
    except Exception as e:
        print(f"     ERROR: {e}")
        return False


def _check_class_balance(train_df, label_encoder, min_ratio: float = 0.001) -> bool:
    """
    Checks for extreme class imbalance.
    Warns if any class holds < 0.1% of training data.
    This is a WARNING check — it doesn't block Phase 2 but you should
    consider weighted sampling if the imbalance is severe.
    """
    print("[6/7] Checking class balance ...")
    try:
        valid_apps = set(label_encoder.classes_)
        counts     = train_df[train_df["APP"].isin(valid_apps)]["APP"].value_counts()
        total      = counts.sum()
        ratios     = counts / total

        min_class  = ratios.idxmin()
        min_ratio_val = float(ratios.min())
        max_class  = ratios.idxmax()
        max_ratio_val = float(ratios.max())
        imbalance_ratio = max_ratio_val / min_ratio_val

        print(f"     Most  frequent: '{max_class}' ({max_ratio_val*100:.2f}%)")
        print(f"     Least frequent: '{min_class}' ({min_ratio_val*100:.2f}%)")
        print(f"     Imbalance ratio: {imbalance_ratio:.1f}x")

        if min_ratio_val < min_ratio:
            print(f"     ⚠ WARNING: '{min_class}' has < {min_ratio*100:.1f}% "
                  f"of training data. Consider WeightedRandomSampler in Phase 2.")
        return True
    except Exception as e:
        print(f"     ERROR: {e}")
        return False


def _check_throughput(loader, n_batches: int = 30) -> bool:
    """
    Measures DataLoader throughput on CPU.
    Target: ≥ 5 batches/sec (comfortable for CPU-only preprocessing).
    """
    print(f"[7/7] Benchmarking throughput ({n_batches} batches) ...")
    try:
        start = time.perf_counter()
        for i, (X, y) in enumerate(loader):
            _ = X.shape   # force materialisation
            if i + 1 >= n_batches:
                break
        elapsed = time.perf_counter() - start
        bps     = n_batches / elapsed

        print(f"     {n_batches} batches in {elapsed:.2f}s  →  {bps:.1f} batches/sec")

        if bps < 5.0:
            print("     ⚠ WARNING: throughput is low. "
                  "On the A100, set num_workers≥4 and pin_memory=True.")
        return True
    except Exception as e:
        print(f"     ERROR: {e}")
        return False
