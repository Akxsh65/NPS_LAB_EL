"""
Phase 3 — Deterministic traffic obfuscation on normalized PPI tensors.

Pipeline per flow:
  1. Denormalize (IPT, DIR, SIZE) to raw units
  2. Apply size padding and/or IPT jitter in raw space
  3. Renormalize with the *same* Phase 1 IPT scaler (never refit)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import torch

# Phase 1 imports
PHASE1_DIR = Path(__file__).resolve().parents[1] / "phase1"
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))

from feature_engineering import (  # noqa: E402
    load_ipt_scaler,
    transform_dir,
    transform_ipt,
    transform_size,
)

from config import IPT_CLIP_MAX, LINEAR_BLOCK, LOG_EPS, MTU, SEQ_LEN  # noqa: E402

PaddingType = Literal["none", "linear128", "mtu"]
JitterMode = Literal["none", "low", "medium", "high"]


def _active_mask(dir_seq: np.ndarray, size_seq: np.ndarray) -> np.ndarray:
    """True for real packets; padding uses DIR=0 and SIZE=0."""
    return (np.abs(dir_seq) > 0.5) | (size_seq > 0.0)


def inverse_ipt(norm_ipt: np.ndarray, mean: float, std: float) -> np.ndarray:
    log_ipt = norm_ipt * (std + LOG_EPS) + mean
    raw = np.expm1(log_ipt)
    return np.clip(raw, 0.0, IPT_CLIP_MAX).astype(np.float64)


def inverse_size(norm_size: np.ndarray) -> np.ndarray:
    return (np.clip(norm_size, 0.0, 1.0) * MTU).astype(np.float64)


def denormalize_flow(
    flow: np.ndarray,
    ipt_mean: float,
    ipt_std: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    flow: (3, 30) normalized
    Returns raw (ipt, dir, size) each shape (30,)
    """
    ipt_n, dir_n, size_n = flow[0], flow[1], flow[2]
    ipt = inverse_ipt(ipt_n.astype(np.float64), ipt_mean, ipt_std)
    dir_seq = dir_n.astype(np.float32).copy()
    size = inverse_size(size_n.astype(np.float64))
    return ipt, dir_seq, size


def normalize_flow(
    ipt: np.ndarray,
    dir_seq: np.ndarray,
    size: np.ndarray,
    ipt_mean: float,
    ipt_std: float,
) -> np.ndarray:
    """Rebuild (3, 30) float32 tensor."""
    out = np.zeros((3, SEQ_LEN), dtype=np.float32)
    out[0] = transform_ipt(ipt, ipt_mean, ipt_std)
    out[1] = transform_dir(dir_seq)
    out[2] = transform_size(size)
    return out


def pad_sizes_linear(size: np.ndarray, mask: np.ndarray, block: int = LINEAR_BLOCK) -> np.ndarray:
    out = size.copy()
    for i in np.where(mask)[0]:
        s = out[i]
        if s <= 0:
            continue
        out[i] = float(np.ceil(s / block) * block)
        out[i] = min(out[i], MTU)
    return out


def pad_sizes_mtu(size: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = size.copy()
    for i in np.where(mask)[0]:
        if out[i] > 0:
            out[i] = MTU
    return out


def add_laplace_jitter(
    ipt: np.ndarray,
    mask: np.ndarray,
    scale: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    """
    Add zero-mean Laplace noise to IPT (ms). Negative delays clipped to 0.
    Returns (new_ipt, total_injected_latency_ms).
    """
    out = ipt.copy()
    injected = 0.0
    for i in np.where(mask)[0]:
        if i == 0:
            continue  # first packet IPT is always 0 in CESNET PPI
        noise = rng.laplace(loc=0.0, scale=scale)
        delta = max(0.0, noise)
        out[i] = out[i] + delta
        injected += delta
    out = np.clip(out, 0.0, IPT_CLIP_MAX)
    return out, injected


def compute_overheads(
    size_before: np.ndarray,
    size_after: np.ndarray,
    ipt_before: np.ndarray,
    ipt_after: np.ndarray,
    mask: np.ndarray,
) -> Tuple[float, float]:
    """
    bandwidth_overhead: fractional increase in total bytes (active packets only)
    latency_overhead: sum of positive IPT increases (ms)
    """
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return 0.0, 0.0

    orig_sum = float(size_before[idx].sum())
    new_sum = float(size_after[idx].sum())
    if orig_sum <= 0:
        bw = 0.0
    else:
        bw = (new_sum - orig_sum) / orig_sum

    lat = 0.0
    for i in idx:
        if i == 0:
            continue
        lat += max(0.0, float(ipt_after[i] - ipt_before[i]))

    return bw, lat


def obfuscate(
    flow_tensor: np.ndarray | torch.Tensor,
    padding_type: PaddingType = "none",
    jitter_scale: float = 0.0,
    ipt_mean: Optional[float] = None,
    ipt_std: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Obfuscate a single flow.

    Parameters
    ----------
    flow_tensor : (3, 30) normalized
    padding_type : 'none' | 'linear128' | 'mtu'
    jitter_scale : Laplace scale in ms (0 = no jitter)

    Returns
    -------
    obfuscated : (3, 30) float32
    meta : dict with bandwidth_overhead, latency_overhead_ms
    """
    if isinstance(flow_tensor, torch.Tensor):
        flow = flow_tensor.detach().cpu().numpy()
    else:
        flow = np.asarray(flow_tensor, dtype=np.float32)

    if ipt_mean is None or ipt_std is None:
        ipt_mean, ipt_std = load_ipt_scaler()

    if rng is None:
        rng = np.random.default_rng()

    ipt, dir_seq, size = denormalize_flow(flow, ipt_mean, ipt_std)
    mask = _active_mask(dir_seq, size)

    ipt_before = ipt.copy()
    size_before = size.copy()

    if padding_type == "linear128":
        size = pad_sizes_linear(size, mask)
    elif padding_type == "mtu":
        size = pad_sizes_mtu(size, mask)
    elif padding_type != "none":
        raise ValueError(f"Unknown padding_type: {padding_type}")

    if jitter_scale > 0:
        ipt, _ = add_laplace_jitter(ipt, mask, jitter_scale, rng)

    bw, lat = compute_overheads(size_before, size, ipt_before, ipt, mask)
    out = normalize_flow(ipt, dir_seq, size, ipt_mean, ipt_std)

    meta = {
        "padding_type": padding_type,
        "jitter_scale": jitter_scale,
        "bandwidth_overhead": bw,
        "latency_overhead_ms": lat,
    }
    return out, meta


def obfuscate_batch(
    X: torch.Tensor | np.ndarray,
    padding_type: PaddingType = "none",
    jitter_scale: float = 0.0,
    seed: int = 42,
    ipt_mean: Optional[float] = None,
    ipt_std: Optional[float] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Obfuscate (N, 3, 30) tensor. Returns (X_obf, aggregate_meta).
    """
    if isinstance(X, torch.Tensor):
        arr = X.numpy()
    else:
        arr = np.asarray(X)

    if ipt_mean is None or ipt_std is None:
        ipt_mean, ipt_std = load_ipt_scaler()

    rng = np.random.default_rng(seed)
    n = arr.shape[0]
    out = np.zeros_like(arr, dtype=np.float32)
    bw_list = []
    lat_list = []

    for i in range(n):
        out[i], meta = obfuscate(
            arr[i],
            padding_type=padding_type,
            jitter_scale=jitter_scale,
            ipt_mean=ipt_mean,
            ipt_std=ipt_std,
            rng=rng,
        )
        bw_list.append(meta["bandwidth_overhead"])
        lat_list.append(meta["latency_overhead_ms"])

    aggregate = {
        "padding_type": padding_type,
        "jitter_scale": jitter_scale,
        "mean_bandwidth_overhead": float(np.mean(bw_list)),
        "mean_latency_overhead_ms": float(np.mean(lat_list)),
        "num_flows": n,
    }
    return torch.from_numpy(out), aggregate
