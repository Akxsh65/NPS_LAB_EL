"""
Phase 3 — Deterministic traffic obfuscation on normalized PPI tensors.

Pipeline per flow:
  1. Denormalize (IPT, DIR, SIZE) to raw units
  2. Apply size padding and/or IPT jitter on *active* packets only
  3. Renormalize with the *same* Phase 1 IPT scaler (never refit)

Reproducibility: batch obfuscation uses seed + flow_index per flow RNG.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Literal, Optional, Tuple

import joblib
import numpy as np
import torch

PHASE1_DIR = Path(__file__).resolve().parents[1] / "phase1"
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))

# Import Phase 1 modules BEFORE phase3/settings (phase1 uses `config` module name).
from feature_engineering import (  # noqa: E402
    transform_dir,
    transform_ipt,
    transform_size,
)

from settings import (  # noqa: E402
    ATOL_IPT_PAD,
    ATOL_SIZE_EXACT,
    IPT_CLIP_MAX,
    IPT_SCALER,
    LINEAR_BLOCK,
    LOG_EPS,
    MTU,
    PAD_DIR_EPS,
    PAD_SIZE_EPS,
    SEQ_LEN,
)

PaddingType = Literal["none", "linear128", "mtu"]


def load_ipt_scaler_from(path: Optional[str] = None) -> Tuple[float, float]:
    """
    Load IPT mean/std from an explicit path (default: settings.IPT_SCALER).

    Does NOT use phase1/config ARTIFACTS_DIR (that breaks when cwd is phase3/).
    """
    scaler_path = path or IPT_SCALER
    if not Path(scaler_path).is_file():
        raise FileNotFoundError(
            f"IPT scaler not found: {scaler_path}\n"
            "Copy phase1/artifacts/ipt_scaler.pkl from your Phase 1 machine.\n"
            "Do not move it to phase3/artifacts — keep it with Phase 1 artifacts."
        )
    scaler = joblib.load(scaler_path)
    return float(scaler["mean"]), float(scaler["std"])


def active_packet_mask(dir_seq: np.ndarray, size_seq: np.ndarray) -> np.ndarray:
    """
    True where a real packet exists (Phase 1: padding has DIR≈0 and SIZE≈0).

    Uses AND for padding detection (stricter than OR) so trailing zeros are
    never padded or jittered.
    """
    is_padding = (np.abs(dir_seq) < PAD_DIR_EPS) & (size_seq < PAD_SIZE_EPS)
    return ~is_padding


def padding_slot_mask(dir_seq: np.ndarray, size_seq: np.ndarray) -> np.ndarray:
    """True for padding slots (inverse of active_packet_mask)."""
    return ~active_packet_mask(dir_seq, size_seq)


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
    """flow: (3, 30) normalized → raw (ipt, dir, size)."""
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
    """Rebuild (3, 30) float32 tensor using Phase 1 transforms."""
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
    One-sided Laplace jitter on IPT (ms): sample Laplace(0, scale), clip to ≥0.
    Index 0 is never jittered (CESNET first-packet IPT convention).
    """
    out = ipt.copy()
    injected = 0.0
    for i in np.where(mask)[0]:
        if i == 0:
            continue
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
    bandwidth_overhead: (sum bytes after - before) / sum bytes before (active only)
    latency_overhead_ms: sum of positive IPT increases on active indices > 0
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


def _flow_rng(seed: int, flow_index: int) -> np.random.Generator:
    """Per-flow RNG: reproducible and independent of batch loop order."""
    return np.random.default_rng(seed + int(flow_index))


def obfuscate(
    flow_tensor: np.ndarray | torch.Tensor,
    padding_type: PaddingType = "none",
    jitter_scale: float = 0.0,
    ipt_mean: Optional[float] = None,
    ipt_std: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    flow_index: int = 0,
    seed: int = 42,
) -> Tuple[np.ndarray, dict]:
    """
    Obfuscate a single flow (3, 30).

    DIR is never modified. Only active packet slots receive padding/jitter.
    """
    if isinstance(flow_tensor, torch.Tensor):
        flow = flow_tensor.detach().cpu().numpy()
    else:
        flow = np.asarray(flow_tensor, dtype=np.float32)

    if flow.shape != (3, SEQ_LEN):
        raise ValueError(f"Expected flow shape (3, {SEQ_LEN}), got {flow.shape}")

    if ipt_mean is None or ipt_std is None:
        ipt_mean, ipt_std = load_ipt_scaler_from()

    if rng is None:
        rng = _flow_rng(seed, flow_index)

    ipt, dir_seq, size = denormalize_flow(flow, ipt_mean, ipt_std)
    mask = active_packet_mask(dir_seq, size)
    n_active = int(mask.sum())

    ipt_before = ipt.copy()
    size_before = size.copy()

    if padding_type == "linear128":
        size = pad_sizes_linear(size, mask)
    elif padding_type == "mtu":
        size = pad_sizes_mtu(size, mask)
    elif padding_type != "none":
        raise ValueError(f"Unknown padding_type: {padding_type}")

    injected_lat = 0.0
    if jitter_scale > 0:
        ipt, injected_lat = add_laplace_jitter(ipt, mask, jitter_scale, rng)

    bw, lat = compute_overheads(size_before, size, ipt_before, ipt, mask)
    out = normalize_flow(ipt, dir_seq, size, ipt_mean, ipt_std)

    meta = {
        "padding_type": padding_type,
        "jitter_scale": jitter_scale,
        "bandwidth_overhead": bw,
        "latency_overhead_ms": lat,
        "n_active_packets": n_active,
        "size_changed": bool(np.any(np.abs(size - size_before) > 1e-6)),
        "ipt_changed": bool(np.any(np.abs(ipt - ipt_before) > 1e-6)),
    }
    return out, meta


def _percentiles(arr: np.ndarray, ps: Tuple[float, ...] = (50.0, 95.0)) -> dict:
    if len(arr) == 0:
        return {f"p{int(p)}": 0.0 for p in ps}
    return {f"p{int(p)}": float(np.percentile(arr, p)) for p in ps}


def obfuscate_batch(
    X: torch.Tensor | np.ndarray,
    padding_type: PaddingType = "none",
    jitter_scale: float = 0.0,
    seed: int = 42,
    ipt_mean: Optional[float] = None,
    ipt_std: Optional[float] = None,
    show_progress: bool = False,
) -> Tuple[torch.Tensor, dict]:
    """
    Obfuscate (N, 3, 30). Labels must be applied separately (unchanged).
    """
    if isinstance(X, torch.Tensor):
        arr = X.numpy()
    else:
        arr = np.asarray(X)

    if arr.ndim != 3 or arr.shape[1] != 3 or arr.shape[2] != SEQ_LEN:
        raise ValueError(f"Expected X shape (N, 3, {SEQ_LEN}), got {arr.shape}")

    if ipt_mean is None or ipt_std is None:
        ipt_mean, ipt_std = load_ipt_scaler_from()

    n = arr.shape[0]
    out = np.zeros_like(arr, dtype=np.float32)
    bw_list = np.zeros(n, dtype=np.float64)
    lat_list = np.zeros(n, dtype=np.float64)

    iterator = range(n)
    if show_progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc=f"obfuscate {padding_type}", leave=False)

    for i in iterator:
        out[i], meta = obfuscate(
            arr[i],
            padding_type=padding_type,
            jitter_scale=jitter_scale,
            ipt_mean=ipt_mean,
            ipt_std=ipt_std,
            flow_index=i,
            seed=seed,
        )
        bw_list[i] = meta["bandwidth_overhead"]
        lat_list[i] = meta["latency_overhead_ms"]

    bw_pos = bw_list[bw_list > 1e-9]
    aggregate: dict[str, Any] = {
        "padding_type": padding_type,
        "jitter_scale": jitter_scale,
        "seed": seed,
        "num_flows": n,
        "ipt_scaler_mean": float(ipt_mean),
        "ipt_scaler_std": float(ipt_std),
        "mean_bandwidth_overhead": float(np.mean(bw_list)),
        "std_bandwidth_overhead": float(np.std(bw_list)),
        "mean_latency_overhead_ms": float(np.mean(lat_list)),
        "std_latency_overhead_ms": float(np.std(lat_list)),
        "fraction_flows_bw_increase": float(np.mean(bw_list > 1e-9)),
        "max_bandwidth_overhead": float(np.max(bw_list)),
        "max_latency_overhead_ms": float(np.max(lat_list)),
    }
    aggregate.update({f"bandwidth_{k}": v for k, v in _percentiles(bw_list).items()})
    aggregate.update({f"latency_{k}": v for k, v in _percentiles(lat_list).items()})
    if len(bw_pos):
        aggregate["mean_bandwidth_overhead_nonzero"] = float(np.mean(bw_pos))
    return torch.from_numpy(out), aggregate


def ipt_scaler_fingerprint(scaler_path: Optional[str] = None) -> dict:
    """Audit trail: hash + stats of Phase 1 scaler file."""
    path = Path(scaler_path or IPT_SCALER)
    if not path.is_file():
        raise FileNotFoundError(f"IPT scaler missing: {path}")
    raw = path.read_bytes()
    mean, std = load_ipt_scaler_from(str(path))
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "mean": float(mean),
        "std": float(std),
    }


def verify_padding_slots_unchanged(
    clean: np.ndarray,
    obfuscated: np.ndarray,
    atol_ipt: float = ATOL_IPT_PAD,
) -> Tuple[bool, str]:
    """
    Padding slots (DIR≈0, SIZE≈0 in clean) should keep DIR/SIZE exact;
    IPT may differ slightly after renorm on zeros.
    """
    _, d0, s0 = clean[0], clean[1], clean[2]
    pad = padding_slot_mask(d0, s0)
    if not pad.any():
        return True, "no padding slots"

    obf = obfuscated
    dir_ok = np.allclose(obf[1][pad], clean[1][pad], atol=0.0)
    size_ok = np.allclose(obf[2][pad], clean[2][pad], atol=ATOL_SIZE_EXACT)
    ipt_ok = np.allclose(obf[0][pad], clean[0][pad], atol=atol_ipt)

    if dir_ok and size_ok and ipt_ok:
        return True, "ok"
    parts = []
    if not dir_ok:
        parts.append("DIR")
    if not size_ok:
        parts.append("SIZE")
    if not ipt_ok:
        parts.append("IPT")
    return False, f"padding drift: {', '.join(parts)}"


def roundtrip_renorm_error(
    flow: np.ndarray,
    ipt_mean: float,
    ipt_std: float,
) -> float:
    """Max abs error after denorm → renorm with no defense (sanity)."""
    ipt, d, s = denormalize_flow(flow, ipt_mean, ipt_std)
    rebuilt = normalize_flow(ipt, d, s, ipt_mean, ipt_std)
    return float(np.max(np.abs(rebuilt - flow)))
