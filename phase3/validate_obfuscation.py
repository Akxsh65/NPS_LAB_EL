"""
Phase 3 validation suite — run after generate_obfuscated.py.

Checks:
  - Labels unchanged vs clean test
  - Tensor shape/dtype/finite values
  - DIR channel unchanged (obfuscation never edits direction)
  - Padding slots preserved
  - Manifest overhead stats match recomputation (sample)
  - Optional identity check for no-op settings

Usage (from phase3/):
  python validate_obfuscation.py
  python validate_obfuscation.py --test-pt ../phase1/artifacts/test_tensors.pt --artifacts-dir ./artifacts
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

PHASE1_DIR = Path(__file__).resolve().parents[1] / "phase1"
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))

from settings import MANIFEST_VERSION, PHASE3_ARTIFACTS, TEST_TENSORS  # noqa: E402
from obfuscator import (  # noqa: E402
    active_packet_mask,
    denormalize_flow,
    ipt_scaler_fingerprint,
    load_ipt_scaler_from,
    obfuscate,
    verify_padding_slots_unchanged,
)


def _load_pt(path: str) -> tuple[np.ndarray, np.ndarray]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    X = data["X"].numpy() if isinstance(data["X"], torch.Tensor) else np.asarray(data["X"])
    y = data["y"].numpy() if isinstance(data["y"], torch.Tensor) else np.asarray(data["y"])
    return X, y


def check_tensor_health(X: np.ndarray, name: str) -> List[str]:
    errors = []
    if X.ndim != 3 or X.shape[1] != 3:
        errors.append(f"{name}: bad shape {X.shape}")
    if not np.isfinite(X).all():
        n_bad = int((~np.isfinite(X)).sum())
        errors.append(f"{name}: {n_bad} non-finite values")
    return errors


def check_dir_unchanged(X_clean: np.ndarray, X_obf: np.ndarray) -> List[str]:
    if not np.array_equal(X_clean[:, 1, :], X_obf[:, 1, :]):
        n_diff = int(np.sum(X_clean[:, 1, :] != X_obf[:, 1, :]))
        return [f"DIR channel modified in {n_diff} positions"]
    return []


def check_labels(y_clean: np.ndarray, y_obf: np.ndarray) -> List[str]:
    if not np.array_equal(y_clean, y_obf):
        return ["Labels differ from clean test set"]
    return []


def sample_padding_check(
    X_clean: np.ndarray,
    X_obf: np.ndarray,
    n_sample: int = 500,
    seed: int = 42,
) -> List[str]:
    rng = np.random.default_rng(seed)
    n = X_clean.shape[0]
    idx = rng.choice(n, size=min(n_sample, n), replace=False)
    failures = 0
    for i in idx:
        ok, _ = verify_padding_slots_unchanged(X_clean[i], X_obf[i])
        if not ok:
            failures += 1
    if failures > 0:
        return [f"Padding slot drift in {failures}/{len(idx)} sampled flows"]
    return []


def recompute_overhead_sample(
    X_clean: np.ndarray,
    X_obf: np.ndarray,
    n_sample: int = 200,
    seed: int = 42,
) -> tuple[list[float], list[float]]:
    """Spot-check manifest means against per-flow recomputation."""
    from obfuscator import compute_overheads

    mean_ipt, std_ipt = load_ipt_scaler_from()
    rng = np.random.default_rng(seed)
    n = X_clean.shape[0]
    idx = rng.choice(n, size=min(n_sample, n), replace=False)
    bw_list, lat_list = [], []

    for i in idx:
        ipt0, d0, s0 = denormalize_flow(X_clean[i], mean_ipt, std_ipt)
        ipt1, d1, s1 = denormalize_flow(X_obf[i], mean_ipt, std_ipt)
        mask = active_packet_mask(d0, s0)
        bw, lat = compute_overheads(s0, s1, ipt0, ipt1, mask)
        bw_list.append(bw)
        lat_list.append(lat)

    return bw_list, lat_list


def validate_one_obfuscated(
    clean_path: str,
    obf_path: str,
    manifest_entry: dict | None,
    n_sample: int,
) -> Dict:
    Xc, yc = _load_pt(clean_path)
    Xo, yo = _load_pt(obf_path)
    errors: List[str] = []
    warnings: List[str] = []

    errors.extend(check_tensor_health(Xc, "clean"))
    errors.extend(check_tensor_health(Xo, "obfuscated"))
    errors.extend(check_labels(yc, yo))
    errors.extend(check_dir_unchanged(Xc, Xo))
    errors.extend(sample_padding_check(Xc, Xo, n_sample=n_sample))

    meta_path = Path(obf_path)
    meta_sidecar = meta_path.with_suffix(".meta.json")
    if meta_sidecar.is_file():
        with open(meta_sidecar, encoding="utf-8") as f:
            sidecar = json.load(f)
        sidecar_n = sidecar.get("num_samples", sidecar.get("num_flows"))
        if sidecar_n is not None and int(sidecar_n) != Xo.shape[0]:
            errors.append(
                f"Sidecar sample count mismatch: sidecar={sidecar_n} tensor={Xo.shape[0]}"
            )

    if manifest_entry:
        padding = manifest_entry.get("padding_type", "none")
        jitter = float(manifest_entry.get("jitter_scale", 0.0))
        bw_list, lat_list = recompute_overhead_sample(Xc, Xo, n_sample=n_sample)
        rec_mean_bw = float(np.mean(bw_list))
        rec_mean_lat = float(np.mean(lat_list))
        man_bw = float(manifest_entry.get("mean_bandwidth_overhead", 0))
        man_lat = float(manifest_entry.get("mean_latency_overhead_ms", 0))
        if man_bw > 1e-6 and abs(rec_mean_bw - man_bw) / man_bw > 0.15:
            warnings.append(
                f"BW overhead sample mean {rec_mean_bw:.4f} vs manifest {man_bw:.4f}"
            )
        if man_lat > 1e-3 and abs(rec_mean_lat - man_lat) / man_lat > 0.20:
            warnings.append(
                f"Latency overhead sample mean {rec_mean_lat:.2f} vs manifest {man_lat:.2f}"
            )

    return {
        "file": obf_path,
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "num_samples": int(Xo.shape[0]),
    }


def load_manifest_entries(manifest_path: str) -> list[dict]:
    """Parse manifest v2 (dict + experiments) or v1 (flat list)."""
    with open(manifest_path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return [e for e in raw if isinstance(e, dict) and "file" in e]
    if isinstance(raw, dict):
        entries = raw.get("experiments", [])
        return [e for e in entries if isinstance(e, dict) and "file" in e]
    raise ValueError(f"Unrecognized manifest format: {manifest_path}")


def validate_identity_obfuscate(n_flows: int = 50, seed: int = 42) -> Dict:
    """No padding + no jitter: output matches input up to renorm round-trip error."""
    if os.path.isfile(TEST_TENSORS):
        X, _ = _load_pt(TEST_TENSORS)
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=min(n_flows, X.shape[0]), replace=False)
        flows = [X[i] for i in idx]
    else:
        flows = []
        f = np.zeros((3, 30), dtype=np.float32)
        f[1, :8] = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.float32)
        f[2, :8] = np.linspace(0.05, 0.4, 8, dtype=np.float32)
        f[0, 1:8] = 0.5
        flows = [f] * min(n_flows, 1)

    max_rt = 0.0
    for i, flow in enumerate(flows):
        out, _ = obfuscate(
            flow, padding_type="none", jitter_scale=0.0, seed=seed, flow_index=i
        )
        max_rt = max(max_rt, float(np.max(np.abs(out - flow))))
    ok = max_rt < 5e-3
    return {"passed": ok, "max_abs_diff": max_rt, "n_flows": len(flows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Phase 3 obfuscated tensors")
    parser.add_argument("--test-pt", default=TEST_TENSORS)
    parser.add_argument("--artifacts-dir", default=PHASE3_ARTIFACTS)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--n-sample", type=int, default=500)
    parser.add_argument("--out", default=None, help="JSON report path")
    args = parser.parse_args()

    manifest_path = args.manifest or os.path.join(args.artifacts_dir, "obfuscation_manifest.json")
    results: Dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_version": MANIFEST_VERSION,
        "clean_test_pt": os.path.abspath(args.test_pt),
        "ipt_scaler": ipt_scaler_fingerprint(),
        "datasets": [],
        "identity_check": validate_identity_obfuscate(),
    }

    manifest_by_file = {}
    if os.path.isfile(manifest_path):
        for entry in load_manifest_entries(manifest_path):
            manifest_by_file[os.path.abspath(entry["file"])] = entry
        results["manifest_path"] = manifest_path
        results["manifest_entries"] = len(manifest_by_file)
    else:
        results["manifest_warning"] = f"Manifest not found: {manifest_path}"

    obf_files = sorted(Path(args.artifacts_dir).glob("obfuscated_*.pt"))
    if not obf_files:
        print(f"No obfuscated_*.pt in {args.artifacts_dir}")
        sys.exit(1)

    all_pass = True
    for p in obf_files:
        entry = manifest_by_file.get(str(p.resolve()))
        rep = validate_one_obfuscated(
            args.test_pt, str(p), entry, n_sample=args.n_sample
        )
        results["datasets"].append(rep)
        status = "PASS" if rep["passed"] else "FAIL"
        print(f"[{status}] {p.name}")
        for e in rep["errors"]:
            print(f"       ERROR: {e}")
        for w in rep["warnings"]:
            print(f"       WARN:  {w}")
        if not rep["passed"]:
            all_pass = False

    id_ok = results["identity_check"]["passed"]
    print(f"\nIdentity (no-op) check: {'PASS' if id_ok else 'FAIL'} "
          f"(max_diff={results['identity_check']['max_abs_diff']:.2e})")
    if not id_ok:
        all_pass = False

    out_path = args.out or os.path.join(args.artifacts_dir, "validation_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nReport -> {out_path}")
    print("ALL CHECKS PASSED" if all_pass else "VALIDATION FAILED")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
