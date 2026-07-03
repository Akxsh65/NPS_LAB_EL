"""
Generate obfuscated train + val tensors for adaptive-adversary retraining.

Usage (from phase3/):
  python generate_obfuscated_splits.py
  python generate_obfuscated_splits.py --only jitter_low jitter_medium
  python generate_obfuscated_splits.py --skip-existing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

from adaptive_policies import POLICIES, get_policy
from obfuscator import ipt_scaler_fingerprint, load_ipt_scaler_from, obfuscate_batch
from settings import (
    ADAPTIVE_ARTIFACTS,
    ADAPTIVE_TRAIN_DIR,
    ADAPTIVE_VAL_DIR,
    JITTER_SCALES,
    MANIFEST_VERSION,
    SEED,
    TRAIN_TENSORS,
    VAL_TENSORS,
)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tensor_sha256(X: torch.Tensor) -> str:
    arr = X.numpy().astype(np.float32, copy=False)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _split_path(split: str, policy_key: str) -> str:
    return os.path.join(
        ADAPTIVE_TRAIN_DIR if split == "train" else ADAPTIVE_VAL_DIR,
        f"{policy_key}_{split}.pt",
    )


def _obfuscate_split(
    source_pt: str,
    out_path: str,
    padding_type: str,
    jitter_key: str,
    seed: int,
    ipt_mean: float,
    ipt_std: float,
) -> dict:
    data = torch.load(source_pt, map_location="cpu", weights_only=False)
    X, y = data["X"], data["y"]
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X/y length mismatch in {source_pt}")

    jitter_scale = 0.0 if jitter_key == "none" else JITTER_SCALES[jitter_key]
    X_obf, meta = obfuscate_batch(
        X,
        padding_type=padding_type if padding_type != "none" else "none",
        jitter_scale=jitter_scale,
        seed=seed,
        ipt_mean=ipt_mean,
        ipt_std=ipt_std,
        show_progress=True,
    )

    if not torch.isfinite(X_obf).all():
        raise RuntimeError(f"Non-finite values in {out_path}")

    payload = {
        "X": X_obf,
        "y": y.clone(),
        "meta": {
            "obfuscation": {
                "padding_type": padding_type,
                "jitter_key": jitter_key,
                "jitter_scale": jitter_scale,
                "seed": seed,
            },
            "provenance": {
                "source_pt": os.path.abspath(source_pt),
                "manifest_version": MANIFEST_VERSION,
            },
        },
    }
    torch.save(payload, out_path)

    sidecar = {
        **meta,
        "num_samples": int(X_obf.shape[0]),
        "output_file": out_path,
        "jitter_key": jitter_key,
        "tensor_sha256": _tensor_sha256(X_obf),
        "labels_unchanged": True,
    }
    sidecar_path = Path(out_path).with_suffix(".meta.json")
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)

    return sidecar


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Obfuscate Phase 1 train/val splits for adaptive attacker training"
    )
    parser.add_argument("--train-pt", default=TRAIN_TENSORS)
    parser.add_argument("--val-pt", default=VAL_TENSORS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Policy keys to generate (default: all except clean)",
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    for path in (args.train_pt, args.val_pt):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing {path}\nCopy Phase 1 artifacts or run: cd ../phase1 && python run_phase1.py"
            )

    ipt_mean, ipt_std = load_ipt_scaler_from()
    scaler_info = ipt_scaler_fingerprint()

    policies = [p for p in POLICIES if not p.is_clean]
    if args.only:
        wanted = set(args.only)
        policies = [get_policy(k) for k in args.only if k != "clean"]
        unknown = wanted - {p.key for p in policies} - {"clean"}
        if unknown:
            raise ValueError(f"Unknown policy keys: {sorted(unknown)}")

    manifest = {
        "manifest_version": "adaptive_splits_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "source_train_pt": os.path.abspath(args.train_pt),
        "source_val_pt": os.path.abspath(args.val_pt),
        "source_train_sha256": _sha256_file(args.train_pt),
        "source_val_sha256": _sha256_file(args.val_pt),
        "ipt_scaler": scaler_info,
        "clean_policy": {
            "train_pt": os.path.abspath(args.train_pt),
            "val_pt": os.path.abspath(args.val_pt),
        },
        "policies": [],
    }

    print(f"Train source: {args.train_pt}")
    print(f"Val source:   {args.val_pt}")
    print(f"Output root:  {ADAPTIVE_ARTIFACTS}")
    print(f"Policies:     {[p.key for p in policies]}")

    for policy in policies:
        print(f"\n=== policy={policy.key} ({policy.description}) ===")
        entry = {
            "policy_key": policy.key,
            "padding_type": policy.padding_type,
            "jitter_key": policy.jitter_key,
            "splits": {},
        }

        for split, source in (("train", args.train_pt), ("val", args.val_pt)):
            out_path = _split_path(split, policy.key)
            if args.skip_existing and os.path.isfile(out_path):
                print(f"  SKIP existing {split} -> {out_path}")
                sidecar_path = Path(out_path).with_suffix(".meta.json")
                if sidecar_path.is_file():
                    with open(sidecar_path, encoding="utf-8") as f:
                        sidecar = json.load(f)
                    entry["splits"][split] = {
                        "file": out_path,
                        "num_samples": sidecar.get("num_samples"),
                        "tensor_sha256": sidecar.get("tensor_sha256"),
                    }
                continue

            print(f"  Generating {split} -> {out_path}")
            sidecar = _obfuscate_split(
                source,
                out_path,
                policy.padding_type,
                policy.jitter_key,
                args.seed,
                ipt_mean,
                ipt_std,
            )
            entry["splits"][split] = {
                "file": out_path,
                "num_samples": sidecar["num_samples"],
                "tensor_sha256": sidecar["tensor_sha256"],
                "mean_bandwidth_overhead": sidecar.get("mean_bandwidth_overhead"),
                "mean_latency_overhead_ms": sidecar.get("mean_latency_overhead_ms"),
            }
            print(
                f"    n={sidecar['num_samples']}  "
                f"bw={sidecar.get('mean_bandwidth_overhead', 0)*100:.2f}%  "
                f"lat={sidecar.get('mean_latency_overhead_ms', 0):.2f} ms"
            )

        manifest["policies"].append(entry)

    manifest_path = os.path.join(ADAPTIVE_ARTIFACTS, "adaptive_splits_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest -> {manifest_path}")


if __name__ == "__main__":
    main()
