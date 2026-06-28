"""
Generate obfuscated test tensor files for Phase 4.

Usage (from phase3/):
  python generate_obfuscated.py
  python generate_obfuscated.py --validate
  python generate_obfuscated.py --test-pt ../phase1/artifacts/test_tensors.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

# obfuscator first: loads phase1/config.py as `config` before phase3 settings
from obfuscator import ipt_scaler_fingerprint, load_ipt_scaler_from, obfuscate_batch
from settings import (
    IPT_SCALER,
    JITTER_SCALES,
    MANIFEST_VERSION,
    PHASE3_ARTIFACTS,
    SEED,
    TEST_TENSORS,
)

EXPERIMENTS = [
    ("obfuscated_linear128", "linear128", "none"),
    ("obfuscated_mtu", "mtu", "none"),
    ("obfuscated_jitter_low", "none", "low"),
    ("obfuscated_jitter_medium", "none", "medium"),
    ("obfuscated_jitter_high", "none", "high"),
    ("obfuscated_linear128_jitter_medium", "linear128", "medium"),
    ("obfuscated_mtu_jitter_medium", "mtu", "medium"),
]


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tensor_sha256(X: torch.Tensor) -> str:
    arr = X.numpy().astype(np.float32, copy=False)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate obfuscated test tensors")
    parser.add_argument("--test-pt", default=TEST_TENSORS)
    parser.add_argument("--out-dir", default=PHASE3_ARTIFACTS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Subset of output names to generate (default: all)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validate_obfuscation.py after generation",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip experiment if output .pt already exists",
    )
    parser.add_argument(
        "--ipt-scaler",
        default=None,
        help=f"Path to Phase 1 ipt_scaler.pkl (default: {IPT_SCALER})",
    )
    args = parser.parse_args()

    ipt_scaler_path = args.ipt_scaler or IPT_SCALER

    if not os.path.isfile(args.test_pt):
        raise FileNotFoundError(
            f"Test tensors not found: {args.test_pt}\n"
            "Copy phase1/artifacts/test_tensors.pt from your Phase 1 machine."
        )
    if not os.path.isfile(ipt_scaler_path):
        raise FileNotFoundError(
            f"IPT scaler not found: {ipt_scaler_path}\n\n"
            "Phase 3 must denormalize IPT using the SAME mean/std as Phase 1.\n"
            "This file is created by Phase 1 (run_phase1.py) and is usually NOT in git.\n\n"
            "Fix — copy from the machine where Phase 1 completed:\n"
            "  scp phase1/artifacts/ipt_scaler.pkl USER@SERVER:"
            f"{os.path.dirname(ipt_scaler_path)}/\n\n"
            "Also copy label_encoder.pkl for Phase 4.\n"
            "Or re-run: cd ../phase1 && python run_phase1.py\n"
            "Check: python check_prerequisites.py"
        )

    os.makedirs(args.out_dir, exist_ok=True)
    data = torch.load(args.test_pt, map_location="cpu", weights_only=False)
    X, y = data["X"], data["y"]
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X/y length mismatch: {X.shape[0]} vs {y.shape[0]}")

    print(f"Loaded test set: X={tuple(X.shape)}  y={tuple(y.shape)}")
    ipt_mean, ipt_std = load_ipt_scaler_from(ipt_scaler_path)
    scaler_info = ipt_scaler_fingerprint(ipt_scaler_path)
    print(f"IPT scaler: mean={ipt_mean:.4f} std={ipt_std:.4f}")
    print(f"IPT scaler path: {scaler_info['path']}")
    print(f"IPT scaler SHA256: {scaler_info['sha256'][:16]}...")

    manifest_header = {
        "manifest_version": MANIFEST_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_test_pt": os.path.abspath(args.test_pt),
        "source_test_sha256": _sha256_file(args.test_pt),
        "seed": args.seed,
        "num_samples": int(X.shape[0]),
        "num_classes": int(y.max().item()) + 1,
        "ipt_scaler": scaler_info,
        "experiments": [],
    }

    exps = EXPERIMENTS
    if args.only:
        names = set(args.only)
        exps = [e for e in EXPERIMENTS if e[0] in names]

    for out_name, padding, jitter_key in exps:
        jitter_scale = 0.0 if jitter_key == "none" else JITTER_SCALES[jitter_key]
        out_path = os.path.join(args.out_dir, f"{out_name}.pt")

        if args.skip_existing and os.path.isfile(out_path):
            print(f"SKIP existing {out_name}")
            sidecar_path = Path(out_path).with_suffix(".meta.json")
            if sidecar_path.is_file():
                with open(sidecar_path, encoding="utf-8") as f:
                    sidecar = json.load(f)
                manifest_header["experiments"].append(
                    {
                        "file": out_path,
                        "jitter_key": jitter_key,
                        **{k: sidecar[k] for k in sidecar if k != "output_file"},
                    }
                )
            else:
                print(f"  WARN: no sidecar {sidecar_path.name} — manifest entry omitted")
            continue

        print(f"\nGenerating {out_name}  padding={padding}  jitter_scale={jitter_scale}")

        X_obf, meta = obfuscate_batch(
            X,
            padding_type=padding if padding != "none" else "none",
            jitter_scale=jitter_scale,
            seed=args.seed,
            ipt_mean=ipt_mean,
            ipt_std=ipt_std,
            show_progress=True,
        )

        if not torch.isfinite(X_obf).all():
            raise RuntimeError(f"Non-finite values in {out_name}")

        payload = {
            "X": X_obf,
            "y": y.clone(),
            "meta": {
                "obfuscation": {
                    "padding_type": padding,
                    "jitter_key": jitter_key,
                    "jitter_scale": jitter_scale,
                    "seed": args.seed,
                },
                "provenance": {
                    "source_test_pt": os.path.abspath(args.test_pt),
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

        print(f"  Saved -> {out_path}")
        print(f"  mean BW overhead: {meta['mean_bandwidth_overhead']*100:.2f}%")
        print(f"  p95 BW overhead:  {meta.get('bandwidth_p95', 0)*100:.2f}%")
        print(f"  mean latency inj.: {meta['mean_latency_overhead_ms']:.2f} ms")

        manifest_header["experiments"].append(
            {"file": out_path, **meta, "jitter_key": jitter_key, "tensor_sha256": sidecar["tensor_sha256"]}
        )

    manifest_path = os.path.join(args.out_dir, "obfuscation_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_header, f, indent=2)
    print(f"\nManifest written -> {manifest_path}")

    if args.validate:
        print("\n>>> Running validation suite ...")
        script = Path(__file__).resolve().parent / "validate_obfuscation.py"
        rc = subprocess.call(
            [
                sys.executable,
                str(script),
                "--test-pt",
                args.test_pt,
                "--artifacts-dir",
                args.out_dir,
                "--manifest",
                manifest_path,
            ]
        )
        if rc != 0:
            raise SystemExit(rc)


if __name__ == "__main__":
    main()
