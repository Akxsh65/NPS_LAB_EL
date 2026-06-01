"""
Generate obfuscated test tensor files for Phase 4.

Usage (from phase3/):
  python generate_obfuscated.py
  python generate_obfuscated.py --test-pt ../phase1/artifacts/test_tensors.pt
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from tqdm import tqdm

from config import JITTER_SCALES, PHASE3_ARTIFACTS, SEED, TEST_TENSORS
from obfuscator import obfuscate_batch


# Experiment grid: (output_name, padding, jitter_key)
EXPERIMENTS = [
    ("obfuscated_linear128", "linear128", "none"),
    ("obfuscated_mtu", "mtu", "none"),
    ("obfuscated_jitter_low", "none", "low"),
    ("obfuscated_jitter_medium", "none", "medium"),
    ("obfuscated_jitter_high", "none", "high"),
    ("obfuscated_linear128_jitter_medium", "linear128", "medium"),
    ("obfuscated_mtu_jitter_medium", "mtu", "medium"),
]


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
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = torch.load(args.test_pt, map_location="cpu")
    X, y = data["X"], data["y"]
    print(f"Loaded test set: X={tuple(X.shape)}  y={tuple(y.shape)}")

    manifest = []
    exps = EXPERIMENTS
    if args.only:
        names = set(args.only)
        exps = [e for e in EXPERIMENTS if e[0] in names]

    for out_name, padding, jitter_key in exps:
        jitter_scale = 0.0 if jitter_key == "none" else JITTER_SCALES[jitter_key]
        print(f"\nGenerating {out_name}  padding={padding}  jitter_scale={jitter_scale}")

        X_obf, meta = obfuscate_batch(
            X,
            padding_type=padding if padding != "none" else "none",
            jitter_scale=jitter_scale,
            seed=args.seed,
        )

        out_path = os.path.join(args.out_dir, f"{out_name}.pt")
        torch.save({"X": X_obf, "y": y}, out_path)
        print(f"  Saved -> {out_path}")
        print(f"  mean BW overhead: {meta['mean_bandwidth_overhead']*100:.2f}%")
        print(f"  mean latency inj.: {meta['mean_latency_overhead_ms']:.2f} ms")

        manifest.append({"file": out_path, **meta, "jitter_key": jitter_key})

    manifest_path = os.path.join(args.out_dir, "obfuscation_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written -> {manifest_path}")


if __name__ == "__main__":
    main()
