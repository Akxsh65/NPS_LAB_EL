"""
Run full Phase 4 evaluation: baseline + all obfuscated test sets.

Usage (from phase4/):
  python run_experiments.py
  python run_experiments.py --skip-cm
"""
from __future__ import annotations

import argparse
import json
import os
from glob import glob

import pandas as pd

from config import MANIFEST, PHASE3_ARTIFACTS, PHASE4_RESULTS, TEST_TENSORS
from evaluate import evaluate_one, resolve_checkpoint


def load_manifest_overheads() -> dict:
    """Map dataset stem -> mean bandwidth / latency from Phase 3 manifest."""
    if not os.path.isfile(MANIFEST):
        return {}
    with open(MANIFEST, encoding="utf-8") as f:
        entries = json.load(f)
    out = {}
    for e in entries:
        stem = Path(e["file"]).stem
        out[stem] = {
            "mean_bandwidth_overhead": e.get("mean_bandwidth_overhead", 0.0),
            "mean_latency_overhead_ms": e.get("mean_latency_overhead_ms", 0.0),
            "padding_type": e.get("padding_type", "unknown"),
            "jitter_scale": e.get("jitter_scale", 0.0),
        }
    return out


from pathlib import Path  # noqa: E402


def discover_obfuscated_files() -> list[str]:
    pattern = os.path.join(PHASE3_ARTIFACTS, "obfuscated_*.pt")
    return sorted(glob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 full experiment runner")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-cm", action="store_true", help="Skip confusion matrix PNGs")
    parser.add_argument("--out-dir", default=PHASE4_RESULTS)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt = resolve_checkpoint(args.checkpoint)
    print(f"Using checkpoint: {ckpt}")

    datasets = [("baseline", TEST_TENSORS)]
    for p in discover_obfuscated_files():
        datasets.append((Path(p).stem, p))

    overheads = load_manifest_overheads()
    rows = []

    for name, pt_path in datasets:
        if not os.path.isfile(pt_path):
            print(f"SKIP missing: {pt_path}")
            continue

        print(f"\n=== Evaluating: {name} ===")
        metrics = evaluate_one(
            pt_path,
            checkpoint=ckpt,
            device=args.device,
            save_cm=not args.skip_cm,
            out_dir=args.out_dir,
        )

        oh = overheads.get(name, {})
        row = {
            "experiment": name,
            "dataset": pt_path,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "num_samples": metrics["num_samples"],
            "mean_bandwidth_overhead": oh.get("mean_bandwidth_overhead", 0.0),
            "mean_latency_overhead_ms": oh.get("mean_latency_overhead_ms", 0.0),
            "padding_type": oh.get("padding_type", "none" if name == "baseline" else ""),
            "jitter_scale": oh.get("jitter_scale", 0.0),
        }
        rows.append(row)
        print(f"  accuracy={row['accuracy']:.4f}  macro_f1={row['macro_f1']:.4f}")

    if not rows:
        print("No datasets evaluated.")
        return

    df = pd.DataFrame(rows)
    baseline_acc = df.loc[df["experiment"] == "baseline", "accuracy"].iloc[0]
    df["accuracy_drop"] = baseline_acc - df["accuracy"]
    df["accuracy_drop_pct"] = (df["accuracy_drop"] / baseline_acc) * 100.0

    csv_path = os.path.join(args.out_dir, "accuracy_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results -> {csv_path}")

    summary = {
        "baseline_accuracy": float(baseline_acc),
        "checkpoint": ckpt,
        "num_experiments": len(rows),
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Pareto plots
    try:
        from pareto import plot_all

        plot_all(csv_path=csv_path, out_dir=args.out_dir)
    except Exception as exc:
        print(f"Warning: could not generate Pareto plots: {exc}")


if __name__ == "__main__":
    main()
