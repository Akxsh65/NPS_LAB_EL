"""
Run full Phase 4 evaluation: baseline + all obfuscated test sets.

Usage (from phase4/):
  python run_experiments.py --device cuda
  python run_experiments.py --checkpoint ../phase2/artifacts/transformer_production.pt --skip-cm
"""
from __future__ import annotations

import argparse
import json
import os
from glob import glob
from pathlib import Path

import pandas as pd

from config import (
    MANIFEST,
    PHASE3_ARTIFACTS,
    PHASE4_RESULTS,
    REFERENCE_TEST_ACCURACY,
    REFERENCE_TEST_MACRO_F1,
    TEST_TENSORS,
)
from evaluate import evaluate_one, resolve_checkpoint, resolve_model_config

DEFAULT_REPORT_EXPERIMENTS = [
    "baseline",
    "obfuscated_jitter_low",
    "obfuscated_mtu",
]

DEFAULT_CONFUSION_EXPERIMENTS = [
    "baseline",
    "obfuscated_jitter_low",
]


def load_manifest_overheads() -> dict:
    """Map dataset stem -> overhead stats from Phase 3 manifest v2."""
    if not os.path.isfile(MANIFEST):
        return {}
    with open(MANIFEST, encoding="utf-8") as f:
        raw = json.load(f)
    entries = raw.get("experiments", raw) if isinstance(raw, dict) else raw
    out = {}
    for e in entries:
        if not isinstance(e, dict) or "file" not in e:
            continue
        stem = Path(e["file"]).stem
        out[stem] = {
            "mean_bandwidth_overhead": e.get("mean_bandwidth_overhead", 0.0),
            "bandwidth_p95": e.get("bandwidth_p95", 0.0),
            "mean_latency_overhead_ms": e.get("mean_latency_overhead_ms", 0.0),
            "latency_p95": e.get("latency_p95", 0.0),
            "padding_type": e.get("padding_type", "unknown"),
            "jitter_scale": e.get("jitter_scale", 0.0),
            "jitter_key": e.get("jitter_key", ""),
        }
    return out


def discover_obfuscated_files() -> list[str]:
    pattern = os.path.join(PHASE3_ARTIFACTS, "obfuscated_*.pt")
    return sorted(glob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 full experiment runner")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None, help="Training config JSON (d_model=160, etc.)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--skip-cm", action="store_true", help="Skip confusion matrix PNGs")
    parser.add_argument(
        "--save-reports",
        action="store_true",
        help="Save per-class report for baseline, jitter_low, mtu",
    )
    parser.add_argument(
        "--reports-only",
        action="store_true",
        help="Only run per-class reports (no full CSV re-eval)",
    )
    parser.add_argument(
        "--confusion-only",
        action="store_true",
        help="Only save publication confusion matrices (baseline + jitter_low)",
    )
    parser.add_argument(
        "--confusion-experiments",
        nargs="*",
        default=None,
        help=f"Subset for confusion PNGs (default: {DEFAULT_CONFUSION_EXPERIMENTS})",
    )
    parser.add_argument(
        "--report-experiments",
        nargs="*",
        default=None,
        help=f"Subset for reports (default: {DEFAULT_REPORT_EXPERIMENTS})",
    )
    parser.add_argument(
        "--attack-model",
        choices=["transformer", "cnn_bilstm"],
        default="transformer",
        help="Frozen attack architecture (default: transformer_masked production ckpt)",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save y_true/y_pred .npz per experiment (for Tier C bootstrap)",
    )
    parser.add_argument("--out-dir", default=PHASE4_RESULTS)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt = resolve_checkpoint(args.checkpoint, attack_model=args.attack_model)
    cfg = resolve_model_config(ckpt, args.config, attack_model=args.attack_model)
    print(f"Attack model: {args.attack_model}")
    print(f"Checkpoint: {ckpt}")
    print(f"Config:     {cfg}")

    if not os.path.isfile(MANIFEST):
        print(f"WARNING: manifest missing: {MANIFEST}")
    else:
        with open(MANIFEST, encoding="utf-8") as f:
            m = json.load(f)
        n_exp = len(m.get("experiments", []))
        print(f"Manifest:   {n_exp} experiments")
        if n_exp == 0:
            print("  WARNING: rebuild manifest with phase3 generate_obfuscated.py --skip-existing")

    report_names = set(args.report_experiments or DEFAULT_REPORT_EXPERIMENTS)
    confusion_names = set(args.confusion_experiments or DEFAULT_CONFUSION_EXPERIMENTS)

    datasets = [("baseline", TEST_TENSORS)]
    for p in discover_obfuscated_files():
        datasets.append((Path(p).stem, p))

    overheads = load_manifest_overheads()
    rows = []
    pred_dir = os.path.join(args.out_dir, "predictions", args.attack_model)

    if args.reports_only:
        datasets = [(n, p) for n, p in datasets if n in report_names]
    elif args.confusion_only:
        datasets = [(n, p) for n, p in datasets if n in confusion_names]

    for name, pt_path in datasets:
        if not os.path.isfile(pt_path):
            print(f"SKIP missing: {pt_path}")
            continue

        print(f"\n=== Evaluating: {name} ===")
        do_report = args.save_reports or args.reports_only
        do_cm = args.confusion_only or (not args.skip_cm and not args.reports_only)
        if args.reports_only and name not in report_names:
            continue
        if args.confusion_only and name not in confusion_names:
            continue

        metrics = evaluate_one(
            pt_path,
            checkpoint=ckpt,
            config=cfg,
            device=args.device,
            experiment_name=name,
            attack_model=args.attack_model,
            save_cm=do_cm and (args.confusion_only or not args.reports_only),
            save_report=do_report and name in report_names,
            save_predictions=args.save_predictions and not args.reports_only and not args.confusion_only,
            predictions_dir=pred_dir,
            report_dir=os.path.join(args.out_dir, "reports"),
            out_dir=args.out_dir,
            batch_size=args.batch_size,
        )

        if args.reports_only:
            print(f"  Saved reports -> {metrics.get('per_class_reports', {})}")
            continue

        if args.confusion_only:
            print(f"  Saved CM -> {metrics.get('confusion_matrix', '')}")
            continue

        oh = overheads.get(name, {})
        row = {
            "experiment": name,
            "dataset": pt_path,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "num_samples": metrics["num_samples"],
            "mean_bandwidth_overhead": oh.get("mean_bandwidth_overhead", 0.0),
            "bandwidth_p95": oh.get("bandwidth_p95", 0.0),
            "mean_latency_overhead_ms": oh.get("mean_latency_overhead_ms", 0.0),
            "latency_p95": oh.get("latency_p95", 0.0),
            "padding_type": oh.get("padding_type", "none" if name == "baseline" else ""),
            "jitter_scale": oh.get("jitter_scale", 0.0),
            "jitter_key": oh.get("jitter_key", ""),
        }
        rows.append(row)
        print(
            f"  accuracy={row['accuracy']:.4f}  macro_f1={row['macro_f1']:.4f}  "
            f"bw_oh={row['mean_bandwidth_overhead']*100:.1f}%  "
            f"lat_oh={row['mean_latency_overhead_ms']:.1f}ms"
        )

    if args.reports_only:
        print("\nReports-only run complete.")
        return

    if args.confusion_only:
        print("\nConfusion-only run complete.")
        return

    if not rows:
        print("No datasets evaluated.")
        return

    df = pd.DataFrame(rows)
    baseline_row = df.loc[df["experiment"] == "baseline"].iloc[0]
    baseline_acc = float(baseline_row["accuracy"])
    baseline_f1 = float(baseline_row["macro_f1"])
    chance = 1.0 / 64.0

    df["accuracy_drop"] = baseline_acc - df["accuracy"]
    df["accuracy_drop_pct"] = (df["accuracy_drop"] / baseline_acc) * 100.0
    df["macro_f1_drop"] = baseline_f1 - df["macro_f1"]

    csv_name = (
        "accuracy_results_bilstm.csv"
        if args.attack_model == "cnn_bilstm"
        else "accuracy_results.csv"
    )
    csv_path = os.path.join(args.out_dir, csv_name)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results -> {csv_path}")

    summary = {
        "attack_model": args.attack_model,
        "baseline_accuracy": baseline_acc,
        "baseline_macro_f1": baseline_f1,
        "random_chance_accuracy": chance,
        "reference_test_accuracy_phase2_finalize": REFERENCE_TEST_ACCURACY,
        "reference_test_macro_f1_phase2_finalize": REFERENCE_TEST_MACRO_F1,
        "checkpoint": ckpt,
        "config": cfg,
        "manifest": MANIFEST,
        "num_experiments": len(rows),
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary -> {summary_path}")

    try:
        from pareto import plot_all

        if args.attack_model == "transformer":
            plot_all(csv_path=csv_path, out_dir=args.out_dir)
    except Exception as exc:
        print(f"Warning: could not generate plots: {exc}")

    if args.save_reports and not args.reports_only:
        print(f"\nPer-class reports (subset) -> {os.path.join(args.out_dir, 'reports')}")


if __name__ == "__main__":
    main()
