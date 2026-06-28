"""
Tier C — BiLSTM comparison, bootstrap CIs, paired tests, channel ablation.

Usage (from phase4/):
  python run_tier_c.py --all --device cuda
  python run_tier_c.py --bilstm --device cuda
  python run_tier_c.py --bootstrap --paired
  python run_tier_c.py --channel-ablation --device cuda
  python run_tier_c.py --summarize-reports
"""
from __future__ import annotations

import argparse
import json
import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CHANNEL_ABLATION_PRESETS,
    PHASE3_ARTIFACTS,
    PHASE4_RESULTS,
    PREDICTIONS_DIR,
    TEST_TENSORS,
)
from evaluate import (
    evaluate_one,
    load_predictions_npz,
    resolve_checkpoint,
    resolve_model_config,
)
from run_experiments import discover_obfuscated_files, load_manifest_overheads
from statistical_analysis import bootstrap_metrics, mcnemar_test, paired_bootstrap_accuracy_diff
from summarize_reports import write_confusion_pairs_from_npz, write_worst_classes_summary

DEFAULT_PAIRED = ("baseline", "obfuscated_jitter_low")
CHANNEL_ABLATION_EXPERIMENTS = ("baseline", "obfuscated_jitter_low")


def _datasets() -> list[tuple[str, str]]:
    out = [("baseline", TEST_TENSORS)]
    for p in discover_obfuscated_files():
        out.append((Path(p).stem, p))
    return out


def _results_name(attack_model: str) -> str:
    if attack_model == "cnn_bilstm":
        return "accuracy_results_bilstm.csv"
    return "accuracy_results.csv"


def run_attack_eval(
    attack_model: str,
    device: str | None,
    batch_size: int,
    out_dir: str,
    checkpoint: str | None,
    config: str | None,
    save_predictions: bool,
) -> str:
    ckpt = resolve_checkpoint(checkpoint, attack_model=attack_model)
    cfg = resolve_model_config(ckpt, config, attack_model=attack_model)
    print(f"Attack model: {attack_model}")
    print(f"Checkpoint:   {ckpt}")
    print(f"Config:       {cfg}")

    pred_dir = os.path.join(out_dir, "predictions", attack_model)
    rows = []
    overheads = load_manifest_overheads()

    for name, pt_path in _datasets():
        if not os.path.isfile(pt_path):
            print(f"SKIP missing: {pt_path}")
            continue
        print(f"\n=== {attack_model}: {name} ===")
        metrics = evaluate_one(
            pt_path,
            checkpoint=ckpt,
            config=cfg,
            device=device,
            experiment_name=name,
            attack_model=attack_model,
            save_predictions=save_predictions,
            predictions_dir=pred_dir,
            out_dir=out_dir,
            batch_size=batch_size,
        )
        oh = overheads.get(name, {})
        rows.append(
            {
                "experiment": name,
                "attack_model": attack_model,
                "dataset": pt_path,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "num_samples": metrics["num_samples"],
                "mean_bandwidth_overhead": oh.get("mean_bandwidth_overhead", 0.0),
                "mean_latency_overhead_ms": oh.get("mean_latency_overhead_ms", 0.0),
            }
        )
        print(f"  acc={metrics['accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}  "
              f"model={metrics.get('model_name', '?')}")

    df = pd.DataFrame(rows)
    bl = df.loc[df["experiment"] == "baseline"].iloc[0]
    df["accuracy_drop"] = bl["accuracy"] - df["accuracy"]
    df["macro_f1_drop"] = bl["macro_f1"] - df["macro_f1"]

    csv_path = os.path.join(out_dir, _results_name(attack_model))
    df.to_csv(csv_path, index=False)
    print(f"\nSaved -> {csv_path}")
    return csv_path


def export_architecture_comparison(
    out_dir: str,
    transformer_csv: str | None = None,
    bilstm_csv: str | None = None,
) -> str:
    t_path = transformer_csv or os.path.join(out_dir, "accuracy_results.csv")
    b_path = bilstm_csv or os.path.join(out_dir, "accuracy_results_bilstm.csv")
    if not os.path.isfile(t_path) or not os.path.isfile(b_path):
        raise FileNotFoundError("Need both accuracy_results.csv and accuracy_results_bilstm.csv")

    tdf = pd.read_csv(t_path)[["experiment", "accuracy", "macro_f1"]].rename(
        columns={"accuracy": "transformer_acc", "macro_f1": "transformer_macro_f1"}
    )
    bdf = pd.read_csv(b_path)[["experiment", "accuracy", "macro_f1"]].rename(
        columns={"accuracy": "bilstm_acc", "macro_f1": "bilstm_macro_f1"}
    )
    merged = tdf.merge(bdf, on="experiment", how="inner")
    merged["acc_gap_transformer_minus_bilstm"] = (
        merged["transformer_acc"] - merged["bilstm_acc"]
    )
    merged["macro_f1_gap_transformer_minus_bilstm"] = (
        merged["transformer_macro_f1"] - merged["bilstm_macro_f1"]
    )
    out = os.path.join(out_dir, "architecture_comparison.csv")
    merged.to_csv(out, index=False)

    tex_lines = [
        "\\begin{tabular}{lcccc}\n\\toprule\n",
        "Setting & Transformer Acc & BiLSTM Acc & Transformer F1 & BiLSTM F1 \\\\\n\\midrule\n",
    ]
    for _, r in merged.iterrows():
        name = r["experiment"].replace("obfuscated_", "")
        tex_lines.append(
            f"{name} & {r['transformer_acc']*100:.1f}\\% & {r['bilstm_acc']*100:.1f}\\% & "
            f"{r['transformer_macro_f1']*100:.1f}\\% & {r['bilstm_macro_f1']*100:.1f}\\% \\\\\n"
        )
    tex_lines.append("\\bottomrule\n\\end{tabular}\n")
    tex_path = os.path.join(out_dir, "table_architecture_comparison.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.writelines(tex_lines)
    print(f"Saved {out}")
    print(f"Saved {tex_path}")
    return out


def run_bootstrap(
    predictions_dir: str,
    out_dir: str,
    n_bootstrap: int,
    seed: int,
) -> str:
    rows = []
    for npz_path in sorted(glob(os.path.join(predictions_dir, "*.npz"))):
        exp = Path(npz_path).stem
        data = load_predictions_npz(npz_path)
        stats = bootstrap_metrics(
            data["y_true"], data["y_pred"], n_bootstrap=n_bootstrap, seed=seed
        )
        rows.append({"experiment": exp, **stats})
        print(
            f"{exp}: acc {stats['accuracy']:.4f} "
            f"[{stats['accuracy_ci_low']:.4f}, {stats['accuracy_ci_high']:.4f}]"
        )

    if not rows:
        raise FileNotFoundError(f"No .npz files in {predictions_dir}")

    out = os.path.join(out_dir, "bootstrap_ci.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved -> {out}")
    return out


def run_paired_test(
    predictions_dir: str,
    out_dir: str,
    exp_a: str,
    exp_b: str,
    n_bootstrap: int,
    seed: int,
) -> str:
    path_a = os.path.join(predictions_dir, f"{exp_a}.npz")
    path_b = os.path.join(predictions_dir, f"{exp_b}.npz")
    da = load_predictions_npz(path_a)
    db = load_predictions_npz(path_b)
    if len(da["y_true"]) != len(db["y_true"]):
        raise ValueError("Prediction arrays must align (same test flows)")

    paired = paired_bootstrap_accuracy_diff(
        da["y_true"], da["y_pred"], db["y_pred"], n_bootstrap=n_bootstrap, seed=seed
    )
    mcn = mcnemar_test(da["y_true"], da["y_pred"], db["y_pred"])
    result = {
        "experiment_a": exp_a,
        "experiment_b": exp_b,
        "comparison": f"{exp_a}_vs_{exp_b}",
        **paired,
        **mcn,
    }
    out = os.path.join(out_dir, f"paired_test_{exp_a}_vs_{exp_b}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"Saved -> {out}")
    return out


def run_channel_ablation(
    device: str | None,
    batch_size: int,
    out_dir: str,
    checkpoint: str | None,
    config: str | None,
) -> str:
    ckpt = resolve_checkpoint(checkpoint, attack_model="transformer")
    cfg = resolve_model_config(ckpt, config, attack_model="transformer")
    rows = []
    for preset_name, channels in CHANNEL_ABLATION_PRESETS.items():
        for exp_name in CHANNEL_ABLATION_EXPERIMENTS:
            pt_path = TEST_TENSORS if exp_name == "baseline" else os.path.join(
                PHASE3_ARTIFACTS, f"{exp_name}.pt"
            )
            if not os.path.isfile(pt_path):
                continue
            tag = f"{exp_name}__{preset_name}"
            print(f"\n=== ablation {tag} channels={channels} ===")
            metrics = evaluate_one(
                pt_path,
                checkpoint=ckpt,
                config=cfg,
                device=device,
                experiment_name=tag,
                attack_model="transformer",
                active_channels=channels,
                batch_size=batch_size,
            )
            rows.append(
                {
                    "experiment": exp_name,
                    "channel_preset": preset_name,
                    "active_channels": ",".join(map(str, channels)),
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                }
            )
            print(f"  acc={metrics['accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}  "
              f"model={metrics.get('model_name', '?')}")

    out = os.path.join(out_dir, "channel_ablation.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved -> {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 Tier C analyses")
    parser.add_argument("--all", action="store_true", help="Run all Tier C steps")
    parser.add_argument("--bilstm", action="store_true", help="CNN-BiLSTM full eval")
    parser.add_argument("--transformer-predictions", action="store_true",
                        help="Re-save transformer predictions for bootstrap")
    parser.add_argument("--bootstrap", action="store_true", help="Bootstrap CIs from .npz")
    parser.add_argument("--paired", action="store_true", help="Paired baseline vs jitter_low")
    parser.add_argument("--channel-ablation", action="store_true", help="IPT/DIR/SIZE ablation")
    parser.add_argument("--summarize-reports", action="store_true", help="Worst-class summary")
    parser.add_argument("--compare-architectures", action="store_true",
                        help="Transformer vs BiLSTM table")
    parser.add_argument("--checkpoint", default=None, help="Transformer checkpoint path")
    parser.add_argument(
        "--bilstm-checkpoint",
        default=None,
        help="CNN-BiLSTM checkpoint (default: phase2/artifacts/cnn_bilstm_best.pt)",
    )
    parser.add_argument("--config", default=None, help="Transformer training config JSON")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--out-dir", default=PHASE4_RESULTS)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predictions-subdir", default="transformer",
                        help="Subdir under predictions/ for bootstrap")
    args = parser.parse_args()

    if not any(
        [
            args.all,
            args.bilstm,
            args.transformer_predictions,
            args.bootstrap,
            args.paired,
            args.channel_ablation,
            args.summarize_reports,
            args.compare_architectures,
        ]
    ):
        parser.error("Specify at least one action (e.g. --all)")

    os.makedirs(args.out_dir, exist_ok=True)
    pred_dir = os.path.join(args.out_dir, "predictions", args.predictions_subdir)

    if args.all or args.transformer_predictions:
        run_attack_eval(
            "transformer",
            args.device,
            args.batch_size,
            args.out_dir,
            args.checkpoint,
            args.config,
            save_predictions=True,
        )

    if args.all or args.bilstm:
        run_attack_eval(
            "cnn_bilstm",
            args.device,
            args.batch_size,
            args.out_dir,
            args.bilstm_checkpoint,
            None,
            save_predictions=True,
        )

    if args.all or args.compare_architectures:
        try:
            export_architecture_comparison(args.out_dir)
        except FileNotFoundError as exc:
            print(f"Skip architecture comparison: {exc}")

    if args.all or args.bootstrap:
        if not os.path.isdir(pred_dir):
            pred_dir = os.path.join(args.out_dir, "predictions", "transformer")
        run_bootstrap(pred_dir, args.out_dir, args.n_bootstrap, args.seed)

    if args.all or args.paired:
        if not os.path.isdir(pred_dir):
            pred_dir = os.path.join(args.out_dir, "predictions", "transformer")
        run_paired_test(
            pred_dir, args.out_dir, DEFAULT_PAIRED[0], DEFAULT_PAIRED[1],
            args.n_bootstrap, args.seed,
        )

    if args.all or args.channel_ablation:
        run_channel_ablation(
            args.device, args.batch_size, args.out_dir, args.checkpoint, args.config
        )

    if args.all or args.summarize_reports:
        reports = os.path.join(args.out_dir, "reports")
        write_worst_classes_summary(reports, os.path.join(args.out_dir, "worst_classes_summary.csv"))
        if os.path.isdir(pred_dir):
            for exp in DEFAULT_PAIRED:
                try:
                    write_confusion_pairs_from_npz(
                        pred_dir,
                        exp,
                        os.path.join(args.out_dir, f"confused_pairs_{exp}.csv"),
                    )
                except FileNotFoundError:
                    pass
        print("Summarized reports.")


if __name__ == "__main__":
    main()
