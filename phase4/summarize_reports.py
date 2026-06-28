"""
Summarize per-class reports for Discussion (Tier C5).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from config import PHASE4_RESULTS
from evaluate import load_class_names
from statistical_analysis import top_confused_pairs


def summarize_per_class_csv(
    csv_path: str,
    experiment: str,
    bottom_k: int = 10,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df.nsmallest(bottom_k, "f1").assign(experiment=experiment)


def write_worst_classes_summary(
    reports_dir: str,
    out_path: str,
    experiments: list[str] | None = None,
    bottom_k: int = 10,
) -> str:
    experiments = experiments or [
        "baseline",
        "obfuscated_jitter_low",
        "obfuscated_mtu",
    ]
    frames = []
    for exp in experiments:
        csv_path = os.path.join(reports_dir, f"{exp}_per_class.csv")
        if not os.path.isfile(csv_path):
            continue
        frames.append(summarize_per_class_csv(csv_path, exp, bottom_k=bottom_k))

    if not frames:
        raise FileNotFoundError(f"No per_class CSVs found in {reports_dir}")

    worst = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    worst.to_csv(out_path, index=False)

    lines = ["# Worst per-class F1 (Tier C summary)\n"]
    for exp in experiments:
        sub = worst[worst["experiment"] == exp]
        if sub.empty:
            continue
        label = exp.replace("obfuscated_", "")
        lines.append(f"\n## {label}\n")
        for _, row in sub.iterrows():
            lines.append(
                f"- **{row['class_name']}** (support={int(row['support'])}): "
                f"P={row['precision']:.3f} R={row['recall']:.3f} F1={row['f1']:.3f}\n"
            )

    md_path = out_path.replace(".csv", ".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return out_path


def write_confusion_pairs_from_npz(
    predictions_dir: str,
    experiment: str,
    out_path: str,
    top_k: int = 15,
) -> str:
    npz = os.path.join(predictions_dir, f"{experiment}.npz")
    if not os.path.isfile(npz):
        raise FileNotFoundError(npz)
    data = __import__("numpy").load(npz)
    names = load_class_names()
    pairs = top_confused_pairs(
        data["y_true"], data["y_pred"], class_names=names, top_k=top_k
    )
    df = pd.DataFrame(pairs)
    df.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize worst-class F1 from reports")
    parser.add_argument("--reports-dir", default=os.path.join(PHASE4_RESULTS, "reports"))
    parser.add_argument("--out", default=os.path.join(PHASE4_RESULTS, "worst_classes_summary.csv"))
    parser.add_argument("--bottom-k", type=int, default=10)
    args = parser.parse_args()
    path = write_worst_classes_summary(args.reports_dir, args.out, bottom_k=args.bottom_k)
    print(f"Wrote {path}")
    print(f"Wrote {path.replace('.csv', '.md')}")


if __name__ == "__main__":
    main()
