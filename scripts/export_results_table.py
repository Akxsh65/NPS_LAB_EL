#!/usr/bin/env python3
"""
Export Phase 4 summary table to LaTeX (Table 1: macro F1 lead, accuracy in parentheses).

Usage (from repo root):
  python scripts/export_results_table.py
  python scripts/export_results_table.py --csv phase4/accuracy_results.csv --out phase4/results/table_phase4.tex
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PHASE4 = ROOT / "phase4"
sys.path.insert(0, str(PHASE4))

from plot_publication import export_latex_table  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Phase 4 results table to LaTeX")
    parser.add_argument(
        "--csv",
        default=str(PHASE4 / "accuracy_results.csv"),
        help="accuracy_results.csv path",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output .tex path (default: same dir as CSV -> table_phase4.tex)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "baseline" not in df["experiment"].values:
        raise SystemExit("CSV must include a baseline row")

    if "macro_f1_drop" not in df.columns:
        bl = df.loc[df["experiment"] == "baseline"].iloc[0]
        df["macro_f1_drop"] = bl["macro_f1"] - df["macro_f1"]
        df["accuracy_drop"] = bl["accuracy"] - df["accuracy"]

    out_dir = str(Path(args.out).parent) if args.out else str(csv_path.parent)
    path = export_latex_table(df, out_dir)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(Path(path).read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Wrote {out_path}")
    else:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
