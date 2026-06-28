"""
Tier A/B publication plots and tables for Phase 4 results.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pareto_frontier import (
    build_frontier_table,
    frontier_curve_points,
    _defended_df,
)
from publication_style import (
    RECOMMENDED_EXPERIMENT,
    add_chance_line,
    bar_colors,
    highlight_recommended_bar,
    highlight_recommended_scatter,
)

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 10})

PRACTICAL_EXCLUDE_MTU = True


def _baseline_row(df: pd.DataFrame) -> pd.Series:
    return df.loc[df["experiment"] == "baseline"].iloc[0]


def _label(exp: str) -> str:
    return exp.replace("obfuscated_", "")


def _barplot_with_tier_b(
    ax,
    plot_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    y_in_percent: bool = True,
    show_baseline_line: bool = False,
    baseline_y: float | None = None,
) -> None:
    exps = plot_df[x_col].tolist()
    heights = plot_df[y_col].tolist()
    x = np.arange(len(exps))
    ax.bar(x, heights, color=bar_colors(exps, default="mediumpurple"))
    add_chance_line(ax, y_in_percent=y_in_percent)
    if show_baseline_line and baseline_y is not None:
        ax.axhline(
            baseline_y * (100.0 if y_in_percent else 1.0),
            color="green",
            ls="--",
            lw=1.5,
            label="Baseline",
        )
    highlight_recommended_bar(ax, exps, heights)
    ax.set_xticks(x)
    ax.set_xticklabels([_label(e) if e != "baseline" else "baseline" for e in exps], rotation=45, ha="right")


def plot_macro_f1_comparison(df: pd.DataFrame, out_dir: str) -> str:
    bl = _baseline_row(df)
    plot_df = df.copy()
    plot_df["macro_f1_pct"] = plot_df["macro_f1"] * 100
    plot_df = plot_df.sort_values("macro_f1_pct", ascending=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    _barplot_with_tier_b(
        ax,
        plot_df,
        "experiment",
        "macro_f1_pct",
        y_in_percent=True,
        show_baseline_line=True,
        baseline_y=float(bl["macro_f1"]),
    )
    ax.set_ylabel("Test macro F1 (%)")
    ax.set_xlabel("Setting")
    ax.set_title("Test macro F1 by obfuscation setting")
    ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "macro_f1_comparison_bars.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def plot_macro_f1_drop(df: pd.DataFrame, out_dir: str) -> str:
    defended = df[df["experiment"] != "baseline"].sort_values("macro_f1_drop", ascending=False).copy()
    defended["macro_f1_drop_pct"] = defended["macro_f1_drop"] * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    _barplot_with_tier_b(ax, defended, "experiment", "macro_f1_drop_pct", y_in_percent=True)
    ax.set_ylabel("Macro F1 drop vs baseline (pp)")
    ax.set_xlabel("Obfuscation setting")
    ax.set_title("Privacy gain (macro F1 reduction under defense)")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "macro_f1_drop_bars.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def plot_dual_metric_top5(df: pd.DataFrame, out_dir: str) -> str:
    """Side-by-side accuracy and macro F1 for top-5 defenses by macro_f1_drop."""
    bl = _baseline_row(df)
    defended = df[df["experiment"] != "baseline"].copy()
    top = defended.nlargest(5, "macro_f1_drop")
    labels = [_label(e) for e in top["experiment"]]
    exps = top["experiment"].tolist()
    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    acc_colors = [
        "#DAA520" if e == RECOMMENDED_EXPERIMENT else "steelblue" for e in exps
    ]
    f1_colors = [
        "#DAA520" if e == RECOMMENDED_EXPERIMENT else "mediumpurple" for e in exps
    ]
    acc_h = top["accuracy"].values * 100
    f1_h = top["macro_f1"].values * 100
    ax.bar(x - w / 2, acc_h, w, label="Accuracy", color=acc_colors)
    ax.bar(x + w / 2, f1_h, w, label="Macro F1", color=f1_colors)
    add_chance_line(ax, y_in_percent=True)
    ax.axhline(float(bl["accuracy"]) * 100, color="steelblue", ls="--", alpha=0.6, label="Baseline acc")
    ax.axhline(float(bl["macro_f1"]) * 100, color="mediumpurple", ls="--", alpha=0.6, label="Baseline macro F1")
    highlight_recommended_bar(ax, exps, np.maximum(acc_h, f1_h))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Test score (%)")
    ax.set_title("Top-5 defenses by macro F1 drop — accuracy vs macro F1")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "dual_metric_top5_bars.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def plot_accuracy_comparison(df: pd.DataFrame, out_dir: str) -> str:
    plot_df = df[["experiment", "accuracy"]].copy()
    plot_df["accuracy_pct"] = plot_df["accuracy"] * 100
    plot_df = plot_df.sort_values("accuracy_pct", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    _barplot_with_tier_b(ax, plot_df, "experiment", "accuracy_pct", y_in_percent=True)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xlabel("Setting")
    ax.set_title("Baseline vs obfuscated test accuracy")
    ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "accuracy_comparison_bars.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def plot_accuracy_drop(df: pd.DataFrame, out_dir: str) -> str:
    defended = df[df["experiment"] != "baseline"].copy()
    if "accuracy_drop_pct" not in defended.columns:
        bl = _baseline_row(df)
        defended["accuracy_drop_pct"] = (bl["accuracy"] - defended["accuracy"]) / bl["accuracy"] * 100
    defended = defended.sort_values("accuracy_drop_pct", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    _barplot_with_tier_b(ax, defended, "experiment", "accuracy_drop_pct", y_in_percent=True)
    ax.set_ylabel("Accuracy drop vs baseline (%)")
    ax.set_xlabel("Obfuscation setting")
    ax.set_title("Classifier degradation under defense")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "accuracy_drop_bars.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def _scatter_pareto(
    df: pd.DataFrame,
    cost_col: str,
    cost_label: str,
    title: str,
    out_path: str,
    include_mtu: bool,
    log_cost: bool = False,
    baseline_acc: float | None = None,
) -> None:
    baseline = df[df["experiment"] == "baseline"].iloc[0]
    defended = _defended_df(df, include_mtu=include_mtu)
    ft = build_frontier_table(df, cost_col, include_mtu=include_mtu).set_index("experiment")

    x_vals, y_vals, colors = [], [], []
    for _, row in defended.iterrows():
        c = float(row[cost_col])
        x_vals.append(c * 100 if cost_col == "mean_bandwidth_overhead" else c)
        y_vals.append(float(row["accuracy"]) * 100)
        colors.append("coral" if ft.loc[row["experiment"], "is_dominated"] else "tab:blue")

    front = frontier_curve_points(df, cost_col, include_mtu=include_mtu)

    plt.figure(figsize=(9, 6))
    plt.scatter(x_vals, y_vals, s=90, c=colors, alpha=0.85, edgecolors="k", linewidths=0.3)

    rec_x, rec_y = None, None
    for x, y, (_, row) in zip(x_vals, y_vals, defended.iterrows()):
        exp = row["experiment"]
        if exp == RECOMMENDED_EXPERIMENT:
            rec_x, rec_y = x, y
        else:
            plt.annotate(_label(exp), (x, y), fontsize=7, alpha=0.85)

    if rec_x is not None:
        highlight_recommended_scatter(plt.gca(), rec_x, rec_y)
        plt.annotate("jitter_low", (rec_x, rec_y), fontsize=7, xytext=(5, 5), textcoords="offset points")

    if not front.empty:
        if cost_col == "mean_bandwidth_overhead":
            fx = front["cost"].values * 100
        else:
            fx = front["cost"].values
        fy = front["accuracy"].values * 100
        plt.plot(fx, fy, "r-", lw=2, marker="o", markersize=6, label="Pareto frontier", zorder=5)

    b_acc = baseline_acc if baseline_acc is not None else float(baseline["accuracy"])
    plt.axhline(b_acc * 100, color="green", ls="--", lw=1.5, label="Baseline accuracy")
    add_chance_line(plt.gca(), y_in_percent=True)
    if log_cost and len(x_vals) and min(x_vals) > 0:
        plt.xscale("log")
    plt.xlabel(cost_label)
    plt.ylabel("Attacker test accuracy (%) — lower is better privacy")
    plt.title(title)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_pareto_bw_practical(df: pd.DataFrame, out_dir: str) -> str:
    path = os.path.join(out_dir, "pareto_bw_accuracy_practical.png")
    _scatter_pareto(
        df,
        "mean_bandwidth_overhead",
        "Mean bandwidth overhead (%)",
        "Privacy–bandwidth tradeoff (practical defenses, Pareto frontier)",
        path,
        include_mtu=False,
        log_cost=False,
    )
    return path


def plot_pareto_lat_practical(df: pd.DataFrame, out_dir: str) -> str:
    path = os.path.join(out_dir, "pareto_latency_accuracy_practical.png")
    _scatter_pareto(
        df,
        "mean_latency_overhead_ms",
        "Mean injected latency per flow (ms)",
        "Privacy–latency tradeoff (practical defenses, Pareto frontier)",
        path,
        include_mtu=False,
        log_cost=False,
    )
    return path


def plot_pareto_bw_full(df: pd.DataFrame, out_dir: str) -> str:
    path = os.path.join(out_dir, "pareto_bw_accuracy_full.png")
    _scatter_pareto(
        df,
        "mean_bandwidth_overhead",
        "Mean bandwidth overhead (%) — log scale",
        "Privacy–bandwidth tradeoff (all defenses incl. MTU, log cost)",
        path,
        include_mtu=True,
        log_cost=True,
    )
    return path


def export_frontier_tables(df: pd.DataFrame, out_dir: str) -> list[str]:
    paths = []
    for cost_col in ("mean_bandwidth_overhead", "mean_latency_overhead_ms"):
        for include_mtu, tag in ((False, "practical"), (True, "full")):
            table = build_frontier_table(df, cost_col, include_mtu=include_mtu)
            fname = f"pareto_frontier_{cost_col.replace('mean_', '').replace('_overhead', '')}_{tag}.csv"
            p = os.path.join(out_dir, fname)
            table.to_csv(p, index=False)
            paths.append(p)
    main_path = os.path.join(out_dir, "pareto_frontier_table.csv")
    build_frontier_table(df, "mean_bandwidth_overhead", include_mtu=False).to_csv(
        main_path, index=False
    )
    paths.append(main_path)
    return paths


def export_latex_table(df: pd.DataFrame, out_dir: str) -> str:
    """Table 1 style: macro F1 lead, accuracy in parens."""
    rows = []
    for _, r in df.iterrows():
        name = _label(r["experiment"]) if r["experiment"] != "baseline" else "baseline"
        mdrop = float(r["macro_f1_drop"]) if pd.notna(r.get("macro_f1_drop")) else 0.0
        rows.append(
            f"{name} & {r['macro_f1']*100:.1f}\\% ({r['accuracy']*100:.1f}\\%) & "
            f"{mdrop*100:.1f} & "
            f"{float(r['mean_bandwidth_overhead'])*100:.1f} & "
            f"{float(r['mean_latency_overhead_ms']):.1f} \\\\"
        )
    tex = (
        "\\begin{tabular}{lcccc}\n\\toprule\n"
        "Setting & Macro F1 (Acc) & $\\Delta$ Macro F1 & BW OH \\% & Lat OH ms \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    path = os.path.join(out_dir, "table_phase4.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(tex)
    return path


def _prepare_results_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "baseline" not in df["experiment"].values:
        raise ValueError("CSV must include baseline row")
    if "macro_f1_drop" not in df.columns:
        bl = df[df["experiment"] == "baseline"].iloc[0]
        df["macro_f1_drop"] = bl["macro_f1"] - df["macro_f1"]
        df["accuracy_drop"] = bl["accuracy"] - df["accuracy"]
    if "accuracy_drop_pct" not in df.columns:
        bl = df[df["experiment"] == "baseline"].iloc[0]
        df["accuracy_drop_pct"] = (df["accuracy_drop"] / bl["accuracy"]) * 100.0
    return df


def plot_tier_a(csv_path: str, out_dir: str) -> list[str]:
    df = _prepare_results_df(csv_path)
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    saved.append(plot_macro_f1_comparison(df, out_dir))
    saved.append(plot_macro_f1_drop(df, out_dir))
    saved.append(plot_dual_metric_top5(df, out_dir))
    saved.append(plot_pareto_bw_practical(df, out_dir))
    saved.append(plot_pareto_lat_practical(df, out_dir))
    saved.append(plot_pareto_bw_full(df, out_dir))
    saved.extend(export_frontier_tables(df, out_dir))
    saved.append(export_latex_table(df, out_dir))
    return saved


def plot_tier_b(csv_path: str, out_dir: str) -> list[str]:
    """Tier B extras: legacy accuracy bar charts with chance line + jitter_low star."""
    df = _prepare_results_df(csv_path)
    os.makedirs(out_dir, exist_ok=True)
    return [
        plot_accuracy_comparison(df, out_dir),
        plot_accuracy_drop(df, out_dir),
    ]


def plot_all_publication(csv_path: str, out_dir: str) -> list[str]:
    """Tier A outputs (bar/scatter styling includes Tier B) + Tier B accuracy bars."""
    paths = plot_tier_a(csv_path, out_dir)
    paths.extend(plot_tier_b(csv_path, out_dir))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 publication plots (Tier A/B)")
    parser.add_argument("--csv", default=os.path.join("results", "accuracy_results.csv"))
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--tier-a-only", action="store_true")
    parser.add_argument("--tier-b-only", action="store_true")
    args = parser.parse_args()

    if args.tier_b_only:
        paths = plot_tier_b(args.csv, args.out_dir)
        label = "Tier B"
    elif args.tier_a_only:
        paths = plot_tier_a(args.csv, args.out_dir)
        label = "Tier A"
    else:
        paths = plot_all_publication(args.csv, args.out_dir)
        label = "Tier A+B"

    print(f"Saved {label} outputs:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
