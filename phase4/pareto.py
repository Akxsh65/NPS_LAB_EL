"""
Pareto-style plots: privacy (accuracy) vs cost (bandwidth / latency overhead).
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_all(csv_path: str, out_dir: str) -> None:
    df = pd.read_csv(csv_path)
    if "baseline" not in df["experiment"].values:
        raise ValueError("CSV must include a 'baseline' row")

    baseline = df[df["experiment"] == "baseline"].iloc[0]
    defended = df[df["experiment"] != "baseline"].copy()

    os.makedirs(out_dir, exist_ok=True)
    sns.set_style("whitegrid")

    # --- Bandwidth vs accuracy ---
    plt.figure(figsize=(8, 6))
    plt.scatter(
        defended["mean_bandwidth_overhead"] * 100,
        defended["accuracy"] * 100,
        s=80,
        alpha=0.85,
    )
    for _, row in defended.iterrows():
        plt.annotate(
            row["experiment"].replace("obfuscated_", ""),
            (row["mean_bandwidth_overhead"] * 100, row["accuracy"] * 100),
            fontsize=7,
            alpha=0.8,
        )
    plt.axhline(baseline["accuracy"] * 100, color="green", linestyle="--", label="Baseline acc.")
    plt.xlabel("Mean bandwidth overhead (%)")
    plt.ylabel("Test accuracy (%)")
    plt.title("Privacy vs bandwidth cost (Pareto view)")
    plt.legend()
    plt.tight_layout()
    p1 = os.path.join(out_dir, "pareto_bandwidth_accuracy.png")
    plt.savefig(p1, dpi=150)
    plt.close()

    # --- Latency vs accuracy ---
    plt.figure(figsize=(8, 6))
    plt.scatter(
        defended["mean_latency_overhead_ms"],
        defended["accuracy"] * 100,
        s=80,
        alpha=0.85,
        color="coral",
    )
    for _, row in defended.iterrows():
        plt.annotate(
            row["experiment"].replace("obfuscated_", ""),
            (row["mean_latency_overhead_ms"], row["accuracy"] * 100),
            fontsize=7,
            alpha=0.8,
        )
    plt.axhline(baseline["accuracy"] * 100, color="green", linestyle="--", label="Baseline acc.")
    plt.xlabel("Mean injected latency per flow (ms)")
    plt.ylabel("Test accuracy (%)")
    plt.title("Privacy vs latency cost")
    plt.legend()
    plt.tight_layout()
    p2 = os.path.join(out_dir, "pareto_latency_accuracy.png")
    plt.savefig(p2, dpi=150)
    plt.close()

    # --- Accuracy drop bar chart ---
    plt.figure(figsize=(10, 5))
    order = defended.sort_values("accuracy_drop_pct", ascending=False)
    sns.barplot(
        data=order,
        x="experiment",
        y="accuracy_drop_pct",
        color="steelblue",
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Accuracy drop vs baseline (%)")
    plt.xlabel("Obfuscation setting")
    plt.title("Classifier degradation under defense")
    plt.tight_layout()
    p3 = os.path.join(out_dir, "accuracy_drop_bars.png")
    plt.savefig(p3, dpi=150)
    plt.close()

    # --- Baseline vs defended grouped bars ---
    plot_df = df[["experiment", "accuracy"]].copy()
    plot_df["accuracy"] *= 100
    plt.figure(figsize=(10, 5))
    sns.barplot(data=plot_df, x="experiment", y="accuracy", color="teal")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Test accuracy (%)")
    plt.xlabel("Setting")
    plt.title("Baseline vs obfuscated test accuracy")
    plt.tight_layout()
    p4 = os.path.join(out_dir, "accuracy_comparison_bars.png")
    plt.savefig(p4, dpi=150)
    plt.close()

    print(f"Saved plots:\n  {p1}\n  {p2}\n  {p3}\n  {p4}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Pareto plots from results CSV")
    parser.add_argument("--csv", default=os.path.join("results", "accuracy_results.csv"))
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()
    plot_all(args.csv, args.out_dir)


if __name__ == "__main__":
    main()
