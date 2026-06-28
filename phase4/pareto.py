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
    plt.title("Privacy vs bandwidth cost (tradeoff scatter)")
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

    print(f"Saved plots:\n  {p1}\n  {p2}")

    try:
        from plot_publication import plot_all_publication

        pub_paths = plot_all_publication(csv_path, out_dir)
        print("Saved Tier A+B publication outputs:")
        for p in pub_paths:
            print(f"  {p}")
    except Exception as exc:
        print(f"Warning: publication plots failed: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Pareto plots from results CSV")
    parser.add_argument("--csv", default=os.path.join("results", "accuracy_results.csv"))
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--tier-a-only", action="store_true", help="Only Tier A publication plots")
    parser.add_argument("--tier-b-only", action="store_true", help="Only Tier B accuracy bar charts")
    args = parser.parse_args()
    if args.tier_a_only:
        from plot_publication import plot_tier_a

        plot_tier_a(args.csv, args.out_dir)
    elif args.tier_b_only:
        from plot_publication import plot_tier_b

        plot_tier_b(args.csv, args.out_dir)
    else:
        plot_all(args.csv, args.out_dir)


if __name__ == "__main__":
    main()
