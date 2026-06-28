"""
Research-grade visualization for hyperparameter sweep results.

Expects sweep layout from hyperparameter_sweep.py:
  artifacts/sweeps/run_01_bs1024_lr0.0003_wd0.01/
    transformer_history.csv   # epoch,lr,train_loss,train_acc,val_loss,val_acc
    transformer_config.json

Usage (from phase2/):
  python plot_sweep_results.py
  python plot_sweep_results.py --sweep-dir ./artifacts/sweeps --model transformer
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

# ── Publication style ───────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "lines.linewidth": 1.8,
    }
)

RUN_DIR_RE = re.compile(
    r"run_(\d+)_bs(\d+)_lr([\d.eE+-]+)_wd([\d.eE+-]+)"
)

METRIC_COLS = ["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc"]


def _parse_run_dir(name: str) -> Dict:
    m = RUN_DIR_RE.match(name)
    if m:
        return {
            "run_id": int(m.group(1)),
            "batch_size": int(m.group(2)),
            "lr": float(m.group(3)),
            "weight_decay": float(m.group(4)),
        }
    return {"run_id": None, "batch_size": None, "lr": None, "weight_decay": None}


def discover_runs(sweep_dir: Path, model: str) -> List[Dict]:
    """Collect history CSV + config JSON for each sweep subdirectory."""
    history_name = f"{model}_history.csv"
    config_name = f"{model}_config.json"
    runs = []

    for sub in sorted(sweep_dir.iterdir()):
        if not sub.is_dir():
            continue
        csv_path = sub / history_name
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        for c in METRIC_COLS:
            if c not in df.columns:
                raise ValueError(f"{csv_path} missing column '{c}'")

        meta = _parse_run_dir(sub.name)
        meta["run_dir"] = str(sub)
        meta["run_label"] = sub.name.replace("run_", "R")

        cfg_path = sub / config_name
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            meta["batch_size"] = cfg.get("batch_size", meta["batch_size"])
            meta["lr"] = cfg.get("lr", meta["lr"])
            meta["weight_decay"] = cfg.get("weight_decay", meta["weight_decay"])
            meta["label_smoothing"] = cfg.get("label_smoothing")
            meta["epochs"] = cfg.get("epochs")
            meta["patience"] = cfg.get("patience")

        runs.append(
            {
                **meta,
                "history": df,
                "csv_path": str(csv_path),
            }
        )

    return runs


def enrich_history(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived metrics for overfitting / generalization analysis."""
    out = df.copy()
    out["acc_gap"] = out["train_acc"] - out["val_acc"]  # positive => train > val
    out["loss_gap"] = out["val_loss"] - out["train_loss"]  # positive => val worse
    out["train_acc_pct"] = out["train_acc"] * 100
    out["val_acc_pct"] = out["val_acc"] * 100
    out["train_loss"] = out["train_loss"]
    out["val_loss"] = out["val_loss"]
    return out


def _format_config_label(batch_size, lr, weight_decay) -> str:
    lr_s = f"{lr:.0e}" if lr is not None and pd.notna(lr) else "NA"
    wd_s = f"{weight_decay:.0e}" if weight_decay is not None and pd.notna(weight_decay) else "NA"
    bs_s = batch_size if batch_size is not None else "NA"
    return f"bs={bs_s}, lr={lr_s}, wd={wd_s}"


def summarize_run(run: Dict) -> Dict:
    df = enrich_history(run["history"])
    best_idx = df["val_acc"].idxmax()
    best_row = df.loc[best_idx]
    final = df.iloc[-1]
    bs = run.get("batch_size")
    lr = run.get("lr")
    wd = run.get("weight_decay")

    return {
        "run_label": run["run_label"],
        "run_dir": run["run_dir"],
        "batch_size": bs,
        "lr": lr,
        "weight_decay": wd,
        "config_label": _format_config_label(bs, lr, wd),
        "epochs_ran": len(df),
        "best_epoch": int(best_row["epoch"]),
        "best_val_acc": float(best_row["val_acc"]),
        "best_val_loss": float(best_row["val_loss"]),
        "train_acc_at_best": float(best_row["train_acc"]),
        "acc_gap_at_best": float(best_row["acc_gap"]),
        "loss_gap_at_best": float(best_row["loss_gap"]),
        "final_val_acc": float(final["val_acc"]),
        "final_train_acc": float(final["train_acc"]),
        "final_acc_gap": float(final["acc_gap"]),
        "max_acc_gap": float(df["acc_gap"].max()),
        "mean_acc_gap_last5": float(df["acc_gap"].tail(5).mean()),
        "final_lr": float(final["lr"]),
    }


def build_summary_table(runs: List[Dict]) -> pd.DataFrame:
    rows = [summarize_run(r) for r in runs]
    df = pd.DataFrame(rows)
    df = df.sort_values("best_val_acc", ascending=False).reset_index(drop=True)
    df["config_label"] = df.apply(
        lambda r: _format_config_label(r["batch_size"], r["lr"], r["weight_decay"]), axis=1
    )
    return df


def _overfit_status(gap: float) -> str:
    """Heuristic labels for acc_gap = train_acc - val_acc."""
    if gap < 0.03:
        return "Good fit"
    if gap < 0.08:
        return "Mild gap"
    if gap < 0.15:
        return "Moderate overfit"
    return "Strong overfit"


def plot_per_run_curves(run: Dict, out_path: Path, model: str) -> None:
    """2x2 panel: acc, loss, gaps, LR — train vs val."""
    df = enrich_history(run["history"])
    label = run["run_label"]
    bs, lr, wd = run.get("batch_size"), run.get("lr"), run.get("weight_decay")

    fig = plt.figure(figsize=(11, 9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.28)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(df["epoch"], df["train_acc_pct"], "o-", label="Train acc", color="#2166ac", markersize=3)
    ax0.plot(df["epoch"], df["val_acc_pct"], "s-", label="Val acc", color="#b2182b", markersize=3)
    best_ep = df.loc[df["val_acc"].idxmax(), "epoch"]
    ax0.axvline(best_ep, color="gray", ls="--", lw=1, alpha=0.7, label=f"Best val @ ep {int(best_ep)}")
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Accuracy (%)")
    ax0.set_title(f"{label} - Accuracy (overfitting if train >> val)")
    ax0.legend(loc="lower right", frameon=True)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(df["epoch"], df["train_loss"], "o-", label="Train loss", color="#2166ac", markersize=3)
    ax1.plot(df["epoch"], df["val_loss"], "s-", label="Val loss", color="#b2182b", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.set_title(f"{label} - Loss (val > train often indicates overfit)")
    ax1.legend(loc="upper right", frameon=True)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(df["epoch"], df["acc_gap"] * 100, color="#4d9221", lw=2)
    ax2.axhline(3, color="orange", ls=":", lw=1, label="3% gap (mild)")
    ax2.axhline(8, color="red", ls=":", lw=1, label="8% gap (moderate)")
    ax2.fill_between(df["epoch"], 0, df["acc_gap"] * 100, alpha=0.15, color="#4d9221")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Train - Val accuracy (%)")
    ax2.set_title("Generalization gap (train acc - val acc)")
    ax2.legend(loc="upper left", frameon=True)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(df["epoch"], df["loss_gap"], color="#762a83", lw=2)
    ax3.axhline(0, color="black", lw=0.8)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Val - Train loss")
    ax3.set_title("Loss gap (val - train)")

    fig.suptitle(
        f"{model} sweep run | bs={bs}, lr={lr:.0e}, wd={wd:.0e}\n"
        f"Status @ best epoch: {_overfit_status(float(df.loc[df['val_acc'].idxmax(), 'acc_gap']))}",
        fontsize=12,
        y=1.02,
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_comparative_best_metrics(summary: pd.DataFrame, out_dir: Path) -> None:
    """Bar charts: best val acc, acc gap at best, best val loss."""
    order = summary.sort_values("best_val_acc", ascending=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(order)))
    y = np.arange(len(order))
    labels = order["config_label"].tolist()

    axes[0].barh(y, order["best_val_acc"] * 100, color=colors)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].set_xlabel("Best validation accuracy (%)")
    axes[0].set_title("Peak val accuracy per sweep")

    gap_colors = [
        "#1b7837" if g < 0.05 else "#fdae61" if g < 0.10 else "#d73027"
        for g in order["acc_gap_at_best"]
    ]
    axes[1].barh(y, order["acc_gap_at_best"] * 100, color=gap_colors)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels, fontsize=7)
    axes[1].set_xlabel("Train - Val acc at best epoch (%)")
    axes[1].set_title("Overfitting proxy @ best val epoch")
    axes[1].axvline(5, color="gray", ls="--", lw=0.8)
    axes[1].axvline(10, color="gray", ls=":", lw=0.8)

    axes[2].barh(y, order["best_val_loss"], color=colors)
    axes[2].set_yticks(y)
    axes[2].set_yticklabels(labels, fontsize=7)
    axes[2].set_xlabel("Val loss at best-acc epoch")
    axes[2].set_title("Validation loss @ best epoch")

    plt.tight_layout()
    plt.savefig(out_dir / "comparative_best_metrics.png", bbox_inches="tight")
    plt.close(fig)


def plot_heatmap_lr_wd(summary: pd.DataFrame, out_dir: Path, value_col: str, title: str, fname: str) -> None:
    sub = summary.dropna(subset=["lr", "weight_decay"])
    if sub.empty or sub["batch_size"].nunique() > 1:
        # If multiple batch sizes, facet by batch size
        fig, axes = plt.subplots(
            1,
            sub["batch_size"].nunique(),
            figsize=(5 * sub["batch_size"].nunique(), 4),
            squeeze=False,
        )
        for ax, (bs, grp) in zip(axes[0], sub.groupby("batch_size")):
            pivot = grp.pivot_table(index="weight_decay", columns="lr", values=value_col, aggfunc="max")
            pivot = pivot.sort_index(ascending=False)
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".3f" if value_col != "best_val_acc" else ".2%",
                cmap="YlGnBu" if "acc" in value_col else "YlOrRd_r",
                ax=ax,
                cbar_kws={"label": value_col},
            )
            ax.set_title(f"bs={bs}")
            ax.set_xlabel("Learning rate")
            ax.set_ylabel("Weight decay")
        fig.suptitle(title, y=1.02)
        plt.tight_layout()
        plt.savefig(out_dir / fname, bbox_inches="tight")
        plt.close(fig)
        return

    pivot = sub.pivot_table(index="weight_decay", columns="lr", values=value_col, aggfunc="max")
    pivot = pivot.sort_index(ascending=False)
    plt.figure(figsize=(7, 5))
    fmt = ".1%" if value_col == "best_val_acc" else ".3f"
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap="YlGnBu", cbar_kws={"label": value_col})
    plt.title(title)
    plt.xlabel("Learning rate")
    plt.ylabel("Weight decay")
    plt.tight_layout()
    plt.savefig(out_dir / fname, bbox_inches="tight")
    plt.close()


def plot_overlay_val_curves(runs: List[Dict], out_dir: Path, model: str, top_k: int = 12) -> None:
    """Overlay validation accuracy curves for top-k runs by best val acc."""
    summaries = [(summarize_run(r), r) for r in runs]
    summaries.sort(key=lambda x: x[0]["best_val_acc"], reverse=True)
    top = summaries[:top_k]

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab20(np.linspace(0, 1, len(top)))

    for i, (summ, run) in enumerate(top):
        df = enrich_history(run["history"])
        ax.plot(
            df["epoch"],
            df["val_acc_pct"],
            label=f"{summ['config_label']} (best={summ['best_val_acc']:.1%})",
            color=cmap[i],
            alpha=0.85,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title(f"{model}: top-{len(top)} sweep runs - val accuracy trajectories")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, frameon=True)
    plt.tight_layout()
    plt.savefig(out_dir / "comparative_val_acc_overlay.png", bbox_inches="tight")
    plt.close(fig)


def plot_overlay_gaps(runs: List[Dict], out_dir: Path, model: str, top_k: int = 12) -> None:
    """Overlay train-val accuracy gap — direct overfitting comparison across sweeps."""
    summaries = [(summarize_run(r), r) for r in runs]
    summaries.sort(key=lambda x: x[0]["best_val_acc"], reverse=True)
    top = summaries[:top_k]

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab20(np.linspace(0, 1, len(top)))

    for i, (summ, run) in enumerate(top):
        df = enrich_history(run["history"])
        ax.plot(
            df["epoch"],
            df["acc_gap"] * 100,
            label=summ["config_label"],
            color=cmap[i],
            alpha=0.85,
        )

    ax.axhline(5, color="gray", ls="--", lw=1, label="5% gap")
    ax.axhline(10, color="gray", ls=":", lw=1, label="10% gap")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train - Val accuracy (%)")
    ax.set_title("Overfitting comparison: generalization gap across top runs")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "comparative_acc_gap_overlay.png", bbox_inches="tight")
    plt.close(fig)


def plot_train_vs_val_scatter(summary: pd.DataFrame, out_dir: Path) -> None:
    """Scatter at best epoch: train acc vs val acc (distance from diagonal = overfit)."""
    fig, ax = plt.subplots(figsize=(7, 7))
    x = summary["train_acc_at_best"] * 100
    y = summary["best_val_acc"] * 100
    sizes = 80 + 40 * summary["acc_gap_at_best"].clip(0, 0.2) / 0.2

    sc = ax.scatter(x, y, s=sizes, c=summary["acc_gap_at_best"], cmap="RdYlGn_r", vmin=0, vmax=0.15, edgecolors="k", lw=0.5)
    lim = [min(x.min(), y.min()) - 2, max(x.max(), y.max()) + 2]
    ax.plot(lim, lim, "k--", lw=1, label="Perfect agreement (no gap)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Train accuracy at best-val epoch (%)")
    ax.set_ylabel("Best validation accuracy (%)")
    ax.set_title("Train vs val @ best epoch\n(size/color ~ overfitting gap)")
    ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax, label="Acc gap (train - val)")

    for _, row in summary.iterrows():
        ax.annotate(
            row["run_label"],
            (row["train_acc_at_best"] * 100, row["best_val_acc"] * 100),
            fontsize=6,
            alpha=0.8,
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "train_vs_val_scatter.png", bbox_inches="tight")
    plt.close(fig)


def plot_lr_schedule_samples(runs: List[Dict], out_dir: Path, n: int = 4) -> None:
    """Cosine LR decay curves for a few representative runs."""
    summaries = sorted([summarize_run(r) for r in runs], key=lambda s: s["best_val_acc"], reverse=True)
    pick_ids = {s["run_label"] for s in summaries[: n // 2]} | {s["run_label"] for s in summaries[-n // 2 :]}
    fig, ax = plt.subplots(figsize=(8, 4))
    for run in runs:
        if run["run_label"] not in pick_ids:
            continue
        df = run["history"]
        ax.plot(df["epoch"], df["lr"], label=run["run_label"], alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("LR schedule (cosine annealing) - sample runs")
    ax.legend(fontsize=7)
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(out_dir / "lr_schedule_samples.png", bbox_inches="tight")
    plt.close(fig)


def plot_final_vs_best_gap(summary: pd.DataFrame, out_dir: Path) -> None:
    """Compare best val acc vs final-epoch val acc (stopped early / drift)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(summary))
    w = 0.35
    ax.bar(x - w / 2, summary["best_val_acc"] * 100, w, label="Best val acc", color="#2166ac")
    ax.bar(x + w / 2, summary["final_val_acc"] * 100, w, label="Final epoch val acc", color="#92c5de")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["run_label"], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Best vs final val accuracy (large drop -> late overfitting or bad stop)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "best_vs_final_val_acc.png", bbox_inches="tight")
    plt.close(fig)


def plot_metric_grid_all_runs(runs: List[Dict], out_dir: Path, model: str) -> None:
    """Small multiples: val acc for every run in a grid."""
    n = len(runs)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for idx, run in enumerate(runs):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        df = enrich_history(run["history"])
        ax.plot(df["epoch"], df["train_acc_pct"], color="#2166ac", alpha=0.7, label="Train")
        ax.plot(df["epoch"], df["val_acc_pct"], color="#b2182b", alpha=0.9, label="Val")
        ax.set_title(run["run_label"], fontsize=8)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=6)
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")
    fig.suptitle(f"{model}: train/val accuracy - all sweep runs", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(out_dir / "grid_all_runs_train_val.png", bbox_inches="tight")
    plt.close(fig)


def generate_report(
    sweep_dir: Path,
    model: str = "transformer",
    out_dir: Optional[Path] = None,
    top_k_overlay: int = 12,
) -> pd.DataFrame:
    sweep_dir = Path(sweep_dir)
    out_dir = Path(out_dir or sweep_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    per_run_dir = out_dir / "per_run"
    per_run_dir.mkdir(exist_ok=True)

    runs = discover_runs(sweep_dir, model)
    if not runs:
        raise FileNotFoundError(
            f"No {model}_history.csv files under {sweep_dir}. "
            "Run hyperparameter_sweep.py first."
        )

    summary = build_summary_table(runs)
    summary["overfit_label"] = summary["acc_gap_at_best"].map(_overfit_status)
    summary.to_csv(out_dir / "sweep_summary.csv", index=False)

    # Per-run detailed panels
    for run in runs:
        fname = per_run_dir / f"{run['run_label']}_curves.png"
        plot_per_run_curves(run, fname, model)

    # Comparative / research plots
    plot_comparative_best_metrics(summary, out_dir)
    plot_heatmap_lr_wd(
        summary, out_dir, "best_val_acc",
        "Hyperparameter heatmap - best validation accuracy",
        "heatmap_best_val_acc.png",
    )
    plot_heatmap_lr_wd(
        summary, out_dir, "acc_gap_at_best",
        "Hyperparameter heatmap - acc gap @ best epoch (overfitting)",
        "heatmap_acc_gap_at_best.png",
    )
    plot_overlay_val_curves(runs, out_dir, model, top_k=top_k_overlay)
    plot_overlay_gaps(runs, out_dir, model, top_k=top_k_overlay)
    plot_train_vs_val_scatter(summary, out_dir)
    plot_lr_schedule_samples(runs, out_dir)
    plot_final_vs_best_gap(summary, out_dir)
    plot_metric_grid_all_runs(runs, out_dir, model)

    # Text report
    best = summary.iloc[0]
    lines = [
        "# HPO Sweep Analysis Report",
        f"Model: {model}",
        f"Runs found: {len(runs)}",
        f"Best config: {best['config_label']}",
        f"Best val accuracy: {best['best_val_acc']:.4f} ({best['best_val_acc']*100:.2f}%)",
        f"Best epoch: {best['best_epoch']}",
        f"Train acc @ best: {best['train_acc_at_best']:.4f}",
        f"Acc gap @ best: {best['acc_gap_at_best']:.4f} ({best['acc_gap_at_best']*100:.2f}%) - {best['overfit_label']}",
        "",
        "## Interpretation",
        "- acc_gap = train_acc - val_acc. Large gap => overfitting.",
        "- loss_gap = val_loss - train_loss. Rising val loss with falling train loss => classic overfit.",
        "- Points far below diagonal in train_vs_val_scatter => high train, lower val.",
        "",
        "## Outputs",
        "- sweep_summary.csv",
        "- per_run/*_curves.png",
        "- comparative_*.png, heatmap_*.png, grid_all_runs_train_val.png",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Processed {len(runs)} sweep runs.")
    print(f"Best: {best['config_label']} | val_acc={best['best_val_acc']:.4f} | gap={best['acc_gap_at_best']:.4f}")
    print(f"Plots saved to: {out_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot HPO sweep results")
    parser.add_argument("--sweep-dir", type=str, default="./artifacts/sweeps")
    parser.add_argument("--model", type=str, default="transformer", choices=["transformer", "cnn_bilstm"])
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=12, help="Top runs in overlay plots")
    args = parser.parse_args()

    generate_report(
        sweep_dir=Path(args.sweep_dir),
        model=args.model,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        top_k_overlay=args.top_k,
    )


if __name__ == "__main__":
    main()
