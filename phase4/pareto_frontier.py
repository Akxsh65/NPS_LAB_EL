"""
Pareto frontier for privacy–cost tradeoffs (Phase 4).

Privacy objective: lower attacker test accuracy is better.
Cost: mean bandwidth overhead or mean latency (separate 1D fronts).

Point A dominates B iff cost_A <= cost_B and acc_A <= acc_B with at least one strict
(lower attacker accuracy = better privacy).

Note: multiple jitter tiers share 0% bandwidth overhead, so only the strongest
privacy point among them lies on the bandwidth Pareto frontier; use the latency
frontier plot to compare jitter operating points.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

CostColumn = Literal["mean_bandwidth_overhead", "mean_latency_overhead_ms"]


def _defended_df(df: pd.DataFrame, include_mtu: bool = True) -> pd.DataFrame:
    """Exclude baseline; optionally exclude MTU padding experiments."""
    out = df[df["experiment"] != "baseline"].copy()
    if not include_mtu:
        out = out[~out["experiment"].str.contains("mtu", case=False, na=False)]
    return out.reset_index(drop=True)


def is_dominated(
    costs: np.ndarray,
    accuracies: np.ndarray,
    index: int,
) -> bool:
    """
    True if point `index` is dominated by another point (lower cost AND lower accuracy).
    """
    c_i, a_i = costs[index], accuracies[index]
    for j in range(len(costs)):
        if j == index:
            continue
        c_j, a_j = costs[j], accuracies[j]
        if c_j <= c_i and a_j <= a_i and (c_j < c_i or a_j < a_i):
            return True
    return False


def pareto_nondominated_indices(costs: np.ndarray, accuracies: np.ndarray) -> np.ndarray:
    """Indices of non-dominated points."""
    n = len(costs)
    return np.array(
        [i for i in range(n) if not is_dominated(costs, accuracies, i)],
        dtype=int,
    )


def build_frontier_table(
    df: pd.DataFrame,
    cost_col: CostColumn,
    include_mtu: bool = True,
) -> pd.DataFrame:
    """
    One row per defended experiment with dominance flags and frontier membership.
    """
    sub = _defended_df(df, include_mtu=include_mtu)
    if sub.empty:
        return pd.DataFrame()

    costs = sub[cost_col].astype(float).values
    accs = sub["accuracy"].astype(float).values

    dominated = [is_dominated(costs, accs, i) for i in range(len(sub))]
    nd_idx = set(pareto_nondominated_indices(costs, accs).tolist())

    rows = []
    for i, (_, row) in enumerate(sub.iterrows()):
        rows.append(
            {
                "experiment": row["experiment"],
                "cost_metric": cost_col,
                "cost": float(costs[i]),
                "cost_pct": float(costs[i] * 100) if cost_col == "mean_bandwidth_overhead" else float(costs[i]),
                "accuracy": float(accs[i]),
                "accuracy_pct": float(accs[i] * 100),
                "macro_f1": float(row["macro_f1"]),
                "macro_f1_pct": float(row["macro_f1"] * 100),
                "accuracy_drop": float(row.get("accuracy_drop", 0)),
                "macro_f1_drop": float(row.get("macro_f1_drop", 0)),
                "is_dominated": bool(dominated[i]),
                "on_pareto_frontier": i in nd_idx,
                "include_mtu": include_mtu,
            }
        )
    return pd.DataFrame(rows)


def frontier_curve_points(
    df: pd.DataFrame,
    cost_col: CostColumn,
    include_mtu: bool = True,
) -> pd.DataFrame:
    """
    Non-dominated points sorted by cost (for line plot).
    """
    table = build_frontier_table(df, cost_col, include_mtu=include_mtu)
    front = table[table["on_pareto_frontier"]].sort_values("cost")
    return front.reset_index(drop=True)
