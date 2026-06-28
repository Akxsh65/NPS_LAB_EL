"""Tests for Phase 4 Pareto frontier and publication plotting."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PHASE4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE4))

from pareto_frontier import (  # noqa: E402
    build_frontier_table,
    frontier_curve_points,
    is_dominated,
)
from plot_publication import plot_tier_a  # noqa: E402


def _sample_results_df() -> pd.DataFrame:
    baseline_acc, baseline_f1 = 0.7777, 0.7441
    rows = [
        ("baseline", 0.0, 0.0, baseline_acc, baseline_f1),
        ("obfuscated_jitter_low", 0.0, 11.0, 0.7684, 0.7272),
        ("obfuscated_jitter_medium", 0.0, 55.0, 0.6883, 0.6301),
        ("obfuscated_jitter_high", 0.0, 220.0, 0.5468, 0.4766),
        ("obfuscated_linear128", 0.1744, 0.0, 0.5906, 0.6201),
        ("obfuscated_linear128_jitter_medium", 0.1744, 55.0, 0.5230, 0.4999),
        ("obfuscated_mtu", 2.7397, 0.0, 0.0214, 0.0030),
    ]
    data = []
    for exp, bw, lat, acc, f1 in rows:
        data.append(
            {
                "experiment": exp,
                "mean_bandwidth_overhead": bw,
                "mean_latency_overhead_ms": lat,
                "accuracy": acc,
                "macro_f1": f1,
                "accuracy_drop": baseline_acc - acc,
                "macro_f1_drop": baseline_f1 - f1,
            }
        )
    return pd.DataFrame(data)


class TestParetoDominance(unittest.TestCase):
    def test_jitter_high_not_dominated_at_zero_bw(self):
        df = _sample_results_df()
        sub = df[df["experiment"] != "baseline"]
        costs = sub["mean_bandwidth_overhead"].values
        accs = sub["accuracy"].values
        idx = list(sub["experiment"]).index("obfuscated_jitter_high")
        self.assertFalse(is_dominated(costs, accs, idx))

    def test_jitter_low_dominated_by_high_on_bw(self):
        df = _sample_results_df()
        sub = df[df["experiment"] != "baseline"]
        costs = sub["mean_bandwidth_overhead"].values
        accs = sub["accuracy"].values
        idx = list(sub["experiment"]).index("obfuscated_jitter_low")
        self.assertTrue(is_dominated(costs, accs, idx))


class TestPlotPublication(unittest.TestCase):
    def test_plot_tier_a_outputs(self):
        csv_path = PHASE4 / "results" / "accuracy_results.csv"
        if not csv_path.is_file():
            csv_path = PHASE4 / "accuracy_results.csv"
        if not csv_path.is_file():
            self.skipTest("accuracy_results.csv not present")
        with tempfile.TemporaryDirectory() as tmp:
            paths = plot_tier_a(str(csv_path), tmp)
            names = {Path(p).name for p in paths}
            self.assertIn("macro_f1_comparison_bars.png", names)
            self.assertIn("pareto_frontier_table.csv", names)


if __name__ == "__main__":
    unittest.main()
