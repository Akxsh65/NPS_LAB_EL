"""Tests for Phase 4 Tier C statistics and helpers."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PHASE4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE4))

from evaluate import apply_channel_mask  # noqa: E402
from statistical_analysis import (  # noqa: E402
    bootstrap_metrics,
    mcnemar_test,
    paired_bootstrap_accuracy_diff,
    top_confused_pairs,
)
import torch  # noqa: E402


class TestChannelMask(unittest.TestCase):
    def test_ipt_only_zeros_other_channels(self):
        X = torch.ones(2, 3, 4)
        out = apply_channel_mask(X, [0])
        self.assertTrue(torch.all(out[:, 0, :] == 1))
        self.assertTrue(torch.all(out[:, 1:, :] == 0))


class TestBootstrap(unittest.TestCase):
    def test_ci_contains_point_estimate(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 4, size=500)
        y_pred = y_true.copy()
        y_pred[:50] = (y_pred[:50] + 1) % 4
        stats = bootstrap_metrics(y_true, y_pred, n_bootstrap=200, seed=0)
        self.assertLessEqual(stats["accuracy_ci_low"], stats["accuracy"])
        self.assertGreaterEqual(stats["accuracy_ci_high"], stats["accuracy"])
        self.assertLess(stats["accuracy_ci_high"] - stats["accuracy_ci_low"], 0.2)


class TestPaired(unittest.TestCase):
    def test_mcnemar_detects_difference(self):
        y_true = np.array([0, 1, 2, 3, 0, 1])
        pred_a = y_true.copy()
        pred_b = y_true.copy()
        pred_b[0] = 99
        out = mcnemar_test(y_true, pred_a, pred_b)
        self.assertGreater(out["mcnemar_n_discordant"], 0)

    def test_paired_bootstrap_positive_when_a_better(self):
        y_true = np.ones(100, dtype=int)
        pred_a = np.ones(100, dtype=int)
        pred_b = np.zeros(100, dtype=int)
        out = paired_bootstrap_accuracy_diff(y_true, pred_a, pred_b, n_bootstrap=100, seed=0)
        self.assertGreater(out["accuracy_diff_mean"], 0.5)


class TestConfusedPairs(unittest.TestCase):
    def test_off_diagonal_pairs(self):
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        pairs = top_confused_pairs(y_true, y_pred, top_k=5, min_count=1)
        self.assertTrue(any(p["true_id"] == 0 and p["pred_id"] == 1 for p in pairs))


if __name__ == "__main__":
    unittest.main()
