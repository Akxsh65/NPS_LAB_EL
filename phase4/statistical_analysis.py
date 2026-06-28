"""
Bootstrap CIs and paired significance tests for Phase 4 (Tier C).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """Flow-level bootstrap 95% CIs for accuracy and macro F1."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        raise ValueError("empty arrays")

    rng = np.random.default_rng(seed)
    accs: List[float] = []
    f1s: List[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        accs.append(float(accuracy_score(yt, yp)))
        f1s.append(float(f1_score(yt, yp, average="macro", zero_division=0)))

    accs_arr = np.array(accs)
    f1s_arr = np.array(f1s)
    lo = 100.0 * alpha / 2.0
    hi = 100.0 * (1.0 - alpha / 2.0)

    return {
        "accuracy": float(accs_arr.mean()),
        "accuracy_ci_low": float(np.percentile(accs_arr, lo)),
        "accuracy_ci_high": float(np.percentile(accs_arr, hi)),
        "macro_f1": float(f1s_arr.mean()),
        "macro_f1_ci_low": float(np.percentile(f1s_arr, lo)),
        "macro_f1_ci_high": float(np.percentile(f1s_arr, hi)),
        "n_bootstrap": int(n_bootstrap),
        "n_samples": int(n),
    }


def paired_bootstrap_accuracy_diff(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap distribution of accuracy(A) - accuracy(B) on identical flows."""
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    diffs: List[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        acc_a = accuracy_score(yt, y_pred_a[idx])
        acc_b = accuracy_score(yt, y_pred_b[idx])
        diffs.append(float(acc_a - acc_b))

    diffs_arr = np.array(diffs)
    lo = 100.0 * alpha / 2.0
    hi = 100.0 * (1.0 - alpha / 2.0)
    return {
        "accuracy_diff_mean": float(diffs_arr.mean()),
        "accuracy_diff_ci_low": float(np.percentile(diffs_arr, lo)),
        "accuracy_diff_ci_high": float(np.percentile(diffs_arr, hi)),
        "fraction_a_better": float(np.mean(diffs_arr > 0)),
        "n_bootstrap": int(n_bootstrap),
    }


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> Dict[str, float]:
    """McNemar exact test for paired classifier errors."""
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)
    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n_discordant = b + c
    if n_discordant == 0:
        p_value = 1.0
    else:
        try:
            from scipy.stats import binomtest

            p_value = float(binomtest(b, n_discordant, 0.5).pvalue)
        except ImportError:
            from math import erfc, sqrt

            stat = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) else 0.0
            p_value = float(erfc(sqrt(stat / 2.0)))
    return {
        "mcnemar_b": b,
        "mcnemar_c": c,
        "mcnemar_n_discordant": n_discordant,
        "mcnemar_p_value": p_value,
    }


def top_confused_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    top_k: int = 15,
    min_count: int = 5,
) -> List[Dict]:
    """Off-diagonal confusion pairs sorted by count (true -> predicted)."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j:
                continue
            cnt = int(cm[i, j])
            if cnt < min_count:
                continue
            ti = class_names[i] if class_names and i < len(class_names) else str(i)
            pj = class_names[j] if class_names and j < len(class_names) else str(j)
            pairs.append(
                {
                    "true_class": ti,
                    "pred_class": pj,
                    "true_id": i,
                    "pred_id": j,
                    "count": cnt,
                }
            )
    pairs.sort(key=lambda x: x["count"], reverse=True)
    return pairs[:top_k]
