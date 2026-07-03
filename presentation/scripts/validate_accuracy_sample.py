"""
Sample live API / InferenceService predictions and compare to Phase 4 reported accuracy.

Uses the same code path as presentation/api_server.py (not HTTP), so it works even when
the HTTP server is the old build — as long as this script imports the current api_server.

Usage:
  python presentation/scripts/validate_accuracy_sample.py
  python presentation/scripts/validate_accuracy_sample.py --n 2000 --defense jitter_low
  python presentation/scripts/validate_accuracy_sample.py --full  # all 49305 flows (slow)
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "presentation"))

from api_server import POPULATION_METRICS, get_service  # noqa: E402


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return 100 * (centre - margin) / denom, 100 * (centre + margin) / denom


def evaluate_sample(
    defense: str,
    model: str,
    indices: np.ndarray,
    label: str,
) -> None:
    svc = get_service()
    expected = POPULATION_METRICS.get(defense, POPULATION_METRICS["baseline"])["accuracy"]

    correct = 0
    t0 = time.perf_counter()
    for i, idx in enumerate(indices):
        r = svc.predict(int(idx), defense, model, top_k=1)
        if r["correct"]:
            correct += 1
        if (i + 1) % 500 == 0:
            print(f"  ... {i + 1}/{len(indices)}", flush=True)

    elapsed = time.perf_counter() - t0
    n = len(indices)
    emp = 100.0 * correct / n
    lo, hi = wilson_ci(correct, n)
    delta = emp - expected

    print(f"\n{label}")
    print(f"  Defense:     {defense}")
    print(f"  Model:       {model}")
    print(f"  Sample size: {n:,}")
    print(f"  Correct:     {correct:,} / {n:,}")
    print(f"  Sample acc:  {emp:.2f}%  (95% CI {lo:.2f}% – {hi:.2f}%)")
    print(f"  Reported:    {expected:.2f}%  (Phase 4 test-set mean)")
    print(f"  Delta sample vs reported: {delta:+.2f} pp")
    print(f"  Elapsed:     {elapsed:.1f}s ({n / max(elapsed, 0.001):.1f} flows/s)")

    if expected < lo - 0.5 or expected > hi + 0.5:
        print("  WARN: Reported accuracy OUTSIDE 95% CI of this sample (can happen by chance on small n)")
    else:
        print("  OK: Reported accuracy within 95% CI of this sample")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate sample accuracy vs Phase 4 CSV")
    parser.add_argument("--n", type=int, default=1000, help="Random sample size (default 1000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--defense", default="baseline", choices=list(POPULATION_METRICS.keys()))
    parser.add_argument("--model", default="transformer", choices=["transformer", "bilstm"])
    parser.add_argument("--full", action="store_true", help="Evaluate all test flows")
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Also run baseline + jitter_low with n each",
    )
    args = parser.parse_args()

    print("Loading models and test tensors ...")
    svc = get_service()
    print(f"Device: {svc.device}  |  test flows: {svc.n_flows:,}")

    rng = np.random.default_rng(args.seed)

    if args.multi:
        for defense in ("baseline", "jitter_low", "jitter_medium"):
            n = args.n
            idx = rng.choice(svc.n_flows, size=n, replace=False)
            evaluate_sample(defense, args.model, idx, f"=== {defense} ===")
        return

    if args.full:
        indices = np.arange(svc.n_flows)
        label = "=== FULL TEST SET ==="
    else:
        indices = rng.choice(svc.n_flows, size=min(args.n, svc.n_flows), replace=False)
        label = f"=== RANDOM SAMPLE (n={len(indices)}) ==="

    evaluate_sample(args.defense, args.model, indices, label)


if __name__ == "__main__":
    main()
