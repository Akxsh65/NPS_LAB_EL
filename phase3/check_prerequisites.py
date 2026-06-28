"""
Verify Phase 3 inputs before generate_obfuscated.py.

Usage (from phase3/):
  python check_prerequisites.py
  python check_prerequisites.py --test-pt ../phase1/artifacts/test_tensors.pt
"""
from __future__ import annotations

import argparse
import os
import sys

from settings import IPT_SCALER, PHASE1_ARTIFACTS, TEST_TENSORS


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Phase 3 prerequisites")
    parser.add_argument("--test-pt", default=TEST_TENSORS)
    parser.add_argument("--ipt-scaler", default=IPT_SCALER)
    args = parser.parse_args()

    required = [
        ("Test tensors", args.test_pt),
        ("IPT scaler (Phase 1)", args.ipt_scaler),
    ]
    recommended = [
        ("Label encoder (Phase 4)", os.path.join(PHASE1_ARTIFACTS, "label_encoder.pkl")),
        ("Train tensors", os.path.join(PHASE1_ARTIFACTS, "train_tensors.pt")),
    ]

    missing = []
    print("Phase 3 prerequisite check\n" + "=" * 50)
    for label, path in required:
        ok = os.path.isfile(path)
        status = "OK" if ok else "MISSING"
        print(f"  [{status}] {label}")
        print(f"         {path}")
        if not ok:
            missing.append((label, path))

    print("\nRecommended (Phase 4 / audit):")
    for label, path in recommended:
        ok = os.path.isfile(path)
        print(f"  [{'OK' if ok else 'missing'}] {label}")

    if missing:
        print("\n" + "=" * 50)
        print("FIX: Copy missing files from the machine where Phase 1 finished:")
        print("  scp phase1/artifacts/ipt_scaler.pkl USER@SERVER:.../phase1/artifacts/")
        print("  scp phase1/artifacts/label_encoder.pkl USER@SERVER:.../phase1/artifacts/")
        print("\nOr re-run Phase 1 on this server (creates scaler + tensors):")
        print("  cd ../phase1 && python run_phase1.py")
        print("\nThen:")
        print("  python generate_obfuscated.py --validate")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("All required files present. Run:")
    print("  python generate_obfuscated.py --validate")


if __name__ == "__main__":
    main()
