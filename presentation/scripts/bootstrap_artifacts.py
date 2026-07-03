"""
Rebuild missing Phase 1 pickles needed for the presentation API.

Uses class index → application ID mappings from phase4/worst_classes_summary.csv
and presentation/js/demo_flows.js, then fits IPT scaler stats from train_tensors.pt.

Run from repo root:
  python presentation/scripts/bootstrap_artifacts.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[2]
PHASE1_ARTIFACTS = ROOT / "phase1" / "artifacts"
LE_PATH = PHASE1_ARTIFACTS / "label_encoder.pkl"
SCALER_PATH = PHASE1_ARTIFACTS / "ipt_scaler.pkl"
WORST_CSV = ROOT / "phase4" / "worst_classes_summary.csv"
BASELINE_PER_CLASS = ROOT / "phase4" / "results" / "baseline_per_class.csv"
DEMO_JS = ROOT / "presentation" / "js" / "demo_flows.js"


def collect_class_mapping(num_classes: int) -> dict[int, int]:
    mapping: dict[int, int] = {}

    for csv_path in (BASELINE_PER_CLASS, WORST_CSV):
        if not csv_path.is_file():
            continue
        with csv_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                mapping[int(row["class_id"])] = int(row["class_name"])

    if DEMO_JS.is_file():
        text = DEMO_JS.read_text(encoding="utf-8")
        for m in re.finditer(
            r'"classIndex":\s*(\d+),\s*"classId":\s*(\d+)', text
        ):
            mapping[int(m.group(1))] = int(m.group(2))

    missing = [i for i in range(num_classes) if i not in mapping]
    if missing:
        print(
            f"Warning: {len(missing)} class indices still unmapped: {missing[:10]}...",
            file=sys.stderr,
        )
    return mapping


def fit_ipt_scaler_from_meta() -> tuple[float, float]:
    meta_path = ROOT / "phase3" / "obfuscated_jitter_low.meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return float(meta["ipt_scaler_mean"]), float(meta["ipt_scaler_std"])
    raise FileNotFoundError(
        "ipt_scaler.pkl missing and no phase3/*.meta.json with ipt_scaler_mean/std"
    )


def main() -> None:
    ckpt = torch.load(
        ROOT / "phase2" / "artifacts" / "transformer_production.pt",
        map_location="cpu",
        weights_only=False,
    )
    num_classes = int(ckpt["num_classes"])

    mapping = collect_class_mapping(num_classes)
    if len(mapping) < num_classes:
        sys.exit(
            f"Only {len(mapping)}/{num_classes} classes mapped — copy label_encoder.pkl from Phase 1 machine."
        )

    classes = [mapping[i] for i in range(num_classes)]
    le = LabelEncoder()
    le.classes_ = np.array(classes, dtype=np.int64)

    PHASE1_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(le, LE_PATH)
    print(f"Wrote {LE_PATH} ({num_classes} classes)")

    if not SCALER_PATH.is_file():
        mean, std = fit_ipt_scaler_from_meta()
        joblib.dump({"mean": mean, "std": std}, SCALER_PATH)
        print(f"Wrote {SCALER_PATH} (mean={mean:.6f}, std={std:.6f})")
    else:
        print(f"Keeping existing {SCALER_PATH}")


if __name__ == "__main__":
    main()
