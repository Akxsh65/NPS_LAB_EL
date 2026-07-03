"""
Export real CESNET-QUIC22 test flows for the presentation simulator.

Reads phase1/artifacts/test_tensors.pt, denormalizes PPI channels, and writes:
  - presentation/data/demo_flows.json
  - presentation/js/demo_flows.js  (window.DEMO_FLOWS — works without fetch)

Usage:
  python presentation/scripts/export_demo_flows.py
  python presentation/scripts/export_demo_flows.py --count 12 --annotate
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
PHASE1_ARTIFACTS = ROOT / "phase1" / "artifacts"
PHASE2_ARTIFACTS = ROOT / "phase2" / "artifacts"
OUT_JSON = ROOT / "presentation" / "data" / "demo_flows.json"
OUT_JS = ROOT / "presentation" / "js" / "demo_flows.js"

LOG_EPS = 1e-8
IPT_CLIP_MAX = 1_000_000.0
MTU = 1500.0
PAD_DIR_EPS = 0.5
PAD_SIZE_EPS = 1e-6
TEST_WEEK = "W-2022-45"

TAG_META = {
    "bulk": {"emoji": "📥", "shape": "steady bulk sizes"},
    "mixed": {"emoji": "🌐", "shape": "mixed request/response sizes"},
    "small": {"emoji": "📞", "shape": "mostly small packets"},
    "regular_ipt": {"emoji": "⏱", "shape": "regular timing"},
    "large_var": {"emoji": "🎬", "shape": "large packets with variation"},
    "medium": {"emoji": "📱", "shape": "medium-sized bursts"},
    "bursty": {"emoji": "⚡", "shape": "bursty packet train"},
    "sparse": {"emoji": "〰", "shape": "sparse active packets"},
    "high_ipt": {"emoji": "🐢", "shape": "high IPT variance"},
    "correct_ex": {"emoji": "✓", "shape": "Transformer classifies correctly (baseline)"},
    "misclass_ex": {"emoji": "✗", "shape": "Transformer misclassifies (baseline)"},
}


def inverse_ipt(norm_ipt: np.ndarray, mean: float, std: float) -> np.ndarray:
    log_ipt = norm_ipt * (std + LOG_EPS) + mean
    raw = np.expm1(log_ipt)
    return np.clip(raw, 0.0, IPT_CLIP_MAX)


def inverse_size(norm_size: np.ndarray) -> np.ndarray:
    return np.clip(norm_size, 0.0, 1.0) * MTU


def active_mask(dir_seq: np.ndarray, size_seq: np.ndarray) -> np.ndarray:
    is_padding = (np.abs(dir_seq) < PAD_DIR_EPS) & (size_seq < PAD_SIZE_EPS)
    return ~is_padding


def flow_stats(ipt: np.ndarray, dir_seq: np.ndarray, size: np.ndarray) -> dict | None:
    mask = active_mask(dir_seq, size)
    n = int(mask.sum())
    if n < 5:
        return None
    sizes = size[mask]
    ipts = ipt[mask]
    return {
        "n": n,
        "mean_size": float(sizes.mean()),
        "std_size": float(sizes.std()),
        "max_size": float(sizes.max()),
        "ipt_std": float(ipts[1:].std()) if n > 2 else 0.0,
        "ipt_mean": float(ipts[1:].mean()) if n > 2 else 0.0,
    }


def denormalize_flow(flow: np.ndarray, ipt_mean: float, ipt_std: float):
    ipt = inverse_ipt(flow[0].astype(np.float64), ipt_mean, ipt_std)
    dir_seq = flow[1].astype(np.float32)
    size = inverse_size(flow[2].astype(np.float64))
    return ipt, dir_seq, size


def pick_diverse_flows(
    X: np.ndarray,
    y: np.ndarray,
    ipt_mean: float,
    ipt_std: float,
    count: int,
    baseline_correct: dict[int, bool] | None = None,
) -> list[tuple[int, int, str]]:
    by_class: dict[int, list[int]] = defaultdict(list)
    for i in range(len(y)):
        by_class[int(y[i])].append(i)

    heuristics: list[tuple[str, callable]] = [
        ("bulk", lambda s: s["std_size"] < 50 and s["mean_size"] > 1200),
        ("mixed", lambda s: s["std_size"] > 400),
        ("small", lambda s: s["mean_size"] < 200),
        ("regular_ipt", lambda s: s["ipt_std"] < 5 and s["n"] >= 10),
        ("large_var", lambda s: s["max_size"] > 1400 and s["std_size"] > 200),
        ("medium", lambda s: 300 < s["mean_size"] < 800),
        ("bursty", lambda s: s["n"] >= 12 and s["std_size"] > 250),
        ("sparse", lambda s: 5 <= s["n"] <= 8),
        ("high_ipt", lambda s: s["ipt_std"] > 30 and s["n"] >= 8),
    ]

    picked: list[tuple[int, int, str]] = []
    used_classes: set[int] = set()
    used_indices: set[int] = set()

    def try_add(flow_index: int, class_index: int, tag: str) -> bool:
        if flow_index in used_indices or class_index in used_classes:
            return False
        used_indices.add(flow_index)
        used_classes.add(class_index)
        picked.append((flow_index, class_index, tag))
        return True

    if baseline_correct:
        for want_correct, tag in ((True, "correct_ex"), (False, "misclass_ex")):
            for i, ok in baseline_correct.items():
                if ok != want_correct or i in used_indices:
                    continue
                cls = int(y[i])
                if try_add(i, cls, tag):
                    break

    for tag, fn in heuristics:
        if len(picked) >= count:
            break
        best: tuple[int, int, str] | None = None
        for cls, idxs in by_class.items():
            if cls in used_classes:
                continue
            for i in idxs[:400]:
                if i in used_indices:
                    continue
                ipt, dir_seq, size = denormalize_flow(X[i], ipt_mean, ipt_std)
                stats = flow_stats(ipt, dir_seq, size)
                if stats and fn(stats):
                    best = (i, cls, tag)
                    break
            if best:
                break
        if best:
            try_add(best[0], best[1], best[2])

    if len(picked) < count:
        for i in range(len(y)):
            if len(picked) >= count:
                break
            cls = int(y[i])
            if i in used_indices or cls in used_classes:
                continue
            try_add(i, cls, "medium")

    return picked[:count]


def export_flow(
    X: np.ndarray,
    flow_index: int,
    class_index: int,
    tag: str,
    class_id: int,
    ipt_mean: float,
    ipt_std: float,
    baseline_correct: bool | None = None,
) -> dict:
    flow = X[flow_index]
    ipt, dir_seq, size = denormalize_flow(flow, ipt_mean, ipt_std)
    meta = TAG_META.get(tag, TAG_META["medium"])
    active = int(active_mask(dir_seq, size).sum())
    hint = ""
    if baseline_correct is True:
        hint = " ✓ baseline correct"
    elif baseline_correct is False:
        hint = " ✗ baseline misclass"

    return {
        "id": f"{tag}_{class_id}",
        "flowIndex": flow_index,
        "classIndex": class_index,
        "classId": int(class_id),
        "name": f"App {class_id} · {meta['shape']}{hint}",
        "emoji": meta["emoji"],
        "tag": tag,
        "activePackets": active,
        "testWeek": TEST_WEEK,
        "baselineCorrect": baseline_correct,
        "desc": (
            f"Real CESNET-QUIC22 test flow #{flow_index} ({TEST_WEEK}): "
            f"{active} active packets, application ID {class_id} ({meta['shape']})."
        ),
        "sizes": [int(round(v)) for v in size.tolist()],
        "ipts": [round(float(v), 2) for v in ipt.tolist()],
        "dirs": [int(v) for v in dir_seq.tolist()],
    }


def compute_baseline_correct(
    X: np.ndarray, y: np.ndarray, le, max_scan: int = 8000
) -> dict[int, bool]:
    ckpt = PHASE2_ARTIFACTS / "transformer_production.pt"
    cfg = PHASE2_ARTIFACTS / "transformer_masked_config.json"
    if not ckpt.is_file():
        print("Skipping annotate — no transformer checkpoint", file=sys.stderr)
        return {}

    phase4 = ROOT / "phase4"
    for p in (PHASE2_ARTIFACTS, ROOT / "phase1", phase4):
        s = str(p)
        if s not in sys.path:
            sys.path.append(s)
    sys.modules.pop("config", None)
    sys.path.insert(0, str(phase4))

    from evaluate import load_model  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = load_model(str(ckpt), device, config_path=str(cfg), attack_model="transformer")
    model.eval()

    out: dict[int, bool] = {}
    n = min(len(y), max_scan)
    with torch.no_grad():
        batch = X[:n]
        logits = model(torch.from_numpy(batch).float().to(device))
        preds = logits.argmax(dim=1).cpu().numpy()
        for i in range(n):
            out[i] = int(preds[i]) == int(y[i])
    print(f"Annotated baseline correctness on {n} flows", file=sys.stderr)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export demo flows for presentation")
    parser.add_argument("--count", type=int, default=12, help="Number of curated flows")
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Label baseline correct/misclass examples (loads Transformer)",
    )
    args = parser.parse_args()

    scaler_path = PHASE1_ARTIFACTS / "ipt_scaler.pkl"
    tensors_path = PHASE1_ARTIFACTS / "test_tensors.pt"
    le_path = PHASE1_ARTIFACTS / "label_encoder.pkl"

    for path in (scaler_path, tensors_path, le_path):
        if not path.is_file():
            print(f"Missing artifact: {path}", file=sys.stderr)
            sys.exit(1)

    scaler = joblib.load(scaler_path)
    ipt_mean, ipt_std = float(scaler["mean"]), float(scaler["std"])
    le = joblib.load(le_path)
    data = torch.load(tensors_path, map_location="cpu", weights_only=False)
    X = data["X"].numpy()
    y = data["y"].numpy()

    baseline_map: dict[int, bool] | None = None
    if args.annotate:
        baseline_map = compute_baseline_correct(X, y, le)

    selections = pick_diverse_flows(
        X, y, ipt_mean, ipt_std, args.count, baseline_correct=baseline_map
    )
    if len(selections) < 4:
        print("Could not find enough diverse flows.", file=sys.stderr)
        sys.exit(1)

    flows: dict[str, dict] = {}
    for flow_index, class_index, tag in selections:
        class_id = int(le.classes_[class_index])
        bc = baseline_map.get(flow_index) if baseline_map else None
        entry = export_flow(
            X, flow_index, class_index, tag, class_id, ipt_mean, ipt_std, bc
        )
        flows[entry["id"]] = entry

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": str(tensors_path.relative_to(ROOT)),
        "testWeek": TEST_WEEK,
        "testFlows": int(len(y)),
        "flows": flows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    js_body = (
        "/* Auto-generated by presentation/scripts/export_demo_flows.py — do not edit. */\n"
        "window.DEMO_FLOWS = "
        + json.dumps(flows, indent=2)
        + ";\n"
    )
    OUT_JS.write_text(js_body, encoding="utf-8")

    print(f"Exported {len(flows)} flows:")
    for key, f in flows.items():
        print(f"  {key}: flow #{f['flowIndex']}, app ID {f['classId']}, {f['activePackets']} pkts")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_JS}")


if __name__ == "__main__":
    main()
