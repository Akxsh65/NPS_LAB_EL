"""Audit curated demo flows vs population accuracy."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "presentation"))

from api_server import POPULATION_METRICS, get_service  # noqa: E402

DEMO_JS = ROOT / "presentation" / "js" / "demo_flows.js"


def parse_flows() -> list[tuple[str, int, int]]:
    text = DEMO_JS.read_text(encoding="utf-8")
    data = json.loads(text.split("window.DEMO_FLOWS = ", 1)[1].rsplit(";", 1)[0])
    out = []
    for fid, f in data.items():
        out.append((fid, int(f["flowIndex"]), int(f["classId"])))
    return out


def main() -> None:
    flows = parse_flows()
    svc = get_service()
    print(f"Curated demo flows: {len(flows)}\n")

    for defense in ("baseline", "jitter_low"):
        exp = POPULATION_METRICS[defense]["accuracy"]
        print(f"=== {defense} (Phase 4 reported {exp}%) ===")
        for model in ("transformer", "bilstm"):
            ok = 0
            for fid, fi, cid in flows:
                r = svc.predict(fi, defense, model, top_k=1)
                mark = "OK  " if r["correct"] else "MISS"
                print(
                    f"  {mark} {model:11} {fid:18} #{fi:5} true={cid:3} pred={r['predicted_app_id']:3}"
                )
                if r["correct"]:
                    ok += 1
            pct = 100 * ok / len(flows)
            print(f"  --> {model}: {ok}/{len(flows)} = {pct:.1f}% (NOT population {exp}%)\n")

    # Simulate 20x next-example on jitter_low (default UI defense)
    print("=== Simulate 20 clicks 'Next example' (jitter_low + transformer) ===")
    keys = list(json.loads(DEMO_JS.read_text().split("=", 1)[1].rsplit(";", 1)[0]).keys())
    correct = 0
    for i in range(20):
        key = keys[i % len(keys)]
        fi = json.loads(DEMO_JS.read_text().split("=", 1)[1].rsplit(";", 1)[0])[key]["flowIndex"]
        r = svc.predict(int(fi), "jitter_low", "transformer", top_k=1)
        if r["correct"]:
            correct += 1
    print(f"  20 cycles through 12 flows: {correct}/20 = {correct/20*100:.0f}% correct")


if __name__ == "__main__":
    main()
