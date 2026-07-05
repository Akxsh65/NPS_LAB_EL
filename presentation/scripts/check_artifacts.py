"""
Verify all files required to run the presentation + inference API.

Usage (from repo root):
  python presentation/scripts/check_artifacts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

REQUIRED: list[tuple[str, str]] = [
    ("phase1/artifacts/test_tensors.pt", "CESNET test flows (49,305 x 3x30 tensors)"),
    ("phase1/artifacts/label_encoder.pkl", "Maps class index to CESNET app ID"),
    ("phase1/artifacts/ipt_scaler.pkl", "IPT denormalization for obfuscator"),
    ("phase2/artifacts/transformer_production.pt", "Masked Transformer checkpoint"),
    ("phase2/artifacts/transformer_masked_config.json", "Transformer architecture config"),
    ("phase2/artifacts/cnn_bilstm_best.pt", "CNN-BiLSTM checkpoint"),
    ("presentation/js/demo_flows.js", "12 curated demo flows (packet timelines)"),
    ("presentation/js/test_flow_catalog.js", "Full test-set flow index catalog"),
    ("presentation/js/app.js", "Main UI"),
    ("presentation/index.html", "Presentation page"),
]

OPTIONAL: list[tuple[str, str]] = [
    ("phase1/artifacts/train_tensors.pt", "Only needed to re-run Phase 1 / bootstrap from scratch"),
    ("phase1/artifacts/val_tensors.pt", "Only needed to re-run Phase 1 training"),
]


def main() -> int:
    print("Checking presentation artifacts under:", ROOT)
    print()
    missing = 0
    for rel, desc in REQUIRED:
        path = ROOT / rel
        if path.is_file():
            mb = path.stat().st_size / (1024 * 1024)
            print(f"  OK   {rel}  ({mb:.2f} MB) - {desc}")
        else:
            missing += 1
            print(f"  MISS {rel} - {desc}", file=sys.stderr)

    print("\nOptional:")
    for rel, desc in OPTIONAL:
        path = ROOT / rel
        tag = "OK  " if path.is_file() else "skip"
        print(f"  {tag} {rel} - {desc}")

    if missing:
        print(
            f"\n{missing} required file(s) missing. See presentation/README.md "
            "(Artifacts must be committed or copied from the team drive).",
            file=sys.stderr,
        )
        if not (ROOT / "phase1/artifacts/label_encoder.pkl").is_file():
            print(
                "  Tip: python presentation/scripts/bootstrap_artifacts.py "
                "(needs phase2 checkpoint + phase4 CSVs)",
                file=sys.stderr,
            )
        return 1

    print("\nAll required artifacts present. Start the demo:")
    print("  Terminal 1: python presentation/api_server.py")
    print("  Terminal 2: python -m http.server 8080")
    print("  Browser:    http://localhost:8080/presentation/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
