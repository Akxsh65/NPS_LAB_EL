#!/usr/bin/env bash
# Generate obfuscated train/val splits (Phase 3). Run from repo root or phase2/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT/phase3"

echo ">>> Generating adaptive train/val obfuscated tensors"
python generate_obfuscated_splits.py --skip-existing "$@"

echo ""
echo "Outputs:"
echo "  phase3/artifacts/adaptive/train/{policy}_train.pt"
echo "  phase3/artifacts/adaptive/val/{policy}_val.pt"
echo "  phase3/artifacts/adaptive/adaptive_splits_manifest.json"
