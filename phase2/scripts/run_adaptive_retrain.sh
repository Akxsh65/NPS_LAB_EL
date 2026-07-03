#!/usr/bin/env bash
# Retrain-from-scratch adaptive attacker grid: 8 policies × 2 architectures = 16 runs.
# Run from phase2/ on GPU server.
set -euo pipefail
cd "$(dirname "$0")/.."

NUM_WORKERS="${NUM_WORKERS:-4}"
export NUM_WORKERS

echo "=============================================="
echo "Adaptive adversary — RETRAIN FROM SCRATCH"
echo "  policies : 8 (clean + 7 obfuscated)"
echo "  models   : transformer_masked (d=160), cnn_bilstm"
echo "  jobs     : 16"
echo "=============================================="

# Step 0 — obfuscated train/val (skip if already generated)
if [[ "${SKIP_OBFUSCATE:-0}" != "1" ]]; then
  bash scripts/run_adaptive_obfuscate_splits.sh
fi

# Step 1 — verify frozen baselines exist (needed later for PRR; finetune uses them too)
for ckpt in \
  artifacts/transformer_production.pt \
  artifacts/cnn_bilstm_best.pt; do
  if [[ ! -f "$ckpt" ]]; then
    echo "ERROR: missing frozen checkpoint $ckpt"
    echo "Copy production checkpoints before adaptive training."
    exit 1
  fi
done

EXTRA=()
if [[ "${SKIP_EXISTING:-1}" == "1" ]]; then
  EXTRA+=(--skip-existing)
fi

python run_adaptive_grid.py retrain "${EXTRA[@]}" "$@"

echo ""
echo "Done. Checkpoints under artifacts/adaptive/retrain/{policy}/{model}/"
echo "Registry: artifacts/adaptive/adaptive_registry.json"
