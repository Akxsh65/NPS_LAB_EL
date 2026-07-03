#!/usr/bin/env bash
# Fine-tune frozen production checkpoints on obfuscated train data.
# 8 policies × 2 architectures = 16 runs (8 epochs, lr=1e-4).
set -euo pipefail
cd "$(dirname "$0")/.."

NUM_WORKERS="${NUM_WORKERS:-4}"
export NUM_WORKERS

echo "=============================================="
echo "Adaptive adversary — FINE-TUNE FROM FROZEN CKPT"
echo "  init     : transformer_production.pt, cnn_bilstm_best.pt"
echo "  epochs   : 8 (patience 3), lr=1e-4"
echo "  jobs     : 16"
echo "=============================================="

if [[ "${SKIP_OBFUSCATE:-0}" != "1" ]]; then
  bash scripts/run_adaptive_obfuscate_splits.sh
fi

for ckpt in \
  artifacts/transformer_production.pt \
  artifacts/cnn_bilstm_best.pt; do
  if [[ ! -f "$ckpt" ]]; then
    echo "ERROR: missing frozen checkpoint $ckpt"
    exit 1
  fi
done

EXTRA=()
if [[ "${SKIP_EXISTING:-1}" == "1" ]]; then
  EXTRA+=(--skip-existing)
fi
if [[ "${FREEZE_BACKBONE:-0}" == "1" ]]; then
  EXTRA+=(--freeze-backbone)
fi

python run_adaptive_grid.py finetune "${EXTRA[@]}" "$@"

echo ""
echo "Done. Checkpoints under artifacts/adaptive/finetune/{policy}/{model}/"
