#!/usr/bin/env bash
# Full adaptive-adversary pipeline: obfuscate splits → retrain → finetune → registry.
#
# Usage (A100 server, from phase2/):
#   chmod +x scripts/run_adaptive_*.sh
#   nohup bash scripts/run_adaptive_pipeline.sh > adaptive_pipeline.log 2>&1 &
#   tail -f adaptive_pipeline.log
#
# Estimated GPU time: ~12–16 A100-hours total (16 retrains + 16 fine-tunes).
set -euo pipefail
cd "$(dirname "$0")/.."

LOG="${ADAPTIVE_LOG:-adaptive_pipeline.log}"
exec > >(tee -a "$LOG") 2>&1

echo "=============================================="
echo "Adaptive pipeline started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Log: $LOG"
echo "=============================================="

bash scripts/run_adaptive_obfuscate_splits.sh

echo ""
echo ">>> Phase A: retrain-from-scratch (16 jobs)"
bash scripts/run_adaptive_retrain.sh --skip-existing

echo ""
echo ">>> Phase B: fine-tune from frozen (16 jobs)"
bash scripts/run_adaptive_finetune.sh --skip-existing

echo ""
echo ">>> Registry"
python -c "from adaptive_registry import write_registry; print(write_registry())"

echo ""
echo "=============================================="
echo "Adaptive pipeline complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Next: evaluate adapted checkpoints in phase4 (see adaptive/README.md)"
echo "=============================================="
