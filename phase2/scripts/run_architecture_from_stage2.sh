#!/usr/bin/env bash
# =============================================================================
# Stage 3 — Architecture ablations using Stage 2 winner hyperparameters.
#
# Prerequisite: Stage 2 complete with sweep_summary.csv under:
#   artifacts/refine/stage2/sweep_summary.csv
#   (or phase2/stage2/sweep_summary.csv — set SWEEP_CSV below)
#
# Run from phase2/:
#   chmod +x scripts/run_architecture_from_stage2.sh
#   nohup bash scripts/run_architecture_from_stage2.sh > arch_ablation.log 2>&1 &
#   tail -f arch_ablation.log
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

# --- Load winner HP (edit scripts/stage2_winner.env or regenerate) ---
if [[ -f scripts/stage2_winner.env ]]; then
  # shellcheck source=/dev/null
  source scripts/stage2_winner.env
else
  echo "ERROR: scripts/stage2_winner.env missing. Run: python scripts/pick_stage2_winner.py"
  exit 1
fi

# Optional: regenerate env from latest CSV
SWEEP_CSV="${SWEEP_CSV:-artifacts/refine/stage2/sweep_summary.csv}"
if [[ -f "$SWEEP_CSV" ]]; then
  python scripts/pick_stage2_winner.py --csv "$SWEEP_CSV" --out scripts/stage2_winner.env
  # shellcheck source=/dev/null
  source scripts/stage2_winner.env
fi

TRAIN_PT="${TRAIN_PT:-../phase1/artifacts/train_tensors.pt}"
VAL_PT="${VAL_PT:-../phase1/artifacts/val_tensors.pt}"
NUM_WORKERS="${NUM_WORKERS:-4}"

echo "=============================================="
echo "STAGE 2 WINNER (baseline to beat)"
echo "  run     : $STAGE2_WINNER_RUN"
echo "  dir     : $STAGE2_WINNER_DIR"
echo "  val acc : see sweep_summary (target ~82.5%)"
echo "=============================================="
echo "ARCHITECTURE TRAINING USES:"
echo "  bs=$WIN_BS  lr=$WIN_LR  wd=$WIN_WD  ls=$WIN_LS"
echo "  epochs=$WIN_EPOCHS  patience=$WIN_PATIENCE  warmup=$WIN_WARMUP"
echo "=============================================="

mkdir -p artifacts/refine/architecture
mkdir -p artifacts/refine/plots/architecture

# --- 0) Freeze Stage 2 baseline checkpoint (no retrain) ---
if [[ -f "${STAGE2_WINNER_DIR}/${STAGE2_WINNER_CKPT}" ]]; then
  cp "${STAGE2_WINNER_DIR}/${STAGE2_WINNER_CKPT}" artifacts/stage2_winner_baseline.pt
  echo "Copied Stage 2 winner -> artifacts/stage2_winner_baseline.pt"
else
  echo "WARN: ${STAGE2_WINNER_DIR}/${STAGE2_WINNER_CKPT} not found; skip copy"
fi

# --- 1) Masked Transformer d_model=128 (primary Phase 3/4 candidate) ---
echo ""
echo ">>> [1/2] transformer_masked  d_model=128"
python train.py --model transformer_masked --batch-size "$WIN_BS" --num-workers "$NUM_WORKERS" \
  --train-pt "$TRAIN_PT" --val-pt "$VAL_PT" \
  --lr "$WIN_LR" --weight-decay "$WIN_WD" --label-smoothing "$WIN_LS" \
  --epochs "$WIN_EPOCHS" --patience "$WIN_PATIENCE" --t-max "$WIN_TMAX" \
  --warmup-epochs "$WIN_WARMUP" \
  --d-model 128 --nhead 8 --num-layers 4 --ff-dim 256 --dropout 0.2 \
  --out-dir ./artifacts/refine/architecture/run_masked_d128

# --- 2) Masked Transformer d_model=160 (capacity ablation) ---
echo ""
echo ">>> [2/2] transformer_masked  d_model=160"
python train.py --model transformer_masked --batch-size "$WIN_BS" --num-workers "$NUM_WORKERS" \
  --train-pt "$TRAIN_PT" --val-pt "$VAL_PT" \
  --lr "$WIN_LR" --weight-decay "$WIN_WD" --label-smoothing "$WIN_LS" \
  --epochs "$WIN_EPOCHS" --patience "$WIN_PATIENCE" --t-max "$WIN_TMAX" \
  --warmup-epochs "$WIN_WARMUP" \
  --d-model 160 --nhead 8 --num-layers 4 --ff-dim 512 --dropout 0.2 \
  --out-dir ./artifacts/refine/architecture/run_masked_d160

# --- Plots for architecture runs (history CSVs must exist) ---
echo ""
echo ">>> Architecture training curves"
python plot_sweep_results.py \
  --sweep-dir ./artifacts/refine/architecture \
  --model transformer_masked \
  --out-dir ./artifacts/refine/plots/architecture

echo ""
echo "=============================================="
echo "DONE. Next steps:"
echo "1. Open artifacts/refine/plots/architecture/sweep_summary.csv"
echo "2. Pick run with highest best_val_acc AND acc_gap_at_best < 0.02"
echo "3. If masked_d128 >= stage2 winner (~0.825):"
echo "     cp artifacts/refine/architecture/run_masked_d128/transformer_masked_best_acc.pt \\"
echo "        artifacts/transformer_production.pt"
echo "   Else keep Stage 2:"
echo "     cp artifacts/stage2_winner_baseline.pt artifacts/transformer_production.pt"
echo "4. Phase 3: cd ../phase3 && python generate_obfuscated.py"
echo "5. Phase 4: cd ../phase4 && python run_experiments.py \\"
echo "     --checkpoint ../phase2/artifacts/transformer_production.pt"
echo "=============================================="
