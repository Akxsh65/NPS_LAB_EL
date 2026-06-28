#!/usr/bin/env bash
# =============================================================================
# Phase 2 refinement pipeline (sequential)
# Run from phase2/:  bash scripts/run_refinement_pipeline.sh
# Optional: nohup bash scripts/run_refinement_pipeline.sh > refine_pipeline.log 2>&1 &
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
echo "Working directory: $(pwd)"

TRAIN_PT="${TRAIN_PT:-../phase1/artifacts/train_tensors.pt}"
VAL_PT="${VAL_PT:-../phase1/artifacts/val_tensors.pt}"
BS=1024
WD=1e-2
NUM_WORKERS="${NUM_WORKERS:-4}"

mkdir -p artifacts/refine/stage2
mkdir -p artifacts/refine/architecture
mkdir -p artifacts/refine/plots

# -----------------------------------------------------------------------------
# STAGE 2 — HPO refinement (7 runs, ~hours each on GPU)
# -----------------------------------------------------------------------------
echo ""
echo "========== STAGE 2: HPO refinement (7 runs) =========="

# Priority 1 — learning rate
python train.py --model transformer --batch-size "$BS" --num-workers "$NUM_WORKERS" \
  --train-pt "$TRAIN_PT" --val-pt "$VAL_PT" \
  --lr 7e-4 --weight-decay "$WD" --label-smoothing 0.05 \
  --epochs 70 --patience 12 --t-max 70 \
  --out-dir ./artifacts/refine/stage2/run_01_bs1024_lr7e-4_wd0.01

python train.py --model transformer --batch-size "$BS" --num-workers "$NUM_WORKERS" \
  --train-pt "$TRAIN_PT" --val-pt "$VAL_PT" \
  --lr 1e-3 --weight-decay "$WD" --label-smoothing 0.05 \
  --epochs 80 --patience 15 --t-max 80 \
  --out-dir ./artifacts/refine/stage2/run_02_bs1024_lr0.001_wd0.01

python train.py --model transformer --batch-size "$BS" --num-workers "$NUM_WORKERS" \
  --train-pt "$TRAIN_PT" --val-pt "$VAL_PT" \
  --lr 1.2e-3 --weight-decay "$WD" --label-smoothing 0.05 \
  --epochs 70 --patience 12 --t-max 70 \
  --out-dir ./artifacts/refine/stage2/run_03_bs1024_lr0.0012_wd0.01

python train.py --model transformer --batch-size "$BS" --num-workers "$NUM_WORKERS" \
  --train-pt "$TRAIN_PT" --val-pt "$VAL_PT" \
  --lr 1.5e-3 --weight-decay "$WD" --label-smoothing 0.05 \
  --epochs 70 --patience 12 --t-max 70 \
  --out-dir ./artifacts/refine/stage2/run_04_bs1024_lr0.0015_wd0.01

# Priority 2 — regularization at lr=1e-3
python train.py --model transformer --batch-size "$BS" --num-workers "$NUM_WORKERS" \
  --train-pt "$TRAIN_PT" --val-pt "$VAL_PT" \
  --lr 1e-3 --weight-decay 5e-3 --label-smoothing 0.05 \
  --epochs 70 --patience 12 --t-max 70 \
  --out-dir ./artifacts/refine/stage2/run_05_bs1024_lr0.001_wd0.005

python train.py --model transformer --batch-size "$BS" --num-workers "$NUM_WORKERS" \
  --train-pt "$TRAIN_PT" --val-pt "$VAL_PT" \
  --lr 1e-3 --weight-decay "$WD" --label-smoothing 0.0 \
  --epochs 70 --patience 12 --t-max 70 \
  --out-dir ./artifacts/refine/stage2/run_06_bs1024_lr0.001_wd0.01

python train.py --model transformer --batch-size "$BS" --num-workers "$NUM_WORKERS" \
  --train-pt "$TRAIN_PT" --val-pt "$VAL_PT" \
  --lr 1e-3 --weight-decay "$WD" --label-smoothing 0.03 \
  --epochs 70 --patience 12 --t-max 70 \
  --out-dir ./artifacts/refine/stage2/run_07_bs1024_lr0.001_wd0.01

echo ""
echo "========== STAGE 2 plots =========="
python plot_sweep_results.py \
  --sweep-dir ./artifacts/refine/stage2 \
  --model transformer \
  --out-dir ./artifacts/refine/plots/stage2

# -----------------------------------------------------------------------------
# STAGE 2b — single best-bet run (optional baseline before architecture)
# -----------------------------------------------------------------------------
echo ""
echo "========== STAGE 2b: best-bet config =========="
python train.py --model transformer --batch-size "$BS" --num-workers "$NUM_WORKERS" \
  --train-pt "$TRAIN_PT" --val-pt "$VAL_PT" \
  --lr 1e-3 --weight-decay "$WD" --label-smoothing 0.03 \
  --epochs 80 --patience 15 --t-max 80 --warmup-epochs 3 \
  --out-dir ./artifacts/refine/best_bet

# -----------------------------------------------------------------------------
# STAGE 3 — architecture (padding mask + capacity) at winning HP
# Tune LR/LS here if stage2 summary shows a different winner.
# -----------------------------------------------------------------------------
# Stage 2 winner: R06 — lr=1e-3, wd=1e-2, label_smoothing=0.0 (see stage2/STAGE2_ANALYSIS.md)
WIN_LR=1e-3
WIN_LS=0.0
WIN_EPOCHS=80
WIN_PATIENCE=15
WIN_TMAX=80

echo ""
echo "========== STAGE 3a: transformer + padding mask =========="
python train.py --model transformer_masked --batch-size "$BS" --num-workers "$NUM_WORKERS" \
  --train-pt "$TRAIN_PT" --val-pt "$VAL_PT" \
  --lr "$WIN_LR" --weight-decay "$WD" --label-smoothing "$WIN_LS" \
  --epochs "$WIN_EPOCHS" --patience "$WIN_PATIENCE" --t-max "$WIN_TMAX" --warmup-epochs 3 \
  --out-dir ./artifacts/refine/architecture/run_masked_d128

echo ""
echo "========== STAGE 3b: masked + d_model=160 =========="
python train.py --model transformer_masked --batch-size "$BS" --num-workers "$NUM_WORKERS" \
  --train-pt "$TRAIN_PT" --val-pt "$VAL_PT" \
  --lr "$WIN_LR" --weight-decay "$WD" --label-smoothing "$WIN_LS" \
  --epochs "$WIN_EPOCHS" --patience "$WIN_PATIENCE" --t-max "$WIN_TMAX" --warmup-epochs 3 \
  --d-model 160 --nhead 8 --num-layers 4 --ff-dim 512 --dropout 0.2 \
  --out-dir ./artifacts/refine/architecture/run_masked_d160

echo ""
echo "========== STAGE 3c: optional batch-size check (only if stage2 plateaued) =========="
# Uncomment to run:
# python train.py --model transformer_masked --batch-size 1280 --num-workers "$NUM_WORKERS" \
#   --train-pt "$TRAIN_PT" --val-pt "$VAL_PT" \
#   --lr "$WIN_LR" --weight-decay "$WD" --label-smoothing "$WIN_LS" \
#   --epochs 70 --patience 12 --t-max 70 \
#   --out-dir ./artifacts/refine/architecture/run_masked_bs1280

echo ""
echo "========== Pick winner & copy production checkpoint =========="
echo "1. Open artifacts/refine/plots/stage2/sweep_summary.csv"
echo "2. Pick highest best_val_acc with acc_gap_at_best < 0.02"
echo "3. Copy that run's transformer_best_acc.pt, e.g.:"
echo "   cp artifacts/refine/architecture/run_masked_d128/transformer_masked_best_acc.pt \\"
echo "      artifacts/transformer_production.pt"
echo ""
echo "Pipeline finished."
