#!/usr/bin/env bash
# Stage 2 HPO only (7 runs) + plots. Run from phase2/.
set -euo pipefail
cd "$(dirname "$0")/.."

export RUN_STAGE2_ONLY=1
# Runs 01-07 and stage2 plots (extracted from full pipeline)
TRAIN_PT="${TRAIN_PT:-../phase1/artifacts/train_tensors.pt}"
VAL_PT="${VAL_PT:-../phase1/artifacts/val_tensors.pt}"
BS=1024
WD=1e-2
NUM_WORKERS="${NUM_WORKERS:-4}"

mkdir -p artifacts/refine/stage2
mkdir -p artifacts/refine/plots/stage2

_run() {
  echo ""
  echo ">>> $*"
  "$@"
}

_run python train.py --model transformer --batch-size "$BS" --num-workers "$NUM_WORKERS" \
  --train-pt "$TRAIN_PT" --val-pt "$VAL_PT" --lr 7e-4 --weight-decay "$WD" --label-smoothing 0.05 \
  --epochs 70 --patience 12 --t-max 70 --out-dir ./artifacts/refine/stage2/run_01_bs1024_lr7e-4_wd0.01

_run python train.py --model transformer --batch-size "$BS" --num-workers "$NUM_WORKERS" \
  --train-pt "$TRAIN_PT" --val-pt "$VAL_PT" --lr 1e-3 --weight-decay "$WD" --label-smoothing 0.05 \
  --epochs 80 --patience 15 --t-max 80 --out-dir ./artifacts/refine/stage2/run_02_bs1024_lr0.001_wd0.01

_run python train.py --model transformer --batch-size "$BS" --num-workers "$NUM_WORKERS" \
  --train-pt "$TRAIN_PT" --val-pt "$VAL_PT" --lr 1.2e-3 --weight-decay "$WD" --label-smoothing 0.05 \
  --epochs 70 --patience 12 --t-max 70 --out-dir ./artifacts/refine/stage2/run_03_bs1024_lr0.0012_wd0.01

_run python train.py --model transformer --batch-size "$BS" --num-workers "$NUM_WORKERS" \
  --train-pt "$TRAIN_PT" --val-pt "$VAL_PT" --lr 1.5e-3 --weight-decay "$WD" --label-smoothing 0.05 \
  --epochs 70 --patience 12 --t-max 70 --out-dir ./artifacts/refine/stage2/run_04_bs1024_lr0.0015_wd0.01

_run python train.py --model transformer --batch-size "$BS" --num-workers "$NUM_WORKERS" \
  --train-pt "$TRAIN_PT" --val-pt "$VAL_PT" --lr 1e-3 --weight-decay 5e-3 --label-smoothing 0.05 \
  --epochs 70 --patience 12 --t-max 70 --out-dir ./artifacts/refine/stage2/run_05_bs1024_lr0.001_wd0.005

_run python train.py --model transformer --batch-size "$BS" --num-workers "$NUM_WORKERS" \
  --train-pt "$TRAIN_PT" --val-pt "$VAL_PT" --lr 1e-3 --weight-decay "$WD" --label-smoothing 0.0 \
  --epochs 70 --patience 12 --t-max 70 --out-dir ./artifacts/refine/stage2/run_06_bs1024_lr0.001_wd0.01

_run python train.py --model transformer --batch-size "$BS" --num-workers "$NUM_WORKERS" \
  --train-pt "$TRAIN_PT" --val-pt "$VAL_PT" --lr 1e-3 --weight-decay "$WD" --label-smoothing 0.03 \
  --epochs 70 --patience 12 --t-max 70 --out-dir ./artifacts/refine/stage2/run_07_bs1024_lr0.001_wd0.01

python plot_sweep_results.py --sweep-dir ./artifacts/refine/stage2 --model transformer \
  --out-dir ./artifacts/refine/plots/stage2

echo "Stage 2 complete. See artifacts/refine/plots/stage2/sweep_summary.csv"
