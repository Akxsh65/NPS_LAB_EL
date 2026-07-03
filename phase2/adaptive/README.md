# Adaptive Adversary Training (Phase 2 extension)

Train attackers that **adapt** to obfuscation: retrain-from-scratch and fine-tune-from-frozen on obfuscated **train/val** data, then evaluate on matching obfuscated **test** tensors (Phase 4 extension — not included here).

## Experiment grid

| Setting | Train/val data | Test data (Phase 3) |
|---------|----------------|---------------------|
| `clean` | Phase 1 `train_tensors.pt` / `val_tensors.pt` | `test_tensors.pt` |
| `jitter_low` … `mtu_jitter_medium` | `phase3/artifacts/adaptive/{train,val}/{policy}_*.pt` | `phase3/artifacts/obfuscated_{policy}.pt` |

**16 jobs per mode** = 8 policies × 2 architectures (`transformer_masked` d=160, `cnn_bilstm`).

## Prerequisites (GPU server)

```bash
# Phase 1 tensors + scaler
ls ../phase1/artifacts/train_tensors.pt
ls ../phase1/artifacts/val_tensors.pt
ls ../phase1/artifacts/ipt_scaler.pkl

# Frozen production attackers (fine-tune init + PRR denominator)
ls artifacts/transformer_production.pt
ls artifacts/cnn_bilstm_best.pt

# Phase 3 obfuscated TEST sets (already used in Phase 4)
ls ../phase3/artifacts/obfuscated_jitter_low.pt
```

Copy missing artifacts from your Phase 1/2/3 machine if needed.

## Install

```bash
cd phase2
pip install torch numpy tqdm scikit-learn pandas
# CUDA build, e.g.:
# pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Step 1 — Generate obfuscated train/val splits (~CPU, 30–60 min)

```bash
cd phase3
python generate_obfuscated_splits.py
# Or from phase2:
bash scripts/run_adaptive_obfuscate_splits.sh
```

**Outputs**

```
phase3/artifacts/adaptive/
  train/jitter_low_train.pt
  val/jitter_low_val.pt
  ... (7 policies × 2 splits)
  adaptive_splits_manifest.json
```

`clean` policy reuses Phase 1 tensors (no duplicate files).

## Step 2 — Retrain from scratch (16 runs, ~6–8 A100-hours)

Each run trains from random init on policy-matched obfuscated train data.

| Model | Hyperparameters |
|-------|-----------------|
| `transformer_masked` | Stage 2 winner: bs=1024, lr=1e-3, wd=0.01, 80 ep, d_model=160 |
| `cnn_bilstm` | bs=1024, lr=1e-3, wd=0.01, 60 ep |

```bash
cd phase2

# Dry run (print jobs only)
python run_adaptive_grid.py retrain --dry-run

# Full grid
python run_adaptive_grid.py retrain --skip-existing

# Or via shell wrapper
chmod +x scripts/run_adaptive_retrain.sh
nohup bash scripts/run_adaptive_retrain.sh > adaptive_retrain.log 2>&1 &
tail -f adaptive_retrain.log
```

**Checkpoint layout**

```
phase2/artifacts/adaptive/retrain/
  clean/transformer_masked/transformer_masked_best_acc.pt
  clean/cnn_bilstm/cnn_bilstm_best_acc.pt
  jitter_low/transformer_masked/...
  ...
```

## Step 3 — Fine-tune from frozen checkpoint (16 runs, ~6–8 A100-hours)

Warm-starts from `transformer_production.pt` / `cnn_bilstm_best.pt`, trains 8 epochs at lr=1e-4 on obfuscated train data.

```bash
cd phase2

python run_adaptive_grid.py finetune --skip-existing

# Head-only fine-tune (optional ablation)
python run_adaptive_grid.py finetune --freeze-backbone --skip-existing

# Shell wrapper
nohup bash scripts/run_adaptive_finetune.sh > adaptive_finetune.log 2>&1 &
```

**Checkpoint layout**

```
phase2/artifacts/adaptive/finetune/{policy}/{model}/{model}_best_acc.pt
```

## Step 4 — Full pipeline (recommended)

```bash
cd phase2
chmod +x scripts/run_adaptive_*.sh

nohup bash scripts/run_adaptive_pipeline.sh > adaptive_pipeline.log 2>&1 &
tail -f adaptive_pipeline.log
```

Skip re-generating splits if already done:

```bash
SKIP_OBFUSCATE=1 bash scripts/run_adaptive_pipeline.sh
```

## Partial runs

```bash
# Single policy, single model
python run_adaptive_grid.py retrain --only jitter_low --models transformer_masked

# Subset of jitter policies
python run_adaptive_grid.py finetune --only jitter_low jitter_medium jitter_high

# Regenerate registry after manual edits
python scripts/build_adaptive_registry.py
```

## Registry

After training, `artifacts/adaptive/adaptive_registry.json` maps every job to:

- `train_pt`, `val_pt`, `test_stem`
- `checkpoint` path (best val-acc)
- `frozen_init_checkpoint` (finetune jobs only)
- `status`: `complete` | `missing`

Use this file to drive Phase 4 adaptive evaluation and Privacy Recovery Rate computation.

## New `train.py` flags

```bash
python train.py \
  --model transformer_masked \
  --train-pt ../phase3/artifacts/adaptive/train/jitter_low_train.pt \
  --val-pt ../phase3/artifacts/adaptive/val/jitter_low_val.pt \
  --init-checkpoint artifacts/transformer_production.pt \
  --epochs 8 --lr 1e-4 --patience 3 \
  --d-model 160 --ff-dim 512 \
  --out-dir artifacts/adaptive/finetune/jitter_low/transformer_masked

# Optional: train classifier head only
  --freeze-backbone
```

## Files added

| File | Role |
|------|------|
| `phase3/adaptive_policies.py` | 8-policy grid definition |
| `phase3/generate_obfuscated_splits.py` | Obfuscate train/val |
| `phase2/adaptive_registry.py` | Paths + registry builder |
| `phase2/run_adaptive_grid.py` | 16-job orchestrator |
| `phase2/scripts/run_adaptive_*.sh` | GPU server wrappers |
| `phase2/train.py` | `--init-checkpoint`, `--freeze-backbone` |

## Next step (Phase 4)

Evaluate each adapted checkpoint on its **matched** obfuscated test tensor, then compute:

\[
\text{PRR} = \frac{F1_{\text{adapted}} - F1_{\text{frozen}}}{F1_{\text{baseline}} - F1_{\text{frozen}}}
\]

using frozen results already in `phase4/accuracy_results.csv` as \(F1_{\text{frozen}}\) and \(F1_{\text{baseline}}\).
