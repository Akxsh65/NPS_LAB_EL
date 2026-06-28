# Reproducibility appendix

This document lists artifacts, seeds, and commands to reproduce Phase 4 numbers cited
in the Phase 5 manuscript.

## Dataset

| Item | Value |
|------|-------|
| Dataset | CESNET-QUIC22-XS (Zenodo) |
| Train week | W-2022-44 |
| Test week | W-2022-45 (held out) |
| Test flows | 49,305 |
| Classes | 64 (after MIN_CLASS_SAMPLES filter) |
| Tensor shape | (N, 3, 30) — channels: IPT, direction, packet size |

## Attack models (frozen at test time)

| Model | Checkpoint | Config |
|-------|------------|--------|
| Masked Transformer | `phase2/artifacts/transformer_production.pt` | `phase2/artifacts/refine/architecture/run_masked_d160/transformer_masked_config.json` (d_model=160) |
| CNN-BiLSTM | `phase2/artifacts/cnn_bilstm_best.pt` | From sweep `run_01_bs1024_lr0.001_wd0.01` |

## Defense settings (Phase 3)

Obfuscated tensors: `phase3/artifacts/obfuscated_*.pt` with overhead manifest
(`phase3/artifacts/obfuscation_manifest.json`).

| Setting | Mechanism |
|---------|-----------|
| `jitter_low` / `_medium` / `_high` | Laplace timing jitter on non-IPT packets |
| `linear128` | Round packet sizes to 128 B blocks |
| `linear128_jitter_medium` | Combined |
| `mtu` / `mtu_jitter_medium` | Pad to MTU (1500 B) |

## Evaluation commands

```bash
# Full Transformer sweep (Tier A/B)
cd phase4
python run_experiments.py --device cuda --checkpoint ../phase2/artifacts/transformer_production.pt

# Tier C (statistics + ablation + architecture compare)
python run_tier_c.py --all --device cuda \
  --checkpoint ../phase2/artifacts/transformer_production.pt

python run_tier_c.py --bilstm --compare-architectures --device cuda \
  --bilstm-checkpoint ../phase2/artifacts/cnn_bilstm_best.pt
```

## Key output files

| File | Content |
|------|---------|
| `phase4/results/accuracy_results.csv` | Transformer metrics |
| `phase4/results/accuracy_results_bilstm.csv` | BiLSTM metrics |
| `phase4/bootstrap_ci.csv` | Bootstrap 95% CIs (2000 resamples) |
| `phase4/paired_test_baseline_vs_obfuscated_jitter_low.json` | Paired bootstrap + McNemar |
| `phase4/architecture_comparison.csv` | Per-setting architecture gaps |
| `phase4/channel_ablation.csv` | Single/dual-channel ablation |
| `phase4/results/pareto_frontier_table.csv` | Formal Pareto dominance |

Phase 5 copies statistical CSVs into `phase5/data/` for manuscript drafting.

## Environment

Pin versions on the evaluation machine (see `phase4/requirements.txt`). sklearn
version affects per-class report label formatting only; metrics are unchanged.
