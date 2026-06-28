# Phase 4 — Evaluation

Evaluates the frozen **transformer_masked (d_model=160)** on clean and obfuscated **test** tensors.

## Prerequisites

1. `phase1/artifacts/test_tensors.pt`, `label_encoder.pkl`
2. `phase2/artifacts/transformer_production.pt` (copy from `run_masked_d160/transformer_masked_best_acc.pt`)
3. `phase2/artifacts/refine/architecture/run_masked_d160/transformer_masked_config.json`
4. `phase3/artifacts/obfuscated_*.pt` + `obfuscation_manifest.json` (7 experiments)

## Run full evaluation (GPU)

```bash
cd phase4
pip install -r requirements.txt

# One-time: production checkpoint
cp ../phase2/artifacts/refine/architecture/run_masked_d160/transformer_masked_best_acc.pt \
   ../phase2/artifacts/transformer_production.pt

python run_experiments.py --device cuda --batch-size 1024

# Faster (no confusion matrices):
python run_experiments.py --device cuda --skip-cm
```

## Outputs (`phase4/results/`)

- `accuracy_results.csv` — accuracy, macro F1, overhead, drops vs baseline
- `summary.json`
- `pareto_*.png`, `accuracy_*_bars.png` (legacy)
- **Tier A (publication):** `macro_f1_*_bars.png`, `dual_metric_top5_bars.png`,
  `pareto_bw_accuracy_practical.png`, `pareto_latency_accuracy_practical.png`,
  `pareto_bw_accuracy_full.png`, `pareto_frontier_table.csv`, `table_phase4.tex`
- `reports/*_classification_report.txt`, `reports/*_per_class.csv`
- `confusion_*.png` (unless `--skip-cm`)

### Tier A / B plots only (from existing CSV)

```bash
python plot_publication.py --csv accuracy_results.csv --out-dir results
# Tier A only, Tier B only (accuracy bars), or both (default):
python plot_publication.py --csv accuracy_results.csv --tier-a-only
python plot_publication.py --csv accuracy_results.csv --tier-b-only
python pareto.py --csv accuracy_results.csv --tier-a-only
```

**Tier B** adds: chance line (1.56%) on all bar charts, gold star on `jitter_low`,
`accuracy_comparison_bars.png` / `accuracy_drop_bars.png`, LaTeX via `scripts/export_results_table.py`.

### Confusion matrices (baseline + jitter_low, top-20 inset)

```bash
python run_experiments.py --confusion-only --device cuda \
  --checkpoint ../phase2/artifacts/transformer_production.pt
```

Outputs: `results/confusion_baseline.png`, `results/confusion_obfuscated_jitter_low.png`.

### LaTeX table

```bash
python ../scripts/export_results_table.py --csv accuracy_results.csv
```

### Per-class reports only (3 GPU passes)

```bash
python run_experiments.py --reports-only --save-reports --device cuda \
  --checkpoint ../phase2/artifacts/transformer_production.pt
```

## Tier C — statistics & architecture comparison

Requires `phase2/artifacts/cnn_bilstm_best.pt` (or `cnn_bilstm/cnn_bilstm_best.pt`) for BiLSTM runs.

```bash
# Full Tier C on GPU
python run_tier_c.py --all --device cuda --batch-size 1024 \
  --checkpoint ../phase2/artifacts/transformer_production.pt

# Step-by-step:
python run_tier_c.py --transformer-predictions --device cuda --checkpoint ../phase2/artifacts/transformer_production.pt
python run_tier_c.py --bilstm --device cuda
python run_tier_c.py --bootstrap --paired --compare-architectures
python run_tier_c.py --channel-ablation --device cuda --checkpoint ../phase2/artifacts/transformer_production.pt
python run_tier_c.py --summarize-reports
```

**Tier C outputs:** `accuracy_results_bilstm.csv`, `architecture_comparison.csv`,
`bootstrap_ci.csv`, `paired_test_*.json`, `channel_ablation.csv`, `worst_classes_summary.csv`,
`predictions/{transformer,cnn_bilstm}/*.npz`

```bash
python run_experiments.py --device cuda --save-predictions --skip-cm
python run_experiments.py --attack-model cnn_bilstm --device cuda --skip-cm
```

## Single dataset

```bash
python evaluate.py --pt ../phase1/artifacts/test_tensors.pt --device cuda --save-cm
```

Baseline was **77.77%** test accuracy / **74.41%** macro F1 (Phase 2 finalize). Report privacy drop vs those numbers.

**Pareto note:** On the bandwidth axis, all jitter settings share 0% overhead, so the formal frontier
includes `jitter_high` (strongest privacy at zero BW), not `jitter_low`. Use
`pareto_latency_accuracy_practical.png` to compare jitter tiers; cite `jitter_low` as the
recommended operating point (small latency, minimal F1 drop) even when it is dominated on BW.
