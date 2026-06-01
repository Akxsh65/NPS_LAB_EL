# Phase 4 — Evaluation

Evaluates the frozen **Transformer** on:
- clean test tensors (`phase1/artifacts/test_tensors.pt`)
- obfuscated tensors from Phase 3 (`phase3/artifacts/obfuscated_*.pt`)

## Prerequisites

1. Phase 1 artifacts (test tensors, label encoder, ipt scaler)
2. Phase 2 checkpoint: `phase2/artifacts/transformer_best_acc.pt` (or `transformer_best.pt`)
3. Phase 3 obfuscated files: run `python generate_obfuscated.py` in `phase3/`

## Run full evaluation

```bash
cd phase4
pip install -r requirements.txt
python run_experiments.py
```

## Outputs (`phase4/results/`)

- `accuracy_results.csv` — accuracy, F1, overhead, accuracy drop
- `summary.json`
- `pareto_bandwidth_accuracy.png`
- `pareto_latency_accuracy.png`
- `accuracy_drop_bars.png`
- `accuracy_comparison_bars.png`
- `confusion_*.png` (per setting)

## Single dataset eval

```bash
python evaluate.py --pt ../phase1/artifacts/test_tensors.pt --save-cm
```
