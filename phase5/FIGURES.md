# Figure manifest (Phase 5 manuscript)

All files live in `manuscript/figures/` unless noted.

| Paper fig | LaTeX label | Source file | Caption summary |
|-----------|-------------|-------------|-----------------|
| Fig. 1 | `fig:macro_f1` | `macro_f1_comparison_bars.png` | Macro F1 across all defense settings (chance line at 1.56%) |
| Fig. 2 | `fig:pareto_lat` | `pareto_latency_accuracy_practical.png` | Latency overhead vs. accuracy (jitter tiers highlighted) |
| Fig. 3 | `fig:pareto_bw` | `pareto_bw_accuracy_practical.png` | Bandwidth overhead vs. accuracy (padding tiers; MTU omitted from practical view) |
| Fig. 4 | `fig:dual_metric` | `dual_metric_top5_bars.png` | Top-5 settings by dual privacy–cost score |
| Fig. 5a | `fig:confusion_base` | `confusion_baseline.png` | Confusion matrix, clean test traffic |
| Fig. 5b | `fig:confusion_jlow` | `confusion_obfuscated_jitter_low.png` | Confusion matrix under `jitter_low` |
| Table I | `tab:main_results` | `tables/table_phase4.tex` | Macro F1, accuracy, ΔF1, BW/latency overhead |
| Table II | `tab:arch` | `tables/table_architecture_comparison.tex` | Transformer vs. CNN-BiLSTM per setting |

## Supplementary (not in main draft)

| File | Use |
|------|-----|
| `macro_f1_drop_bars.png` | Supplement: absolute F1 degradation |
| `accuracy_comparison_bars.png` | Supplement: accuracy-focused view |
| `pareto_bw_accuracy_full.png` | Supplement: includes MTU point |
| `confusion_obfuscated_*.png` | Supplement: remaining defense matrices |
| `confusion_test_tensors.png` | Legacy duplicate of baseline — do not cite |

## Regenerating figures

From repo root (GPU server):

```bash
cd phase4
python plot_publication.py --device cuda
python pareto_frontier.py
```

Then copy PNGs into `phase5/manuscript/figures/`.
