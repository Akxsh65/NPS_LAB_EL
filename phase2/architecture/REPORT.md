# HPO Sweep Analysis Report
Model: transformer
Runs found: 2
Best config: bs=1024, lr=1e-03, wd=1e-02
Best val accuracy: 0.8568 (85.68%)
Best epoch: 69
Train acc @ best: 0.8723
Acc gap @ best: 0.0155 (1.55%) - Good fit

## Interpretation
- acc_gap = train_acc - val_acc. Large gap => overfitting.
- loss_gap = val_loss - train_loss. Rising val loss with falling train loss => classic overfit.
- Points far below diagonal in train_vs_val_scatter => high train, lower val.

## Outputs
- sweep_summary.csv
- per_run/*_curves.png
- comparative_*.png, heatmap_*.png, grid_all_runs_train_val.png