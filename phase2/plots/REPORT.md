# HPO Sweep Analysis Report
Model: transformer
Runs found: 10
Best config: bs=1024, lr=1e-03, wd=1e-02
Best val accuracy: 0.8038 (80.38%)
Best epoch: 49
Train acc @ best: 0.8099
Acc gap @ best: 0.0061 (0.61%) - Good fit

## Interpretation
- acc_gap = train_acc - val_acc. Large gap => overfitting.
- loss_gap = val_loss - train_loss. Rising val loss with falling train loss => classic overfit.
- Points far below diagonal in train_vs_val_scatter => high train, lower val.

## Outputs
- sweep_summary.csv
- per_run/*_curves.png
- comparative_*.png, heatmap_*.png, grid_all_runs_train_val.png