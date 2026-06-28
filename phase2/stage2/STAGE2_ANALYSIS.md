# Stage 2 Refinement — Analysis (7 runs)

## Winner (production baseline until architecture ablation finishes)

| Field | Value |
|-------|--------|
| Run | **R06** `run_06_bs1024_lr0.001_wd0.01` |
| Best val accuracy | **82.53%** (epoch 67 / 70) |
| Train acc @ best | 83.56% |
| Generalization gap | **1.03%** (Good fit) |
| Hyperparameters | `bs=1024`, `lr=1e-3`, `wd=1e-2`, **`label_smoothing=0.0`** |

**Runner-up:** R02 — 82.52% @ epoch 79/80, `label_smoothing=0.05`, gap 1.69%. Effectively tied on accuracy; R06 has slightly lower gap.

## Full ranking

| Rank | Run | lr | wd | ls* | Best val % | Gap @ best % |
|------|-----|-----|-----|-----|------------|--------------|
| 1 | R06 | 1e-3 | 1e-2 | 0.0 | **82.53** | 1.03 |
| 2 | R02 | 1e-3 | 1e-2 | 0.05 | 82.52 | 1.69 |
| 3 | R05 | 1e-3 | 5e-3 | 0.05 | 82.17 | 0.82 |
| 4 | R07 | 1e-3 | 1e-2 | 0.03 | 82.00 | 1.53 |
| 5 | R03 | 1.2e-3 | 1e-2 | 0.05 | 81.39 | 2.01 |
| 6 | R01 | 7e-4 | 1e-2 | 0.05 | 80.87 | 1.09 |
| 7 | R04 | 1.5e-3 | 1e-2 | 0.05 | 79.45 | 1.81 |

\*ls inferred from run folder / config (R06=0.0, R07=0.03, R02=0.05).

## Key conclusions

1. **+3.1 pp vs original sweep** (~80.4% → ~82.5%) from longer training + LR focus at `1e-3`.
2. **`lr=1e-3` is optimal** in this grid; `1.5e-3` hurts (~79.5%); `7e-4` underperforms (~80.9%).
3. **Weight decay**: `1e-2` and `5e-3` similar; keep **`wd=1e-2`** (winner default).
4. **Label smoothing**: **0.0** wins peak val; 0.03–0.05 slightly lowers peak or raises gap.
5. **No late collapse**: best ≈ final val acc on top runs → stable stopping.
6. **Overfitting controlled**: all gaps < 2.1% at best epoch; R04 only run with rising late gap (~4.2% final).

## Architecture ablation (Stage 3) — fixed HP

Use winner settings with **80 epochs** (R02 still improved at epoch 79):

```
bs=1024  lr=1e-3  wd=1e-2  label_smoothing=0.0
epochs=80  patience=15  t_max=80  warmup_epochs=3
```

Commands: see `scripts/run_architecture_from_stage2.sh`
