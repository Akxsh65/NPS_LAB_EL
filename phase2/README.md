# Phase 2 - Model Training (A100)

This folder contains the Phase 2 training stack for QUIC traffic classification.

## Files

- `models.py`: CNN+BiLSTM and Transformer classifiers.
- `train.py`: AMP-enabled training loop with AdamW, cosine scheduler, class-weighted loss, and early stopping.
- `hyperparameter_sweep.py`: basic LR/weight-decay sweep runner.

## Expected Inputs (from Phase 1)

- `../phase1/artifacts/train_tensors.pt`
- `../phase1/artifacts/val_tensors.pt`

Each `.pt` file must contain:
- `X`: float32 tensor of shape `(N, 3, 30)`
- `y`: int64 labels

## Install (A100 machine)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy tqdm
```

## Train CNN+BiLSTM

```bash
cd phase2
python train.py --model cnn_bilstm --batch-size 1024 --num-workers 4 --epochs 60 --patience 10
```

## Train Transformer

```bash
cd phase2
python train.py --model transformer --batch-size 1024 --num-workers 4 --epochs 60 --patience 10
```

## Outputs

Saved in `./artifacts/`:
- `cnn_bilstm_best.pt` or `transformer_best.pt`
- `*_history.csv` (loss/accuracy logs per epoch)
- `*_config.json` (run config for reproducibility)

## Notes

- Uses class-weighted cross entropy to mitigate imbalance.
- Uses AMP (`bf16` by default; switch with `--amp-dtype fp16`).
- Early stopping monitors validation accuracy.
- If you run on CPU, training still works but AMP is disabled automatically.
