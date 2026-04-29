# Phase 2 - Model Training (A100)

This folder contains the Phase 2 training stack for QUIC traffic classification.

## Files

- `models.py`: CNN+BiLSTM and Transformer classifiers.
- `train.py`: AMP-enabled single-model trainer with AdamW, cosine scheduler, class-weighted loss, early stopping, checkpointing, and metric logging.
- `train_all.py`: convenience runner that trains both models sequentially with shared settings.
- `hyperparameter_sweep.py`: grid sweep over learning rate, weight decay, and batch size.

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

## Train Single Model

```bash
cd phase2
python train.py --model cnn_bilstm --batch-size 1024 --num-workers 4 --epochs 60 --patience 10
```

```bash
cd phase2
python train.py --model transformer --batch-size 1024 --num-workers 4 --epochs 60 --patience 10
```

## Train Both Models (Sequential)

```bash
cd phase2
python train_all.py --batch-size 1024 --num-workers 4 --epochs 60 --patience 10 --amp-dtype bf16
```

This writes model-specific outputs under:
- `./artifacts/cnn_bilstm/`
- `./artifacts/transformer/`

## Hyperparameter Sweep

Default sweep explores:
- learning rates: `1e-3`, `3e-4`, `1e-4`
- weight decays: `1e-2`, `1e-3`
- batch sizes: `512`, `1024`, `2048`

```bash
cd phase2
python hyperparameter_sweep.py --model cnn_bilstm --epochs 30 --patience 10
```

Custom search space example:

```bash
cd phase2
python hyperparameter_sweep.py \
  --model transformer \
  --learning-rates 0.001 0.0003 \
  --weight-decays 0.01 0.001 \
  --batch-sizes 512 1024
```

## Outputs

For `train.py` runs (default `--out-dir ./artifacts`), saved in `./artifacts/`:
- `cnn_bilstm_best.pt` or `transformer_best.pt`
- `*_history.csv` (loss/accuracy logs per epoch)
- `*_config.json` (run config for reproducibility)

For `train_all.py` and sweep runs, each run gets its own output directory.

## Notes

- Uses class-weighted cross entropy to mitigate imbalance.
- Uses AMP (`bf16` by default; switch with `--amp-dtype fp16`).
- Early stopping monitors validation loss.
- Best checkpoint is restored after training ends.
- If you run on CPU, training still works but AMP is disabled automatically.
