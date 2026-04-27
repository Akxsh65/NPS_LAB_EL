import argparse
import itertools
from pathlib import Path

from train import TrainConfig, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple grid sweep for Phase 2")
    parser.add_argument("--model", required=True, choices=["cnn_bilstm", "transformer"])
    parser.add_argument("--train-pt", default="../phase1/artifacts/train_tensors.pt")
    parser.add_argument("--val-pt", default="../phase1/artifacts/val_tensors.pt")
    parser.add_argument("--out-dir", default="./artifacts/sweeps")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16"])
    args = parser.parse_args()

    learning_rates = [1e-3, 3e-4, 1e-4]
    weight_decays = [1e-2, 1e-3]

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    run_idx = 0
    for lr, wd in itertools.product(learning_rates, weight_decays):
        run_idx += 1
        run_out = Path(args.out_dir) / f"run_{run_idx:02d}_lr{lr}_wd{wd}"
        cfg = TrainConfig(
            model=args.model,
            train_pt=args.train_pt,
            val_pt=args.val_pt,
            out_dir=str(run_out),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            epochs=args.epochs,
            patience=args.patience,
            lr=lr,
            weight_decay=wd,
            t_max=args.epochs,
            amp_dtype=args.amp_dtype,
        )
        print(f"\n=== Sweep run {run_idx}: lr={lr}, wd={wd} ===")
        run_training(cfg)


if __name__ == "__main__":
    main()
