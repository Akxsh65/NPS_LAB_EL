import argparse
from pathlib import Path

from train import TrainConfig, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train both Phase 2 models sequentially")
    parser.add_argument("--train-pt", type=str, default="../phase1/artifacts/train_tensors.pt")
    parser.add_argument("--val-pt", type=str, default="../phase1/artifacts/val_tensors.pt")
    parser.add_argument("--out-root", type=str, default="./artifacts")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.set_defaults(pin_memory=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--monitor-metric", type=str, default="val_acc", choices=["val_acc", "val_loss"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    common = dict(
        train_pt=args.train_pt,
        val_pt=args.val_pt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        t_max=args.epochs,
        min_lr=args.min_lr,
        amp_dtype=args.amp_dtype,
        label_smoothing=args.label_smoothing,
        max_grad_norm=args.max_grad_norm,
        monitor_metric=args.monitor_metric,
        seed=args.seed,
    )

    run_training(
        TrainConfig(
            model="cnn_bilstm",
            out_dir=str(out_root / "cnn_bilstm"),
            **common,
        )
    )
    run_training(
        TrainConfig(
            model="transformer",
            out_dir=str(out_root / "transformer"),
            **common,
        )
    )


if __name__ == "__main__":
    main()
