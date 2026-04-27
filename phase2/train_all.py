from train import TrainConfig, run_training


def main() -> None:
    common = dict(
        train_pt="../phase1/artifacts/train_tensors.pt",
        val_pt="../phase1/artifacts/val_tensors.pt",
        out_dir="./artifacts",
        batch_size=1024,
        num_workers=4,
        pin_memory=True,
        epochs=60,
        patience=10,
        lr=3e-4,
        weight_decay=1e-2,
        t_max=60,
        min_lr=1e-6,
        amp_dtype="bf16",
        seed=42,
    )

    run_training(TrainConfig(model="cnn_bilstm", **common))
    run_training(TrainConfig(model="transformer", **common))


if __name__ == "__main__":
    main()
