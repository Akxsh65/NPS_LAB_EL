"""
Run adaptive-adversary training grid: retrain-from-scratch or fine-tune.

Usage (from phase2/):
  python run_adaptive_grid.py retrain --device cuda
  python run_adaptive_grid.py finetune --device cuda
  python run_adaptive_grid.py retrain --only jitter_low --models transformer_masked
  python run_adaptive_grid.py finetune --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from adaptive_registry import (
    POLICIES,
    best_checkpoint_path,
    job_id,
    resolve_frozen_checkpoint,
    run_out_dir,
    train_pt_for_policy,
    val_pt_for_policy,
    write_registry,
)
from train import TrainConfig, run_training

AttackModel = str
MODES = ("retrain", "finetune")
MODELS = ("transformer_masked", "cnn_bilstm")


def _load_stage2_winner_env() -> dict:
    """Best-effort load of Stage 2 winner hyperparameters."""
    env_path = Path(__file__).resolve().parent / "scripts" / "stage2_winner.env"
    values: dict = {
        "WIN_BS": 1024,
        "WIN_LR": 1e-3,
        "WIN_WD": 1e-2,
        "WIN_LS": 0.0,
        "WIN_EPOCHS": 80,
        "WIN_PATIENCE": 15,
        "WIN_TMAX": 80,
        "WIN_WARMUP": 3,
    }
    if not env_path.is_file():
        return values
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if key in values:
            if key in ("WIN_BS", "WIN_EPOCHS", "WIN_PATIENCE", "WIN_TMAX", "WIN_WARMUP"):
                values[key] = int(val)
            else:
                values[key] = float(val)
    return values


def _transformer_train_config(
    policy_key: str,
    out_dir: Path,
    mode: str,
    winner: dict,
    init_checkpoint: str | None,
    freeze_backbone: bool,
) -> TrainConfig:
    if mode == "retrain":
        return TrainConfig(
            model="transformer_masked",
            train_pt=train_pt_for_policy(policy_key),
            val_pt=val_pt_for_policy(policy_key),
            out_dir=str(out_dir),
            batch_size=int(winner["WIN_BS"]),
            lr=float(winner["WIN_LR"]),
            weight_decay=float(winner["WIN_WD"]),
            label_smoothing=float(winner["WIN_LS"]),
            epochs=int(winner["WIN_EPOCHS"]),
            patience=int(winner["WIN_PATIENCE"]),
            t_max=int(winner["WIN_TMAX"]),
            warmup_epochs=int(winner["WIN_WARMUP"]),
            d_model=160,
            nhead=8,
            num_layers=4,
            ff_dim=512,
            dropout=0.2,
            init_checkpoint=init_checkpoint,
            freeze_backbone=freeze_backbone,
        )

    return TrainConfig(
        model="transformer_masked",
        train_pt=train_pt_for_policy(policy_key),
        val_pt=val_pt_for_policy(policy_key),
        out_dir=str(out_dir),
        batch_size=int(winner["WIN_BS"]),
        lr=1e-4,
        weight_decay=float(winner["WIN_WD"]),
        label_smoothing=0.0,
        epochs=8,
        patience=3,
        t_max=8,
        warmup_epochs=0,
        d_model=160,
        nhead=8,
        num_layers=4,
        ff_dim=512,
        dropout=0.2,
        init_checkpoint=init_checkpoint,
        freeze_backbone=freeze_backbone,
    )


def _bilstm_train_config(
    policy_key: str,
    out_dir: Path,
    mode: str,
    init_checkpoint: str | None,
    freeze_backbone: bool,
) -> TrainConfig:
    if mode == "retrain":
        return TrainConfig(
            model="cnn_bilstm",
            train_pt=train_pt_for_policy(policy_key),
            val_pt=val_pt_for_policy(policy_key),
            out_dir=str(out_dir),
            batch_size=1024,
            lr=1e-3,
            weight_decay=1e-2,
            label_smoothing=0.05,
            epochs=60,
            patience=10,
            t_max=60,
            init_checkpoint=init_checkpoint,
            freeze_backbone=freeze_backbone,
        )

    return TrainConfig(
        model="cnn_bilstm",
        train_pt=train_pt_for_policy(policy_key),
        val_pt=val_pt_for_policy(policy_key),
        out_dir=str(out_dir),
        batch_size=1024,
        lr=1e-4,
        weight_decay=1e-2,
        label_smoothing=0.0,
        epochs=8,
        patience=3,
        t_max=8,
        init_checkpoint=init_checkpoint,
        freeze_backbone=freeze_backbone,
    )


def build_job_config(
    mode: str,
    policy_key: str,
    model: str,
    winner: dict,
    freeze_backbone: bool,
    init_checkpoint: str | None = None,
) -> TrainConfig:
    out_dir = run_out_dir(mode, policy_key, model)
    init_ckpt = init_checkpoint
    if mode == "finetune" and init_ckpt is None:
        init_ckpt = resolve_frozen_checkpoint(model)

    if model == "transformer_masked":
        return _transformer_train_config(
            policy_key, out_dir, mode, winner, init_ckpt, freeze_backbone
        )
    if model == "cnn_bilstm":
        return _bilstm_train_config(policy_key, out_dir, mode, init_ckpt, freeze_backbone)
    raise ValueError(f"Unknown model: {model}")


def _verify_inputs(cfg: TrainConfig) -> None:
    for label, path in (("train", cfg.train_pt), ("val", cfg.val_pt)):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing {label} tensor for adaptive job: {path}\n"
                "Run: cd ../phase3 && python generate_obfuscated_splits.py"
            )
    if cfg.init_checkpoint and not os.path.isfile(cfg.init_checkpoint):
        raise FileNotFoundError(f"Missing init checkpoint: {cfg.init_checkpoint}")


def run_one_job(
    mode: str,
    policy_key: str,
    model: str,
    winner: dict,
    skip_existing: bool,
    dry_run: bool,
    freeze_backbone: bool,
) -> dict:
    out_dir = run_out_dir(mode, policy_key, model)
    ckpt = best_checkpoint_path(out_dir, model)
    jid = job_id(mode, policy_key, model)

    if skip_existing and ckpt.is_file():
        print(f"SKIP existing {jid} -> {ckpt}")
        return {"job_id": jid, "status": "skipped", "checkpoint": str(ckpt)}

    init_ckpt = None
    if mode == "finetune":
        if dry_run:
            try:
                init_ckpt = resolve_frozen_checkpoint(model)
            except FileNotFoundError:
                init_ckpt = f"<missing frozen checkpoint for {model}>"
        else:
            init_ckpt = resolve_frozen_checkpoint(model)

    cfg = build_job_config(
        mode, policy_key, model, winner, freeze_backbone, init_checkpoint=init_ckpt
    )

    if not dry_run:
        _verify_inputs(cfg)

    print("\n" + "=" * 72)
    print(f"JOB {jid}")
    print(f"  train: {cfg.train_pt}")
    print(f"  val:   {cfg.val_pt}")
    print(f"  out:   {cfg.out_dir}")
    if cfg.init_checkpoint:
        print(f"  init:  {cfg.init_checkpoint}  freeze_backbone={cfg.freeze_backbone}")
    print(f"  hp:    epochs={cfg.epochs} lr={cfg.lr} bs={cfg.batch_size}")
    print("=" * 72)

    if dry_run:
        return {"job_id": jid, "status": "dry_run", "config": cfg.__dict__}

    out_dir.mkdir(parents=True, exist_ok=True)
    run_training(cfg)
    status = "complete" if ckpt.is_file() else "failed"
    return {"job_id": jid, "status": status, "checkpoint": str(ckpt) if ckpt.is_file() else None}


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive adversary training grid")
    parser.add_argument("mode", choices=MODES, help="retrain (from scratch) or finetune")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Policy keys (default: all 8 including clean)",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(MODELS),
        choices=MODELS,
        help="Attack architectures to train",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Fine-tune head only (requires finetune mode + init checkpoint)",
    )
    parser.add_argument(
        "--generate-splits",
        action="store_true",
        help="Run phase3/generate_obfuscated_splits.py before training",
    )
    args = parser.parse_args()

    if args.freeze_backbone and args.mode != "finetune":
        parser.error("--freeze-backbone only applies to finetune mode")

    policy_keys = [p.key for p in POLICIES]
    if args.only:
        unknown = set(args.only) - set(policy_keys)
        if unknown:
            parser.error(f"Unknown policies: {sorted(unknown)}")
        policy_keys = args.only

    if args.generate_splits:
        script = Path(__file__).resolve().parent.parent / "phase3" / "generate_obfuscated_splits.py"
        print(f">>> Generating obfuscated train/val splits: {script}")
        subprocess.check_call([sys.executable, str(script), "--skip-existing"])

    winner = _load_stage2_winner_env()
    results = []
    total = len(policy_keys) * len(args.models)
    print(f"Mode={args.mode}  jobs={total}  policies={policy_keys}  models={args.models}")

    for policy_key in policy_keys:
        for model in args.models:
            result = run_one_job(
                args.mode,
                policy_key,
                model,
                winner,
                skip_existing=args.skip_existing,
                dry_run=args.dry_run,
                freeze_backbone=args.freeze_backbone,
            )
            results.append(result)

    if not args.dry_run:
        registry_path = write_registry()
        print(f"\nRegistry updated -> {registry_path}")

    summary_path = Path("artifacts/adaptive") / f"{args.mode}_run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"mode": args.mode, "results": results}, f, indent=2)
    print(f"Run summary -> {summary_path}")

    failed = [r for r in results if r.get("status") == "failed"]
    if failed:
        raise SystemExit(f"{len(failed)} job(s) failed")


if __name__ == "__main__":
    main()
