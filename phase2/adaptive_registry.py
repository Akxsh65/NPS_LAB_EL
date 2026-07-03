"""
Paths and naming conventions for adaptive-adversary training (Phase 2).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Literal

PHASE2_ROOT = Path(__file__).resolve().parent
PHASE3_ROOT = PHASE2_ROOT.parent / "phase3"
PHASE1_ARTIFACTS = PHASE2_ROOT.parent / "phase1" / "artifacts"
PHASE2_ARTIFACTS = PHASE2_ROOT / "artifacts"
PHASE3_ADAPTIVE = PHASE3_ROOT / "artifacts" / "adaptive"

if str(PHASE3_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE3_ROOT))

from adaptive_policies import POLICIES, AdaptivePolicy, get_policy  # noqa: E402

__all__ = [
    "POLICIES",
    "get_policy",
    "AdaptivePolicy",
    "ADAPTIVE_ROOT",
    "REGISTRY_PATH",
    "resolve_frozen_checkpoint",
    "train_pt_for_policy",
    "val_pt_for_policy",
    "run_out_dir",
    "best_checkpoint_path",
    "job_id",
    "write_registry",
    "load_registry",
]

AttackModel = Literal["transformer_masked", "cnn_bilstm"]
AdaptiveMode = Literal["retrain", "finetune"]

ADAPTIVE_ROOT = PHASE2_ARTIFACTS / "adaptive"
REGISTRY_PATH = ADAPTIVE_ROOT / "adaptive_registry.json"
SPLITS_MANIFEST = PHASE3_ADAPTIVE / "adaptive_splits_manifest.json"

# Frozen production checkpoints (non-adaptive baseline attackers)
FROZEN_CHECKPOINTS: dict[AttackModel, Path] = {
    "transformer_masked": PHASE2_ARTIFACTS / "transformer_production.pt",
    "cnn_bilstm": PHASE2_ARTIFACTS / "cnn_bilstm_best.pt",
}

FROZEN_CHECKPOINT_FALLBACKS: dict[AttackModel, tuple[Path, ...]] = {
    "transformer_masked": (
        PHASE2_ARTIFACTS / "refine" / "architecture" / "run_masked_d160" / "transformer_masked_best_acc.pt",
        PHASE2_ARTIFACTS / "transformer_best_acc.pt",
    ),
    "cnn_bilstm": (
        PHASE2_ARTIFACTS / "cnn_bilstm" / "cnn_bilstm_best.pt",
    ),
}


def resolve_frozen_checkpoint(model: AttackModel) -> str:
    primary = FROZEN_CHECKPOINTS[model]
    if primary.is_file():
        return str(primary.resolve())
    for alt in FROZEN_CHECKPOINT_FALLBACKS.get(model, ()):
        if alt.is_file():
            return str(alt.resolve())
    raise FileNotFoundError(
        f"No frozen checkpoint for {model}. Expected {primary} or fallbacks."
    )


def train_pt_for_policy(policy: AdaptivePolicy | str) -> str:
    p = get_policy(policy) if isinstance(policy, str) else policy
    if p.is_clean:
        return str((PHASE1_ARTIFACTS / "train_tensors.pt").resolve())
    path = PHASE3_ADAPTIVE / "train" / f"{p.key}_train.pt"
    return str(path.resolve())


def val_pt_for_policy(policy: AdaptivePolicy | str) -> str:
    p = get_policy(policy) if isinstance(policy, str) else policy
    if p.is_clean:
        return str((PHASE1_ARTIFACTS / "val_tensors.pt").resolve())
    path = PHASE3_ADAPTIVE / "val" / f"{p.key}_val.pt"
    return str(path.resolve())


def run_out_dir(mode: AdaptiveMode, policy_key: str, model: AttackModel) -> Path:
    return ADAPTIVE_ROOT / mode / policy_key / model


def best_checkpoint_path(out_dir: Path, model: AttackModel) -> Path:
    return out_dir / f"{model}_best_acc.pt"


def job_id(mode: AdaptiveMode, policy_key: str, model: AttackModel) -> str:
    return f"{mode}_{policy_key}_{model}"


def build_registry_entry(
    mode: AdaptiveMode,
    policy: AdaptivePolicy,
    model: AttackModel,
    out_dir: Path,
) -> dict:
    ckpt = best_checkpoint_path(out_dir, model)
    return {
        "job_id": job_id(mode, policy.key, model),
        "mode": mode,
        "policy_key": policy.key,
        "attack_model": model,
        "train_pt": train_pt_for_policy(policy),
        "val_pt": val_pt_for_policy(policy),
        "test_stem": policy.test_stem,
        "out_dir": str(out_dir.resolve()),
        "checkpoint": str(ckpt.resolve()) if ckpt.is_file() else None,
        "frozen_init_checkpoint": resolve_frozen_checkpoint(model) if mode == "finetune" else None,
        "status": "complete" if ckpt.is_file() else "missing",
    }


def write_registry(path: Path | None = None) -> str:
    path = path or REGISTRY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    jobs = []
    for mode in ("retrain", "finetune"):
        for policy in POLICIES:
            for model in ("transformer_masked", "cnn_bilstm"):
                out_dir = run_out_dir(mode, policy.key, model)
                jobs.append(build_registry_entry(mode, policy, model, out_dir))

    frozen = {}
    for model in FROZEN_CHECKPOINTS:
        try:
            frozen[model] = resolve_frozen_checkpoint(model)
        except FileNotFoundError:
            frozen[model] = None

    payload = {
        "registry_version": "1.0",
        "splits_manifest": str(SPLITS_MANIFEST.resolve()) if SPLITS_MANIFEST.is_file() else None,
        "frozen_checkpoints": frozen,
        "jobs": jobs,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return str(path.resolve())


def load_registry(path: Path | None = None) -> dict:
    path = path or REGISTRY_PATH
    with open(path, encoding="utf-8") as f:
        return json.load(f)
