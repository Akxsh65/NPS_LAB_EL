#!/usr/bin/env python3
"""
Inspect and summarize PyTorch .pt files.

Usage examples:
  python inspect_pt.py --file artifacts/train_tensors.pt
  python inspect_pt.py --file artifacts/train_tensors.pt --max-items 20 --sample 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


def _tensor_stats(t: torch.Tensor, sample_size: int) -> str:
    shape = tuple(t.shape)
    dtype = str(t.dtype)
    device = str(t.device)
    base = f"Tensor(shape={shape}, dtype={dtype}, device={device})"

    if t.numel() == 0:
        return base + " [empty]"

    # Convert a small sample to float for robust numeric summary.
    flat = t.detach().reshape(-1)
    sample = flat[: min(sample_size, flat.numel())]

    if torch.is_floating_point(sample) or sample.dtype in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        sample_f = sample.float()
        min_v = float(sample_f.min().item())
        max_v = float(sample_f.max().item())
        mean_v = float(sample_f.mean().item())
        return base + f" min={min_v:.6g}, max={max_v:.6g}, mean={mean_v:.6g}"

    return base


def _summarize(
    obj: Any,
    name: str,
    indent: int,
    max_items: int,
    sample_size: int,
    depth: int = 0,
    max_depth: int = 6,
) -> None:
    prefix = "  " * indent

    if depth > max_depth:
        print(f"{prefix}{name}: <max depth reached>")
        return

    if isinstance(obj, torch.Tensor):
        print(f"{prefix}{name}: {_tensor_stats(obj, sample_size)}")
        return

    if isinstance(obj, dict):
        print(f"{prefix}{name}: dict(len={len(obj)})")
        keys = list(obj.keys())
        shown = keys[:max_items]
        for k in shown:
            _summarize(
                obj[k],
                name=f"[{repr(k)}]",
                indent=indent + 1,
                max_items=max_items,
                sample_size=sample_size,
                depth=depth + 1,
                max_depth=max_depth,
            )
        if len(keys) > max_items:
            print(f"{prefix}  ... {len(keys) - max_items} more keys")
        return

    if isinstance(obj, (list, tuple)):
        kind = "list" if isinstance(obj, list) else "tuple"
        print(f"{prefix}{name}: {kind}(len={len(obj)})")
        shown = obj[:max_items]
        for idx, item in enumerate(shown):
            _summarize(
                item,
                name=f"[{idx}]",
                indent=indent + 1,
                max_items=max_items,
                sample_size=sample_size,
                depth=depth + 1,
                max_depth=max_depth,
            )
        if len(obj) > max_items:
            print(f"{prefix}  ... {len(obj) - max_items} more items")
        return

    print(f"{prefix}{name}: {type(obj).__name__} = {repr(obj)[:200]}")


def _print_tensor_samples(name: str, t: torch.Tensor, num_samples: int, seq_preview: int) -> None:
    if t.numel() == 0:
        print(f"\nSamples for {name}: tensor is empty")
        return

    count = min(num_samples, int(t.shape[0])) if t.ndim > 0 else 1
    print(f"\nSamples for {name} (showing {count}):")

    if t.ndim == 0:
        print(f"  {name}[0] = {t.item()}")
        return

    for i in range(count):
        sample = t[i]
        if sample.ndim == 0:
            print(f"  {name}[{i}] = {sample.item()}")
        elif sample.ndim == 1:
            preview = sample[:seq_preview].tolist()
            print(f"  {name}[{i}] first {len(preview)} values: {preview}")
        elif sample.ndim == 2:
            # Typical for your X: (channels, seq_len)
            print(f"  {name}[{i}] shape={tuple(sample.shape)}")
            for ch in range(min(3, sample.shape[0])):
                row = sample[ch, :seq_preview].tolist()
                print(f"    ch{ch} first {len(row)}: {row}")
        else:
            print(f"  {name}[{i}] shape={tuple(sample.shape)} (preview skipped for ndim>2)")


def _print_samples(obj: Any, num_samples: int, seq_preview: int) -> None:
    # Handle common dataset structure: {"X": tensor, "y": tensor}
    if isinstance(obj, dict):
        if "X" in obj and isinstance(obj["X"], torch.Tensor):
            _print_tensor_samples("X", obj["X"], num_samples, seq_preview)
        if "y" in obj and isinstance(obj["y"], torch.Tensor):
            _print_tensor_samples("y", obj["y"], num_samples, seq_preview)
        if "X" in obj and "y" in obj and isinstance(obj["X"], torch.Tensor) and isinstance(obj["y"], torch.Tensor):
            pairs = min(num_samples, int(obj["X"].shape[0]), int(obj["y"].shape[0]))
            print(f"\nSample pairs (X index -> y label), showing {pairs}:")
            for i in range(pairs):
                print(f"  idx {i} -> y={int(obj['y'][i].item())}")
        return

    if isinstance(obj, torch.Tensor):
        _print_tensor_samples("root", obj, num_samples, seq_preview)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect structure of a .pt file")
    parser.add_argument(
        "--file",
        required=True,
        type=Path,
        help="Path to the .pt file to inspect",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=10,
        help="Max dict keys/list items shown at each level",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=10000,
        help="Number of values to sample for min/max/mean",
    )
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="Use torch.load(..., weights_only=True) when supported",
    )
    parser.add_argument(
        "--show-samples",
        action="store_true",
        help="Print actual sample values for top-level tensor fields",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="How many samples to print when --show-samples is used",
    )
    parser.add_argument(
        "--seq-preview",
        type=int,
        default=10,
        help="For sequence tensors, print first N values per row/channel",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    file_path = args.file

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if file_path.suffix.lower() != ".pt":
        print(f"Warning: file extension is '{file_path.suffix}', expected '.pt'")

    print(f"Loading: {file_path.resolve()}")

    load_kwargs = {"map_location": "cpu"}
    if args.weights_only:
        load_kwargs["weights_only"] = True

    try:
        obj = torch.load(file_path, **load_kwargs)
    except TypeError:
        # Older torch versions may not support weights_only.
        if "weights_only" in load_kwargs:
            print("weights_only is not supported in this torch version, retrying.")
            load_kwargs.pop("weights_only", None)
            obj = torch.load(file_path, **load_kwargs)
        else:
            raise

    print("Top-level summary:")
    _summarize(
        obj,
        name="root",
        indent=0,
        max_items=max(1, args.max_items),
        sample_size=max(1, args.sample),
    )
    if args.show_samples:
        _print_samples(
            obj,
            num_samples=max(1, args.num_samples),
            seq_preview=max(1, args.seq_preview),
        )


if __name__ == "__main__":
    main()
