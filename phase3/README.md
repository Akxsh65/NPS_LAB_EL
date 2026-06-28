# Phase 3 — Obfuscation Defense

Applies **deterministic** size padding and **seeded** one-sided Laplace IPT jitter on the **held-out test set** only. Operates in denormalized (raw ms / bytes) space, then re-normalizes with Phase 1 `ipt_scaler.pkl` (never refit).

## Methodology (research protocol)

| Property | Implementation |
|----------|----------------|
| **Active packets** | `~(DIR≈0 ∧ SIZE≈0)` per Phase 1 padding contract |
| **DIR channel** | Never modified |
| **Padding slots** | Never padded or jittered |
| **IPT index 0** | Never jittered (CESNET convention) |
| **Jitter** | Laplace(0, b), negative samples clipped to 0 (causal delays only) |
| **Reproducibility** | `RNG(seed + flow_index)` per flow |
| **Labels** | Copied unchanged from clean `test_tensors.pt` |
| **Audit** | Manifest v2: scaler SHA256, source test hash, per-file tensor hash |

## Prerequisites

`ipt_scaler.pkl` must live in **`phase1/artifacts/`** (same folder as `test_tensors.pt`).
Do **not** copy it to `phase3/artifacts/` — Phase 3 resolves the absolute path automatically.

```bash
python check_prerequisites.py
```

## Run

```bash
cd phase3
pip install -r requirements.txt

# Generate all obfuscated test sets + validate
python generate_obfuscated.py --validate

# Or validate existing artifacts only
python validate_obfuscation.py
```

## Outputs (`phase3/artifacts/`)

| File | Description |
|------|-------------|
| `obfuscated_*.pt` | `X` (obfuscated), `y` (unchanged), `meta` (provenance) |
| `obfuscated_*.meta.json` | Per-run overhead statistics |
| `obfuscation_manifest.json` | v2 manifest (all experiments + provenance) |
| `validation_report.json` | Go/no-go checks after `--validate` |

## Experiment grid

| Output | Padding | Jitter |
|--------|---------|--------|
| `obfuscated_linear128` | 128-byte blocks | none |
| `obfuscated_mtu` | MTU (1500 B) | none |
| `obfuscated_jitter_low/medium/high` | none | 1 / 5 / 20 ms scale |
| `obfuscated_linear128_jitter_medium` | linear128 | 5 ms |
| `obfuscated_mtu_jitter_medium` | MTU | 5 ms |

## API

```python
from obfuscator import obfuscate, obfuscate_batch, ipt_scaler_fingerprint

out, meta = obfuscate(flow, padding_type="linear128", jitter_scale=5.0, seed=42, flow_index=0)
X_obf, agg = obfuscate_batch(X, padding_type="mtu", jitter_scale=0.0, seed=42)
```

## Phase 4 baseline

Report privacy drop vs **clean test** accuracy (~77.8% in your run), not validation accuracy.

```bash
cd ../phase4
python run_experiments.py --checkpoint ../phase2/artifacts/transformer_production.pt
```

Ensure Phase 4 evaluation loads `d_model=160` from the training config (use `eval_clean_test.py` pattern or matching `transformer_masked_config.json`).
