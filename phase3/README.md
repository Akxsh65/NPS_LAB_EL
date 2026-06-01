# Phase 3 — Obfuscation Defense

Applies size padding and IPT jitter on **denormalized** PPI, then re-normalizes with Phase 1 `ipt_scaler.pkl`.

## Run

```bash
cd phase3
pip install -r requirements.txt
python generate_obfuscated.py
```

## Outputs (`phase3/artifacts/`)

| File | Defense |
|------|---------|
| `obfuscated_linear128.pt` | 128-byte linear padding |
| `obfuscated_mtu.pt` | Pad all packets to MTU |
| `obfuscated_jitter_*.pt` | Laplace IPT jitter only |
| `obfuscated_linear128_jitter_medium.pt` | Combined |
| `obfuscation_manifest.json` | Mean bandwidth/latency overhead |

## API

```python
from obfuscator import obfuscate
out, meta = obfuscate(flow_tensor, padding_type="linear128", jitter_scale=5.0)
```
