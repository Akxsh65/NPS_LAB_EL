# QUIC Metadata Privacy — Presentation Demo

Interactive web demo: real CESNET-QUIC22 test flows, Phase 3 obfuscation, and live Transformer / CNN-BiLSTM inference.

**No virtual environment is required** — install dependencies once with `pip` and run two Python processes.

---

## Quick start (teammates)

### 1. Prerequisites

| Requirement | Notes |
|-------------|--------|
| **Python 3.10 or 3.11** | Check: `python --version` |
| **Git clone** | `git clone <repo-url>` then `cd NPS_LAB_EL` |
| **~30 MB disk** | Model checkpoints + test tensors (see below) |
| **Ports free** | `8765` (API), `8080` (static site) |

### 2. Install Python dependencies

From the **repo root** (`NPS_LAB_EL/`):

```powershell
pip install -r presentation/requirements.txt
```

This installs: `torch`, `numpy`, `joblib`, `scikit-learn`, `tqdm`. CPU-only PyTorch is sufficient.

> **Windows:** Use the same `python` you will run the servers with. No venv needed unless you prefer one.

### 3. Verify required files are present

```powershell
python presentation/scripts/check_artifacts.py
```

All lines should show `OK`. If anything is `MISS`, those files were not pulled from Git — see [Artifacts to commit](#artifacts-to-commit).

### 4. Run the demo (two terminals)

**Terminal 1 — inference API**

```powershell
cd path\to\NPS_LAB_EL
python presentation/api_server.py
```

Wait until you see:

```
Starting inference API on http://127.0.0.1:8765
  POST /batch_predict
Loaded transformer from ...
Loaded bilstm from ...
```

**Terminal 2 — static website**

```powershell
cd path\to\NPS_LAB_EL
python -m http.server 8080
```

**Browser:** [http://localhost:8080/presentation/](http://localhost:8080/presentation/)

Hard refresh if needed: `Ctrl+F5`.

The top badge should read **“Live model API v2 (cpu)”**.

### 5. Smoke test (optional)

With both servers running:

```powershell
python presentation/scripts/smoke_test.py
```

---

## One-command launcher (Windows)

```powershell
powershell -File presentation/scripts/run_demo.ps1
```

Opens the API in a separate window and runs the static server in the current window.

---

## Artifacts to commit

These files **must be in the repository** so teammates can run without re-training or re-downloading CESNET data:

| File | Purpose | ~Size |
|------|---------|-------|
| `phase1/artifacts/test_tensors.pt` | 49,305 test flows | 18 MB |
| `phase1/artifacts/label_encoder.pkl` | App ID labels | 1 KB |
| `phase1/artifacts/ipt_scaler.pkl` | IPT scaler for obfuscator | 1 KB |
| `phase2/artifacts/transformer_production.pt` | Transformer model | 4 MB |
| `phase2/artifacts/transformer_masked_config.json` | Model config | 1 KB |
| `phase2/artifacts/cnn_bilstm_best.pt` | BiLSTM model | 3 MB |
| `presentation/js/demo_flows.js` | 12 curated packet examples | — |
| `presentation/js/test_flow_catalog.js` | All test flow indices | ~1.6 MB |

`.gitignore` is configured to **allow these paths** while still ignoring large training outputs (Phase 3 obfuscated tensors, adaptive runs, raw dataset).

After changing `.gitignore`, force-add if files were previously ignored:

```powershell
git add -f phase1/artifacts/test_tensors.pt phase1/artifacts/label_encoder.pkl phase1/artifacts/ipt_scaler.pkl
git add -f phase2/artifacts/transformer_production.pt phase2/artifacts/transformer_masked_config.json phase2/artifacts/cnn_bilstm_best.pt
git status
```

> **GitHub limit:** single files must be under 100 MB. `test_tensors.pt` (~18 MB) is fine.

### If `label_encoder.pkl` / `ipt_scaler.pkl` are missing

```powershell
python presentation/scripts/bootstrap_artifacts.py
```

Requires `phase2/artifacts/transformer_production.pt` and `phase4/results/baseline_per_class.csv`.

---

## What the demo does (live vs pre-loaded)

| Feature | Source |
|---------|--------|
| Packet sizes / timelines | Real `test_tensors.pt` (via API `/flow_vis` or exported `demo_flows.js`) |
| Obfuscation | `phase3/obfuscator.py` (Python, same as research pipeline) |
| Classifier | Live PyTorch forward pass (`/predict`) |
| Verify accuracy (200 flows) | Real batch inference (`/batch_predict`) |
| Charts & defense metrics | Phase 4 numbers baked into `app.js` (no CSV fetch at runtime) |

Without the API, the site still loads but the classifier uses a **random demo fallback** — always run `api_server.py` for real inference.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Badge says “Demo loaded” | Start `python presentation/api_server.py` |
| `batch_predict` / `flow_vis` 404 | Old API process on port 8765 — kill it and restart |
| Port 8765 in use | `netstat -ano \| findstr ":8765"` then `taskkill /PID <pid> /F` |
| `ModuleNotFoundError: torch` | `pip install -r presentation/requirements.txt` |
| `Missing test_tensors.pt` | Pull latest Git or copy artifacts from team |
| `FileNotFoundError: label_encoder.pkl` | Run `bootstrap_artifacts.py` |
| Page loads but no live inference | Hard refresh `Ctrl+F5`; check API terminal for errors |

---

## Regenerate demo data (optional)

```powershell
python presentation/scripts/export_demo_flows.py --count 12 --annotate
python presentation/scripts/export_flow_catalog.py
```

---

## Project layout

```
presentation/
├── requirements.txt          # pip install -r this
├── api_server.py             # inference API :8765
├── index.html
├── js/app.js                 # main UI
├── js/demo_flows.js          # curated flows (generated)
├── js/test_flow_catalog.js   # 49k flow catalog (generated)
├── scripts/
│   ├── check_artifacts.py    # verify files before run
│   ├── smoke_test.py         # end-to-end test
│   ├── bootstrap_artifacts.py
│   ├── export_demo_flows.py
│   └── run_demo.ps1
└── README.md                 # this file
```

---

## Full research pipeline

To re-run Phases 1–4 (not required for the demo), see `phase1/README.md`, `phase2/README.md`, `phase3/README.md`, and `phase4/README.md`. Those phases use separate `requirements.txt` files and much larger artifacts.
