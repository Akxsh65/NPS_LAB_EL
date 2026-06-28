# Project Presentation Frontend

Interactive demo site for the college presentation: problem statement, 4-phase pipeline,
**live attack/defense simulation**, defense cards, and results charts.

## Run locally

From the **repository root** (so figure paths resolve):

```powershell
Set-Location C:\Users\akash\Desktop\NPS_LAB_EL
python -m http.server 8080
```

Open: **http://localhost:8080/presentation/**

Or open `presentation/index.html` directly in the browser — the bundled `js/app.js` works without a server.

> Charts still need internet (Chart.js CDN). Simulation, defense cards, and figures work offline.

## What's included

| Section | Content |
|---------|---------|
| **Overview** | Hero + key stats (Transformer vs BiLSTM baseline accuracy) |
| **Problem** | Attack flow diagram (user → observer → classifier) |
| **Pipeline** | Phases 1–4 cards |
| **Simulation** | Pick app + defense; animated packet stream + timelines + mock classifier |
| **Defenses** | All 8 settings with mechanism text + metrics (click → simulator) |
| **Results** | Chart.js charts + highlight cards + full results table + takeaway cards |

## Simulation logic

Client-side JS mirrors `phase3/obfuscator.py`:

- **Jitter:** one-sided Laplace on IPT (packets 1–29; index 0 fixed)
- **Linear128:** ceil size to 128 B blocks
- **MTU:** pad active packets to 1500 B

Population accuracy/F1 and overhead come from `phase4/results/accuracy_results.csv` (measured on 49,305 test flows).

**Live packet stream:** dual-lane animation replays the selected flow — clean vs obfuscated. Packets launch from the emitter; jitter shows as extra spacing (+ms badges and bridge lines); padding shows bars growing (+B badges). Auto-plays when you change defense or flow; use **Replay stream** to run again.

## Data honesty

| Content | Source |
|---------|--------|
| Charts, tables, defense cards | `phase4/results/accuracy_results.csv` (measured) |
| Architecture chart | `phase4/architecture_comparison.csv` |
| Channel ablation | `phase4/channel_ablation.csv` |
| Packet timelines & fingerprint dropdown | Real test flows from `phase1/artifacts/test_tensors.pt` (via `presentation/scripts/export_demo_flows.py`) |
| Classifier panel in simulator | **Simulated** demo; sampled from measured test-set accuracy |

Obfuscation transforms in the demo mirror `phase3/obfuscator.py` (Laplace jitter 1/5/20 ms, linear-128, MTU 1500 B).

### Regenerate demo flows

After updating Phase 1 artifacts:

```powershell
python presentation/scripts/export_demo_flows.py
```

This writes `presentation/data/demo_flows.json` and `presentation/js/demo_flows.js`.

## Files

```
presentation/
├── index.html
├── css/styles.css
├── js/app.js           # Bundled app (simulation + charts)
├── js/demo_flows.js    # Auto-generated real test flows (do not edit)
├── data/demo_flows.json
├── scripts/export_demo_flows.py
└── README.md
```
