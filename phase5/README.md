# Phase 5 — Manuscript Preparation

IEEE-style draft manuscript for the QUIC metadata obfuscation study. All figures and
tables are bundled under `manuscript/`; statistical summaries live in `data/`.

## Target venues

- **Primary:** IEEE/ACM Transactions on Networking (ToN)
- **Alternate:** IEEE Transactions on Information Forensics and Security (TIFS)

## Directory layout

```
phase5/
├── README.md                 # This file
├── FIGURES.md                # Figure → file mapping for the paper
├── REPRODUCIBILITY.md        # Artifact paths and re-run commands
├── data/                     # Tier C CSVs/JSON used in prose (not duplicated in git figures)
└── manuscript/
    ├── main.tex              # Master file — open this in TeXstudio
    ├── COMPILE.md            # TeXstudio build guide
    ├── references.bib        # Bibliography
    ├── sections/             # One file per section (edit content here)
    ├── figures/              # PNG plots from phase4/results/
    │   └── tikz/             # TikZ diagrams (pipeline, etc.)
    └── tables/               # LaTeX tabular fragments (optional)
```

## Compile locally (TeXstudio — recommended)

Open `manuscript/main.tex` in **TeXstudio**, set as master document, then **F6 → F8 → F6 → F6**.
See **`manuscript/COMPILE.md`** for full steps, figure paths, and troubleshooting.

Requires MiKTeX/TeX Live with `IEEEtran`, `graphicx`, `booktabs`, `subcaption`,
`hyperref`, `cite`, and `tikz`.

```powershell
cd phase5\scripts
.\build.ps1
```

Output: `manuscript/main.pdf`

## Compile on Overleaf (optional)

1. New project → **IEEE Conference Template** (official IEEEtran).
2. Upload `main.tex`, `references.bib`, `sections/`, and `figures/`.
3. Set `main.tex` as the root file and recompile.

## Key claims (aligned with measured results)

1. **Systematic privacy–cost study** on CESNET-QUIC22 backbone QUIC flows with temporal
   holdout, frozen attack models, and manifest-derived bandwidth/latency overhead.
2. **Defense taxonomy:** timing jitter (zero bandwidth cost), linear padding (+17% BW),
   MTU padding (+274% BW); jitter tiers dominate the latency–accuracy frontier.
3. **Deployable operating point:** `jitter_low` — macro F1 72.7% (76.8% acc), only
   11 ms mean latency overhead, statistically significant but small drop vs. baseline
   (Δacc 0.93 pp, McNemar *p* ≈ 5.8×10⁻²²).
4. **Architecture robustness:** masked Transformer outperforms CNN-BiLSTM on clean and
   jitter-defended traffic; BiLSTM collapses more under strong jitter (54.7% vs 36.0%
   acc at `jitter_high`). MTU obfuscation defeats both architectures (~2–3% acc).

> **Note:** On the bandwidth axis, all jitter tiers share 0% overhead, so the formal
> Pareto frontier runs `jitter_high` → `linear128_jitter_medium`. We recommend
> `jitter_low` as the practical operating point and use latency–accuracy plots for
> jitter comparison (see Discussion).

## Before submission checklist

- [ ] Replace `\author{...}` placeholders with real names and affiliations
- [ ] Expand Related Work with venue-specific citations
- [ ] Map numeric class IDs to application names via `label_encoder.pkl` for Discussion
- [ ] Add ethics / data-use statement (CESNET-QUIC22 Zenodo terms)
- [ ] Optional: generate architecture-comparison bar figure from `data/architecture_comparison.csv`
- [ ] Proofread against final `phase4/results/accuracy_results.csv` after any re-run

## Source pipeline

| Phase | Role |
|-------|------|
| Phase 1 | Tensors, temporal split W-2022-44 / W-2022-45 |
| Phase 2 | Transformer (d=160) + CNN-BiLSTM attackers |
| Phase 3 | Deterministic obfuscators + overhead manifest |
| Phase 4 | Evaluation, Pareto, Tier C statistics |

See `REPRODUCIBILITY.md` for exact checkpoint paths and commands.
