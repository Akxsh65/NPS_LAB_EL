# Compile the Phase 5 paper in TeXstudio (local)

Same workflow as the **Major Project Report** — no Overleaf required.

## One-time setup

1. Open **TeXstudio**.
2. **File → Open** → `phase5/manuscript/main.tex`
3. Right-click `main.tex` → **Set as Master Document**.

MiKTeX auto-installs missing packages (`IEEEtran`, `subcaption`, etc.) on first compile.

## Build

| Key | Action |
|-----|--------|
| **F6** | Build & View (PdfLaTeX) |
| **F8** | BibTeX |
| **F7** | View PDF only |

**Full cycle** (citations + references):

1. **F6** → 2. **F8** → 3. **F6** → 4. **F6**

Output: `phase5/manuscript/main.pdf`

## Where to edit (content + diagrams together)

| What | File |
|------|------|
| Title, authors, packages | `main.tex` |
| Abstract | `sections/abstract.tex` |
| Each section | `sections/*.tex` |
| **TikZ diagrams (edit inline)** | `figures/tikz/pipeline.tex`, `obfuscation.tex`, `tensor.tex` |
| **Result plots (PNG)** | `figures/*.png` |
| Bibliography | `references.bib` |

You can open any `sections/*.tex` file directly — each has `% !TeX root = ../main.tex` so **F6** still builds the full paper.

## Adding a figure

### PNG from Phase 4

1. Copy the PNG into `figures/` (or re-run `phase4/plot_publication.py` and copy from `phase4/results/`).
2. In the relevant section file:

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{my_plot_name}
  \caption{Your caption.}
  \label{fig:myplot}
\end{figure}
```

Omit the `.png` extension — `\graphicspath{{figures/}}` in `main.tex` handles the path.

### TikZ diagram (recommended for attack/defense schematics)

1. Create or edit `figures/tikz/my_diagram.tex` (wrap content in a `figure` environment).
2. In a section: `\input{figures/tikz/my_diagram}`

Existing TikZ files:

| File | Shows |
|------|--------|
| `figures/tikz/pipeline.tex` | Phases 1–4 workflow |
| `figures/tikz/tensor.tex` | $(3,30)$ input tensor |
| `figures/tikz/obfuscation.tex` | Jitter / linear-128 / MTU schematic |

## PowerShell build (no TeXstudio)

```powershell
cd phase5\scripts
.\build.ps1
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| PDF not updating | Close PDF tab → **F6** again |
| `IEEEtran.cls not found` | Allow MiKTeX to install `ieeetran` |
| Undefined citations | Run **F8** then **F6** twice |
| Figure not found | PNG must be in `figures/`; no path prefix in `\includegraphics` |
| Wrong file compiled | Master document must be `main.tex` |
| Editing a section file | Press **F6** — root comment points to `main.tex` |
