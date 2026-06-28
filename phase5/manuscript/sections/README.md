# Manuscript sections

Edit these files in TeXstudio — `main.tex` pulls them in with `\input`.

| File | Content |
|------|---------|
| `abstract.tex` | Abstract + IEEE keywords |
| `introduction.tex` | Section I |
| `related.tex` | Related work |
| `threat_model.tex` | Threat model |
| `methodology.tex` | Methods (+ `\input{figures/tikz/pipeline}`) |
| `results.tex` | Results, tables, PNG figures |
| `discussion.tex` | Discussion |
| `conclusion.tex` | Conclusion |
| `appendix_repro.tex` | Reproducibility appendix |

TikZ diagrams live in `../figures/tikz/`. PNG plots live in `../figures/`.

See `../COMPILE.md` for TeXstudio build steps.
