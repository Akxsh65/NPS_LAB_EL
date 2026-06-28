"""Shared styling for Phase 4 publication plots (Tier A/B)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

NUM_CLASSES = 64
CHANCE_ACC = 1.0 / NUM_CLASSES
CHANCE_PCT = CHANCE_ACC * 100.0  # 1.5625%

RECOMMENDED_EXPERIMENT = "obfuscated_jitter_low"
RECOMMENDED_COLOR = "#DAA520"
RECOMMENDED_MARKER = "*"


def bar_colors(experiments: list[str], default: str = "mediumpurple") -> list[str]:
    return [RECOMMENDED_COLOR if e == RECOMMENDED_EXPERIMENT else default for e in experiments]


def add_chance_line(ax, y_in_percent: bool = True, label: str | None = None) -> None:
    """Horizontal reference at random-guess accuracy (1/64)."""
    y = CHANCE_PCT if y_in_percent else CHANCE_ACC
    lbl = label or f"Chance accuracy ({CHANCE_PCT:.2f}%)"
    ax.axhline(y, color="gray", ls=":", lw=1.2, label=lbl, zorder=1)


def highlight_recommended_bar(ax, experiment_names: list[str], heights: list[float]) -> None:
    """Gold star above the recommended operating point bar."""
    for i, (exp, h) in enumerate(zip(experiment_names, heights)):
        if exp == RECOMMENDED_EXPERIMENT:
            ax.plot(
                i,
                h,
                marker=RECOMMENDED_MARKER,
                markersize=16,
                color="darkgoldenrod",
                markeredgecolor="black",
                markeredgewidth=0.4,
                zorder=10,
            )


def highlight_recommended_scatter(
    ax,
    x: float,
    y: float,
    label: str = "jitter_low (recommended)",
) -> None:
    ax.scatter(
        [x],
        [y],
        s=280,
        marker=RECOMMENDED_MARKER,
        c=RECOMMENDED_COLOR,
        edgecolors="black",
        linewidths=0.5,
        label=label,
        zorder=10,
    )
