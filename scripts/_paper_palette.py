"""Unified palette and matplotlib defaults for all paper figures.

Single-column figures in ACL/EMNLP are ~3.3 inch wide. To survive that
without text becoming illegible, every figure draws at its display
size (figsize close to column width) with fonts that already match
the final printed size.
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/data/jehc223/home/tmp/matplotlib")

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


COLOR = {
    "ours":      "#DB7B72",
    "primary":   "#7396C4",
    "secondary": "#8FBE9F",
    "gold":      "#DDBA7E",
    "plum":      "#AC94C4",
    "neutral":   "#C8CDD7",
    "neutral_d": "#A5ACBA",
    "grid":      "#ECEFF4",
    "axis":      "#A5ACBA",
    "text":      "#2A3340",
    "muted":     "#7A8290",
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "axes.labelcolor": COLOR["text"],
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.color": COLOR["text"],
            "ytick.color": COLOR["text"],
            "legend.fontsize": 7.5,
            "axes.edgecolor": COLOR["axis"],
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def style_axes(ax) -> None:
    ax.grid(axis="y", color=COLOR["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(COLOR["axis"])
    ax.tick_params(axis="both", length=0)


def fig_at(width_in: float, height_in: float, **kw):
    return plt.subplots(figsize=(width_in, height_in), dpi=300, **kw)
