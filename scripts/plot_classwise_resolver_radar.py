#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path("paper/figures")
OUT_PATH = OUT_DIR / "classwise_resolver_radar.pdf"

DATA = {
    "HateMM": {
        "color": "#2E5E8C",
        "values": [7.0, 9.3, 5.6, 3.9],
    },
    "MHClip-EN": {
        "color": "#B65A3A",
        "values": [-2.7, 18.4, 4.7, -3.6],
    },
    "MHClip-ZH": {
        "color": "#3D7D4D",
        "values": [4.5, 15.6, 6.4, 0.0],
    },
    "ImpliHateVid": {
        "color": "#7A5EA7",
        "values": [6.8, -6.5, -3.2, 8.5],
    },
}

AXES = [
    "Hateful\nPrecision",
    "Hateful\nRecall",
    "Normal\nPrecision",
    "Normal\nRecall",
]
MIN_VALUE = -8
MAX_VALUE = 20
RINGS = [-8, 0, 8, 16]


def to_radius(values: list[float]) -> np.ndarray:
    values_arr = np.asarray(values, dtype=float)
    return values_arr - MIN_VALUE


def draw_panel(ax: plt.Axes) -> None:
    angles = np.linspace(0, 2 * np.pi, len(AXES), endpoint=False)
    closed_angles = np.concatenate([angles, angles[:1]])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, MAX_VALUE - MIN_VALUE)
    ax.set_xticks(angles)
    ax.set_xticklabels(AXES, fontsize=8.5, linespacing=1.15)
    ax.tick_params(axis="x", pad=9)

    ax.set_yticks([r - MIN_VALUE for r in RINGS])
    ax.set_yticklabels([f"{r:+d}" for r in RINGS], fontsize=7, color="#5F6770")
    ax.set_rlabel_position(24)

    ax.grid(color="#D5DAE0", linewidth=0.65)
    ax.spines["polar"].set_color("#AEB7C2")
    ax.spines["polar"].set_linewidth(0.8)
    ax.set_title("")

    zero_ring = np.full_like(closed_angles, -MIN_VALUE, dtype=float)
    ax.plot(closed_angles, zero_ring, color="#7B8490", linewidth=0.9, linestyle=(0, (3, 2)), zorder=1)

    for name, spec in DATA.items():
        values = spec["values"]
        radii = np.concatenate([to_radius(values), to_radius(values[:1])])
        ax.plot(
            closed_angles,
            radii,
            color=spec["color"],
            linewidth=1.7,
            marker="o",
            markersize=3.0,
            label=name,
            zorder=3,
        )
        ax.fill(closed_angles, radii, color=spec["color"], alpha=0.075, zorder=2)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )

    fig, ax = plt.subplots(figsize=(3.35, 3.35), subplot_kw={"projection": "polar"})
    draw_panel(ax)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
        fontsize=7.6,
        handlelength=1.6,
        columnspacing=1.15,
        labelspacing=0.35,
    )
    fig.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.31)
    fig.savefig(OUT_PATH, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
