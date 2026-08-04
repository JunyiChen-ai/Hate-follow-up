"""Band-analysis subfigures.

Distinct chart types for the two diagnostics so they no longer look the
same at a glance:

  band_error_concentration - paired horizontal bars.
      Story: the boundary region is compact, yet covers a larger share
      of mapper errors than expected from its size.

  band_difficulty_separation - paired horizontal bars.
      Story: in-band videos are hard, out-band videos are easy (a *gap*).
      Annotated as a signed delta "+19.6".
      Palette: plum -> gold (deliberately different from band_error so
      the reader does not mistake the two figures for one another).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/data/jehc223/home/tmp/matplotlib")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt

from _paper_palette import COLOR, apply_style


ROOT = Path("/data/jehc223/EMNLP2")
OUT_DIR = ROOT / "paper" / "figures"

DATASETS = ["HateMM", "MHClip-EN", "MHClip-ZH", "ImpliHateVid"]


BAND_ERROR = {
    "HateMM":       (46.0, 71.0),
    "MHClip-EN":    (57.0, 77.0),
    "MHClip-ZH":    (52.0, 60.0),
    "ImpliHateVid": (38.0, 70.7),
}

BAND_DIFFICULTY = {
    "HateMM":       (70.0, 89.6),
    "MHClip-EN":    (67.0, 86.0),
    "MHClip-ZH":    (75.6, 82.0),
    "ImpliHateVid": (65.0, 90.5),
}


def draw_band_error(ax):
    bar_h = 0.34
    size_color = COLOR["neutral_d"]
    error_color = COLOR["ours"]
    y_centers = list(range(len(DATASETS)))[::-1]

    for i, ds in zip(y_centers, DATASETS):
        region_size, errors_covered = BAND_ERROR[ds]
        ax.barh(i + bar_h / 2, region_size, bar_h,
                color=size_color, edgecolor="white", linewidth=0.5, zorder=3)
        ax.barh(i - bar_h / 2, errors_covered, bar_h,
                color=error_color, edgecolor="white", linewidth=0.5, zorder=3)

        delta = errors_covered - region_size
        sign = "+" if delta >= 0 else "−"
        ax.text(
            80.5, i, f"{sign}{abs(delta):.0f}",
            ha="left", va="center",
            fontsize=7.2, color=COLOR["text"], fontweight="bold",
        )

    ax.set_yticks(y_centers)
    ax.set_yticklabels(DATASETS, fontsize=7.4)
    ax.set_xlabel("Proportion (%)", fontsize=7.5)
    ax.tick_params(axis="x", labelsize=6.8, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=COLOR["grid"], linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(COLOR["axis"])

    ax.set_xlim(0, 90)
    ax.set_xticks([0, 30, 60, 90])

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=size_color, edgecolor="white", linewidth=0.4,
                      label="Region size"),
        plt.Rectangle((0, 0), 1, 1, facecolor=error_color, edgecolor="white", linewidth=0.4,
                      label="Error coverage"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center", bbox_to_anchor=(0.5, 1.03),
        ncol=1, frameon=False,
        handlelength=0.9, handletextpad=0.3, columnspacing=0.7,
        labelspacing=0.2, borderaxespad=0.0, fontsize=6.8,
    )


def plot_band_error():
    apply_style()
    fig, ax = plt.subplots(figsize=(1.85, 1.65), dpi=300)
    draw_band_error(ax)
    fig.tight_layout(pad=0.18)
    fig.savefig(OUT_DIR / "band_error_concentration.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "band_error_concentration.png", bbox_inches="tight")
    plt.close(fig)


def draw_band_difficulty(ax):
    bar_h = 0.36
    in_color = COLOR["plum"]
    out_color = COLOR["gold"]
    y_centers = list(range(len(DATASETS)))[::-1]

    for i, ds in zip(y_centers, DATASETS):
        a, b = BAND_DIFFICULTY[ds]  # inside region, outside region
        ax.barh(i + bar_h / 2, a, bar_h,
                color=in_color, edgecolor="white", linewidth=0.5, zorder=3)
        ax.barh(i - bar_h / 2, b, bar_h,
                color=out_color, edgecolor="white", linewidth=0.5, zorder=3)
        delta = b - a
        sign = "+" if delta >= 0 else "−"
        ax.text(
            93.5, i, f"{sign}{abs(delta):.1f}",
            ha="left", va="center",
            fontsize=7.6, color=COLOR["text"], fontweight="bold",
        )

    ax.set_yticks(y_centers)
    ax.set_yticklabels([])
    ax.set_xlabel("Accuracy (%)", fontsize=7.5)
    ax.tick_params(axis="x", labelsize=6.8, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=COLOR["grid"], linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(COLOR["axis"])

    ax.set_xlim(55, 105)
    ax.set_xticks([60, 75, 90])

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=in_color, edgecolor="white", linewidth=0.4,
                      label="Inside region"),
        plt.Rectangle((0, 0), 1, 1, facecolor=out_color, edgecolor="white", linewidth=0.4,
                      label="Outside region"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center", bbox_to_anchor=(0.5, 1.03),
        ncol=1, frameon=False,
        handlelength=0.9, handletextpad=0.3, columnspacing=0.7,
        labelspacing=0.2, borderaxespad=0.0, fontsize=6.8,
    )


def plot_band_difficulty():
    """Paired horizontal bars showing in-band vs out-band ACC with delta label."""
    apply_style()
    fig, ax = plt.subplots(figsize=(1.85, 1.65), dpi=300)
    draw_band_difficulty(ax)
    fig.tight_layout(pad=0.18)
    fig.savefig(OUT_DIR / "band_difficulty_separation.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "band_difficulty_separation.png", bbox_inches="tight")
    plt.close(fig)


def plot_band_analysis_combined():
    """Two-panel version for direct inclusion without LaTeX subfigures."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(3.7, 1.65), dpi=300)
    draw_band_error(axes[0])
    draw_band_difficulty(axes[1])
    fig.tight_layout(pad=0.18, w_pad=1.0)
    fig.savefig(OUT_DIR / "band_analysis_combined.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "band_analysis_combined.png", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_band_error()
    plot_band_difficulty()
    plot_band_analysis_combined()
    print(OUT_DIR / "band_error_concentration.pdf")
    print(OUT_DIR / "band_difficulty_separation.pdf")
    print(OUT_DIR / "band_analysis_combined.pdf")


if __name__ == "__main__":
    main()
