"""Heatmap brier_reliability.

8 methods (rows) x 4 datasets (cols). Cells annotated with Brier value.
Lower Brier = better, mapped to a lighter color via a sequential map.
Ours row highlighted with a brick-red label and a stronger border.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/data/jehc223/home/tmp/matplotlib")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

from _paper_palette import COLOR, apply_style


ROOT = Path("/data/jehc223/EMNLP3")
OUT_DIR = ROOT / "paper" / "figures"
ANALYSIS_DIR = ROOT / "paper" / "analysis"


DATASETS = ["HateMM", "MHClip_EN", "MHClip_ZH", "ImpliHateVid"]
DATASET_LABELS = {
    "HateMM": "HateMM",
    "MHClip_EN": "MH-EN",
    "MHClip_ZH": "MH-ZH",
    "ImpliHateVid": "ImpliHV",
}

METHODS = ["BERT", "ViViT", "MFCC", "Pro-Cap", "MHCL", "HateMM", "CMFusion", "Ours"]

SUPERVISED_BRIER = {
    "HateMM": {
        "BERT": 0.3195, "ViViT": 0.3891, "MFCC": 0.3435, "Pro-Cap": 0.2967,
        "MHCL": 0.3020, "HateMM": 0.2759, "CMFusion": 0.2888,
    },
    "ImpliHateVid": {
        "BERT": 0.2143, "ViViT": 0.3421, "MFCC": 0.3422, "Pro-Cap": 0.1978,
        "MHCL": 0.2273, "HateMM": 0.2157, "CMFusion": 0.2123,
    },
    "MHClip_EN": {
        "BERT": 0.3687, "ViViT": 0.3968, "MFCC": 0.4854, "Pro-Cap": 0.3345,
        "MHCL": 0.4567, "HateMM": 0.3555, "CMFusion": 0.3434,
    },
    "MHClip_ZH": {
        "BERT": 0.3559, "ViViT": 0.3950, "MFCC": 0.4201, "Pro-Cap": 0.3230,
        "MHCL": 0.4027, "HateMM": 0.3363, "CMFusion": 0.3228,
    },
}


def load_ours_brier() -> dict[str, float]:
    path = ANALYSIS_DIR / "posterior_calibration.csv"
    ours = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["stage"] == "Full" and row["dataset"] in DATASETS:
                ours[row["dataset"]] = float(row["brier"])
    missing = [ds for ds in DATASETS if ds not in ours]
    if missing:
        raise RuntimeError(f"Missing ours Brier for {missing}")
    return ours


def write_csv(values: dict[str, dict[str, float]]) -> None:
    path = OUT_DIR / "brier_reliability.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "method", "brier"])
        writer.writeheader()
        for ds in DATASETS:
            for method in METHODS:
                writer.writerow({"dataset": ds, "method": method, "brier": values[ds][method]})


def plot(values: dict[str, dict[str, float]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()

    grid = np.zeros((len(METHODS), len(DATASETS)))
    for j, ds in enumerate(DATASETS):
        for i, method in enumerate(METHODS):
            grid[i, j] = values[ds][method]

    cmap = LinearSegmentedColormap.from_list(
        "brier_seq",
        ["#F5F1EA", "#D7DCE7", "#9DAEC9", COLOR["primary"]],
        N=256,
    )
    vmin = float(grid.min()) * 0.85
    vmax = float(grid.max()) * 1.02
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(3.45, 2.12), dpi=300)
    im = ax.imshow(grid, aspect="auto", cmap=cmap, norm=norm)

    n_methods, n_datasets = grid.shape
    for i in range(n_methods):
        for j in range(n_datasets):
            v = grid[i, j]
            shade = norm(v)
            text_color = COLOR["text"] if shade < 0.55 else "white"
            weight = "bold" if METHODS[i] == "Ours" else "normal"
            ax.text(
                j, i, f"{v:.3f}",
                ha="center", va="center",
                fontsize=7.3, fontweight=weight,
                color=text_color,
            )

    ax.set_xticks(np.arange(n_datasets))
    ax.set_xticklabels(
        [DATASET_LABELS[d] for d in DATASETS], fontsize=8.1, fontweight="bold",
    )
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    ax.set_yticks(np.arange(n_methods))
    ax.set_yticklabels(METHODS, fontsize=8.1)

    ours_idx = METHODS.index("Ours")
    for label in ax.get_yticklabels():
        if label.get_text() == "Ours":
            label.set_color(COLOR["ours"])
            label.set_fontweight("bold")

    rect = mpatches.Rectangle(
        (-0.5, ours_idx - 0.5), n_datasets, 1.0,
        fill=False, edgecolor=COLOR["ours"], linewidth=1.3, zorder=4,
    )
    ax.add_patch(rect)

    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xticks(np.arange(n_datasets + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_methods + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.03)
    cbar.set_label("Brier score (lower is better)", fontsize=7.3, color=COLOR["text"])
    cbar.ax.tick_params(labelsize=6.6, length=0, pad=2)
    cbar.outline.set_visible(False)

    fig.tight_layout(pad=0.18)
    fig.savefig(OUT_DIR / "brier_reliability.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "brier_reliability.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ours = load_ours_brier()
    values = {}
    for ds in DATASETS:
        values[ds] = dict(SUPERVISED_BRIER[ds])
        values[ds]["Ours"] = ours[ds]
    write_csv(values)
    plot(values)
    print(OUT_DIR / "brier_reliability.pdf")


if __name__ == "__main__":
    main()
