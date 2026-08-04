"""Plot end-to-end efficiency analysis for the paper."""

import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/data/jehc223/home/tmp/matplotlib")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np

from _paper_palette import COLOR, apply_style


ROOT = Path("/data/jehc223/EMNLP3")
OUT_DIR = ROOT / "paper" / "figures"
CSV_PATH = OUT_DIR / "efficiency_setup_tradeoff.csv"
PDF_PATH = OUT_DIR / "efficiency_setup_tradeoff.pdf"
PNG_PATH = OUT_DIR / "efficiency_setup_tradeoff.png"


ROWS = [
    {"method": "LoReHM",   "total_param_b": 43.9,   "total_sec": 15.8},
    {"method": "Ours",     "total_param_b": 21.63,  "total_sec": 5.67},
    {"method": "ALARM",    "total_param_b": 367.0,  "total_sec": 669.0},
    {"method": "MARS",     "total_param_b": 128.00, "total_sec": 30.45},
    {"method": "MoD-HATE", "total_param_b": 284.94, "total_sec": 5.67},
]


METHOD_COLOR = {
    "Ours":     COLOR["ours"],
    "LoReHM":   COLOR["primary"],
    "MARS":     COLOR["secondary"],
    "ALARM":    COLOR["plum"],
    "MoD-HATE": COLOR["gold"],
}

LABEL_OFFSET = {
    "LoReHM":   (7, 6),
    "Ours":     (7, 0),
    "ALARM":    (7, 0),
    "MARS":     (7, 6),
    "MoD-HATE": (7, 0),
}


def write_csv(rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "total_param_b", "total_sec"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in ("method", "total_param_b", "total_sec")})


def draw(ax, rows):
    x = np.array([r["total_param_b"] for r in rows], dtype=float)
    y = np.array([r["total_sec"] for r in rows], dtype=float)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(16, 520)
    ax.set_ylim(4.0, 950)
    ax.grid(color=COLOR["grid"], linewidth=0.7, alpha=1.0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Parameter-weighted cost / video (B)", fontsize=8.2)
    ax.set_ylabel("Wall-clock seconds / video", fontsize=8.2)

    for xi, yi, row in zip(x, y, rows):
        m = row["method"]
        ax.scatter(
            xi, yi,
            marker="o", s=72,
            facecolor=METHOD_COLOR[m],
            edgecolor="white", linewidth=0.9,
            zorder=4,
        )
        dx, dy = LABEL_OFFSET[m]
        ax.annotate(
            m, (xi, yi),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx >= 0 else "right",
            va="center",
            fontsize=8.2,
            fontweight="bold" if m == "Ours" else "normal",
            color=METHOD_COLOR[m] if m == "Ours" else COLOR["text"],
        )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(COLOR["axis"])
    ax.spines["bottom"].set_color(COLOR["axis"])
    ax.tick_params(axis="both", labelsize=7.5, length=0)


def main():
    write_csv(ROWS)
    apply_style()
    fig, ax = plt.subplots(figsize=(3.05, 1.72), dpi=300)
    draw(ax, ROWS)
    fig.tight_layout(pad=0.15)
    fig.savefig(PDF_PATH, bbox_inches="tight")
    fig.savefig(PNG_PATH, dpi=220, bbox_inches="tight")
    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
