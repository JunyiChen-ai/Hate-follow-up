#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/data/jehc223/EMNLP3")
IN_PATH = ROOT / "results" / "boundary_rescue" / "transferability_full_pipeline_2b.csv"
OUT_PATH = ROOT / "paper" / "figures" / "transferability_radar.pdf"

DATASETS = ["HateMM", "MHClip_EN", "MHClip_ZH", "ImpliHateVid"]
DISPLAY = {
    "HateMM": "HateMM",
    "MHClip_EN": "MHClip-EN",
    "MHClip_ZH": "MHClip-ZH",
    "ImpliHateVid": "ImpliHateVid",
}
METRICS = ["ACC", "M-F1", "M-P", "M-R"]
METRIC_KEYS = ["full_acc", "full_mf1", "full_mp", "full_mr"]

TRANSFER_COLOR = "#2E5E8C"
TESTFIT_COLOR = "#4C4C4C"
BASELINE_COLOR = "#C18B21"

# Target-specific TestFit full-pipeline baseline.
TARGET_TESTFIT = {
    "HateMM": [86.5, 85.8, 86.2, 85.5],
    "MHClip_EN": [76.4, 68.4, 73.1, 67.0],
    "MHClip_ZH": [79.9, 74.4, 76.9, 73.0],
    "ImpliHateVid": [81.0, 80.8, 82.5, 81.0],
}

# Dataset-specific best label-free / few-shot baseline from the main table.
BEST_LF_FS = {
    "HateMM": [79.5, 79.4, 79.8, 81.0],
    "MHClip_EN": [76.4, 67.3, 74.0, 65.8],
    "MHClip_ZH": [75.2, 70.2, 71.4, 70.3],
    "ImpliHateVid": [80.3, 80.3, 80.4, 80.3],
}


def load_best_transfer() -> dict[tuple[str, str], list[float]]:
    rows = []
    with open(IN_PATH) as f:
        for row in csv.DictReader(f):
            if int(row["is_diagonal"]):
                continue
            for key in METRIC_KEYS:
                row[key] = float(row[key])
            rows.append(row)

    best: dict[tuple[str, str], list[float]] = {}
    for source in DATASETS:
        for target in DATASETS:
            if source == target:
                continue
            candidates = [r for r in rows if r["source"] == source and r["target"] == target]
            chosen = sorted(candidates, key=lambda r: (r["full_acc"], r["full_mf1"]), reverse=True)[0]
            best[(source, target)] = [chosen[k] * 100 for k in METRIC_KEYS]
    return best


def close(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.concatenate([arr, arr[:1]])


def draw_direction(
    ax: plt.Axes,
    source: str,
    target: str,
    best_transfer: dict[tuple[str, str], list[float]],
) -> None:
    angles = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False)
    closed_angles = np.concatenate([angles, angles[:1]])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(60, 90)
    ax.set_xticks(angles)
    ax.set_xticklabels(METRICS, fontsize=5.8)
    ax.tick_params(axis="x", pad=-1)
    ax.set_yticks([60, 70, 80, 90])
    ax.set_yticklabels(["60", "70", "80", "90"], fontsize=5.2, color="#5F6770")
    ax.set_rlabel_position(22)
    ax.grid(color="#D5DAE0", linewidth=0.5)
    ax.spines["polar"].set_color("#AEB7C2")
    ax.spines["polar"].set_linewidth(0.65)
    ax.set_title(f"{DISPLAY[source]} $\\rightarrow$ {DISPLAY[target]}", fontsize=7.4, fontweight="semibold", pad=8)

    ax.plot(
        closed_angles,
        close(best_transfer[(source, target)]),
        color=TRANSFER_COLOR,
        linewidth=1.25,
        marker="o",
        markersize=2.4,
        label="TRIAGE Transfer",
        zorder=4,
    )
    ax.fill(closed_angles, close(best_transfer[(source, target)]), color=TRANSFER_COLOR, alpha=0.06, zorder=2)
    ax.plot(
        closed_angles,
        close(TARGET_TESTFIT[target]),
        color=TESTFIT_COLOR,
        linewidth=1.15,
        linestyle=(0, (4, 2)),
        marker="s",
        markersize=2.2,
        label="TRIAGE Target Test Fit",
        zorder=3,
    )
    ax.plot(
        closed_angles,
        close(BEST_LF_FS[target]),
        color=BASELINE_COLOR,
        linewidth=1.15,
        linestyle=(0, (1.2, 1.8)),
        marker="D",
        markersize=2.1,
        label="Best Label-Free/Few-Shot Baseline",
        zorder=3,
    )


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
        }
    )

    best_transfer = load_best_transfer()
    directions = [(s, t) for s in DATASETS for t in DATASETS if s != t]
    fig, axes = plt.subplots(3, 4, figsize=(7.55, 7.35), subplot_kw={"projection": "polar"})
    for ax, (source, target) in zip(axes.flat, directions):
        draw_direction(ax, source, target, best_transfer)

    handles = [
        plt.Line2D([0], [0], color=TRANSFER_COLOR, marker="o", linewidth=1.4, markersize=3,
                   label="TRIAGE Transfer"),
        plt.Line2D([0], [0], color=TESTFIT_COLOR, marker="s", linewidth=1.3,
                   linestyle=(0, (4, 2)), markersize=3, label="TRIAGE Target Test Fit"),
        plt.Line2D([0], [0], color=BASELINE_COLOR, marker="D", linewidth=1.3,
                   linestyle=(0, (1.2, 1.8)), markersize=3, label="Best Label-Free/Few-Shot Baseline"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=6.8,
        handlelength=1.8,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.035, right=0.985, top=0.965, bottom=0.075, wspace=0.58, hspace=0.50)
    fig.savefig(OUT_PATH, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
