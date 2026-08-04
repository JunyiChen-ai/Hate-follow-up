#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/data/jehc223/EMNLP2")
IN_PATH = ROOT / "results" / "boundary_rescue" / "reviewer_followups" / "unlabeled_pool_sensitivity_summary.csv"
OUT_PATH = ROOT / "paper" / "figures" / "unlabeled_pool_sensitivity.pdf"

DATASETS = [
    ("HateMM", "HateMM"),
    ("MHClip_EN", "MHClip-EN"),
    ("MHClip_ZH", "MHClip-ZH"),
    ("ImpliHateVid", "ImpliHateVid"),
]


def load_rows() -> dict[str, list[dict[str, float]]]:
    rows: dict[str, list[dict[str, float]]] = {}
    with open(IN_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.setdefault(row["dataset"], []).append(
                {
                    "pool_prop": 100 * float(row["pool_prop"]),
                    "acc_mean": 100 * float(row["acc_mean"]),
                    "acc_std": 100 * float(row["acc_std"]),
                    "mf1_mean": 100 * float(row["mf1_mean"]),
                    "mf1_std": 100 * float(row["mf1_std"]),
                }
            )
    for vals in rows.values():
        vals.sort(key=lambda x: x["pool_prop"])
    return rows


def padded_ylim(values: list[float]) -> tuple[float, float]:
    lo, hi = min(values), max(values)
    pad = max(1.0, (hi - lo) * 0.28)
    return max(0, lo - pad), min(100, hi + pad)


def main() -> None:
    rows = load_rows()
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.75,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 4.65), constrained_layout=False)
    axes = axes.ravel()
    acc_color = "#2f6fba"
    full_color = "#444444"

    for ax, (dataset, title) in zip(axes, DATASETS):
        vals = rows[dataset]
        x = np.asarray([r["pool_prop"] for r in vals])
        acc = np.asarray([r["acc_mean"] for r in vals])
        full_acc = float(acc[-1])

        ax.plot(
            x,
            acc,
            color=acc_color,
            linewidth=1.8,
            marker="o",
            markersize=4.0,
        )
        ax.axhline(
            full_acc,
            color=full_color,
            linewidth=1.0,
            linestyle=(0, (3.0, 2.0)),
            alpha=0.85,
        )
        ax.scatter([100], [full_acc], s=24, color=acc_color, edgecolor="white", linewidth=0.7, zorder=5)

        ax.set_title(title, pad=4)
        ax.set_xlabel("Unlabeled fitting pool (%)")
        ax.set_ylabel("ACC (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(["10", "20", "40", "60", "80", "100"])
        ax.grid(True, axis="y", color="#d8d8d8", linewidth=0.55, alpha=0.9)
        ax.grid(True, axis="x", color="#eeeeee", linewidth=0.45, alpha=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(*padded_ylim(list(acc)))
        ax.margins(x=0.035)

        ax.annotate(
            f"{full_acc:.1f}",
            xy=(100, full_acc),
            xytext=(-3, 7),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=7,
            color=acc_color,
        )
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.10, top=0.94, hspace=0.46, wspace=0.24)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
