#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/data/jehc223/EMNLP2")
IN_PATH = ROOT / "results" / "boundary_rescue" / "k_sensitivity" / "k_sensitivity_summary.csv"
OUT_PATH = ROOT / "paper" / "figures" / "k_sensitivity.pdf"

DATASETS = [
    ("HateMM", "HateMM"),
    ("MHClip_EN", "MHClip-EN"),
    ("MHClip_ZH", "MHClip-ZH"),
    ("ImpliHateVid", "ImpliHateVid"),
]


def load_rows() -> dict[tuple[str, str], list[dict[str, float]]]:
    rows: dict[tuple[str, str], list[dict[str, float]]] = {}
    with open(IN_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["dataset"], row["method"])
            rows.setdefault(key, []).append(
                {
                    "k": float(row["k"]),
                    "acc_default": 100 * float(row["acc_default"]),
                    "acc_std": 100 * float(row["acc_std"]),
                }
            )
    for values in rows.values():
        values.sort(key=lambda x: x["k"])
    return rows


def padded_ylim(values: list[float]) -> tuple[float, float]:
    lo, hi = min(values), max(values)
    pad = max(0.8, (hi - lo) * 0.35)
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
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), constrained_layout=False)
    axes = axes.ravel()
    triage_color = "#2f6fba"
    majority_color = "#5f6368"
    band_majority_color = "#b36b16"

    legend_handles = None
    for ax, (dataset, title) in zip(axes, DATASETS):
        triage = rows[(dataset, "triage")]
        majority = rows[(dataset, "majority")]
        band_majority = rows[(dataset, "band_majority")]

        k = np.asarray([r["k"] for r in triage])
        t_center = np.asarray([r["acc_default"] for r in triage])
        t_std = np.asarray([r["acc_std"] for r in triage])
        m_center = np.asarray([r["acc_default"] for r in majority])
        m_std = np.asarray([r["acc_std"] for r in majority])
        bm_center = np.asarray([r["acc_default"] for r in band_majority])
        bm_std = np.asarray([r["acc_std"] for r in band_majority])

        ax.fill_between(
            k,
            t_center - t_std,
            t_center + t_std,
            color=triage_color,
            alpha=0.15,
            linewidth=0,
        )
        ax.fill_between(
            k,
            m_center - m_std,
            m_center + m_std,
            color=majority_color,
            alpha=0.12,
            linewidth=0,
        )
        ax.fill_between(
            k,
            bm_center - bm_std,
            bm_center + bm_std,
            color=band_majority_color,
            alpha=0.11,
            linewidth=0,
        )
        triage_line = ax.plot(
            k,
            t_center,
            color=triage_color,
            marker="o",
            markersize=3.8,
            linewidth=1.7,
            label="TRIAGE",
        )[0]
        majority_line = ax.plot(
            k,
            m_center,
            color=majority_color,
            marker="s",
            markersize=3.4,
            linewidth=1.4,
            linestyle="--",
            label="MAJORITY VOTE",
        )[0]
        band_majority_line = ax.plot(
            k,
            bm_center,
            color=band_majority_color,
            marker="^",
            markersize=3.6,
            linewidth=1.45,
            linestyle="-.",
            label="BOUNDARY-REGION MAJORITY VOTE",
        )[0]

        ax.axvline(3, color="#c27c2c", linewidth=0.9, linestyle=":", alpha=0.85)
        ax.set_title(title, pad=4)
        ax.set_xticks(k)
        ax.set_xlabel(r"Verifier pool size $K$")
        ax.set_ylabel("ACC (%)")
        ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.55, alpha=0.85)
        ax.grid(True, axis="x", color="#eeeeee", linewidth=0.45, alpha=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        y_values = list(t_center - t_std) + list(t_center + t_std)
        y_values += list(m_center - m_std) + list(m_center + m_std)
        y_values += list(bm_center - bm_std) + list(bm_center + bm_std)
        ax.set_ylim(*padded_ylim(y_values))
        legend_handles = [triage_line, majority_line, band_majority_line]

    fig.legend(
        legend_handles,
        ["TRIAGE", "MAJORITY VOTE", "BOUNDARY-REGION MAJORITY VOTE"],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
        handlelength=1.8,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.095, top=0.87, hspace=0.48, wspace=0.24)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
