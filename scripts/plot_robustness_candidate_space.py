from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/data/jehc223/home/tmp/matplotlib")

import matplotlib.pyplot as plt


ROOT = Path("/data/jehc223/EMNLP3")
FIG_DIR = ROOT / "paper" / "figures"
SOURCE = FIG_DIR / "robustness_all_configs.csv"

DATASETS = ["HateMM", "MHClip_EN", "MHClip_ZH", "ImpliHateVid"]
LABELS = ["HateMM", "MH-EN", "MH-ZH", "ImpliHV"]
FULL = {
    "HateMM": 86.51,
    "MHClip_EN": 79.50,
    "MHClip_ZH": 83.89,
    "ImpliHateVid": 82.75,
}

STAGE1_CANDIDATES = {
    "HateMM": ("2b", "qwen2.5-vl-7b", "pixtral-12b-2409"),
    "MHClip_EN": ("2b", "minicpm-v-26", "qwen2.5-vl-7b"),
    "MHClip_ZH": ("2b", "minicpm-v-26"),
    "ImpliHateVid": ("2b", "qwen2.5-vl-7b", "gemma-3-12b-it-16f"),
}


def read_rows() -> list[dict]:
    with SOURCE.open() as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            ds = row["dataset"]
            if row["saliency_backbone"] not in STAGE1_CANDIDATES[ds]:
                continue
            rows.append(
                {
                    "saliency_backbone": row["saliency_backbone"],
                    "verifier_order": row["verifier_order"],
                    "dataset": ds,
                    "acc": float(row["acc"]),
                }
            )
        return rows


def write_outputs(rows: list[dict]) -> None:
    csv_path = FIG_DIR / "robustness_candidate_space.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["saliency_backbone", "verifier_order", "dataset", "acc"])
        writer.writeheader()
        writer.writerows(rows)

    summary_path = FIG_DIR / "robustness_candidate_space_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "stage1_candidates", "n", "mean", "std", "min", "max"])
        writer.writeheader()
        for ds in DATASETS:
            vals = np.array([r["acc"] for r in rows if r["dataset"] == ds], dtype=float)
            writer.writerow(
                {
                    "dataset": ds,
                    "stage1_candidates": " / ".join(STAGE1_CANDIDATES[ds]),
                    "n": len(vals),
                    "mean": f"{vals.mean():.2f}",
                    "std": f"{vals.std():.2f}",
                    "min": f"{vals.min():.2f}",
                    "max": f"{vals.max():.2f}",
                }
            )


def plot(rows: list[dict]) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
        }
    )
    rng = np.random.default_rng(20260427)
    fig, ax = plt.subplots(figsize=(3.45, 2.35), dpi=300)
    xs = np.arange(1, len(DATASETS) + 1)

    for i, ds in enumerate(DATASETS, start=1):
        vals = np.array([r["acc"] for r in rows if r["dataset"] == ds], dtype=float)
        jitter = rng.uniform(-0.22, 0.22, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            s=7,
            color="#6B8FBF",
            alpha=0.34,
            linewidth=0,
        )
        mean = vals.mean()
        std = vals.std()
        ax.errorbar(
            i,
            mean,
            yerr=std,
            fmt="o",
            color="black",
            ecolor="black",
            elinewidth=1.1,
            capsize=3,
            markersize=3.8,
            zorder=4,
        )
        ax.text(i, 60.2, f"n={len(vals)}", ha="center", va="bottom", fontsize=6.8, color="#697386")

    ax.scatter(
        xs,
        [FULL[ds] for ds in DATASETS],
        marker="*",
        s=96,
        color="#D55E5E",
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
        label="Full",
    )
    ax.errorbar([], [], yerr=[], fmt="o", color="black", label="Mean ± std")

    ax.set_xticks(xs)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("Final ACC (%)")
    ax.set_ylim(58, 89)
    ax.set_yticks([60, 70, 80])
    ax.grid(axis="y", color="#E6EAF0", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#B8C0CC")
    ax.tick_params(axis="both", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    fig.tight_layout(pad=0.2)
    fig.savefig(FIG_DIR / "robustness_candidate_space.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "robustness_candidate_space.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = read_rows()
    write_outputs(rows)
    plot(rows)
    print(FIG_DIR / "robustness_candidate_space.pdf")
    print(FIG_DIR / "robustness_candidate_space.png")
    print(FIG_DIR / "robustness_candidate_space.csv")
    print(FIG_DIR / "robustness_candidate_space_summary.csv")


if __name__ == "__main__":
    main()
