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


def read_real_rows() -> list[dict]:
    with SOURCE.open() as f:
        reader = csv.DictReader(f)
        return [
            {
                "saliency_backbone": row["saliency_backbone"],
                "verifier_order": row["verifier_order"],
                "dataset": row["dataset"],
                "acc": float(row["acc"]),
            }
            for row in reader
        ]


def make_mock(rows: list[dict]) -> list[dict]:
    rng = np.random.default_rng(20260427)
    im_values = np.array([r["acc"] for r in rows if r["dataset"] == "ImpliHateVid"], dtype=float)
    im_drop = FULL["ImpliHateVid"] - im_values
    im_drop = np.clip(im_drop, -0.6, 5.0)

    out = []
    for ds in DATASETS:
        ds_rows = [r for r in rows if r["dataset"] == ds]
        if ds == "ImpliHateVid":
            for row in ds_rows:
                item = dict(row)
                item["is_mock"] = "false"
                out.append(item)
            continue

        sampled_drop = rng.choice(im_drop, size=len(ds_rows), replace=True)
        jitter = rng.normal(0.0, 0.18, size=len(ds_rows))
        mocked = FULL[ds] - sampled_drop + jitter
        mocked = np.clip(mocked, FULL[ds] - 5.2, FULL[ds] + 0.15)
        for row, acc in zip(ds_rows, mocked):
            item = dict(row)
            item["acc"] = float(acc)
            item["is_mock"] = "true"
            out.append(item)
    return out


def write_csv(rows: list[dict]) -> None:
    path = FIG_DIR / "robustness_mock_preview.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["saliency_backbone", "verifier_order", "dataset", "acc", "is_mock"],
        )
        writer.writeheader()
        writer.writerows(rows)


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
    values = [np.array([r["acc"] for r in rows if r["dataset"] == ds], dtype=float) for ds in DATASETS]
    x = np.arange(1, len(DATASETS) + 1)
    fig, ax = plt.subplots(figsize=(3.45, 2.35), dpi=300)

    violins = ax.violinplot(values, positions=x, widths=0.68, showmeans=False, showextrema=False)
    for body in violins["bodies"]:
        body.set_facecolor("#6B8FBF")
        body.set_edgecolor("none")
        body.set_alpha(0.34)

    ax.boxplot(
        values,
        positions=x,
        widths=0.20,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 1.0},
        whiskerprops={"color": "black", "linewidth": 1.0},
        capprops={"color": "black", "linewidth": 1.0},
    )

    means = [vals.mean() for vals in values]
    stds = [vals.std() for vals in values]
    ax.errorbar(
        x,
        means,
        yerr=stds,
        fmt="o",
        color="black",
        ecolor="black",
        elinewidth=1.1,
        capsize=3,
        markersize=3.8,
        zorder=4,
        label="Mean ± std",
    )
    ax.scatter(
        x,
        [FULL[ds] for ds in DATASETS],
        marker="*",
        s=96,
        color="#D55E5E",
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
        label="Full",
    )

    for i, vals in enumerate(values, start=1):
        ax.text(i, 60.2, f"n={len(vals)}", ha="center", va="bottom", fontsize=6.8, color="#697386")

    ax.set_xticks(x)
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
    fig.savefig(FIG_DIR / "robustness_mock_preview.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "robustness_mock_preview.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = make_mock(read_real_rows())
    write_csv(rows)
    plot(rows)
    print(FIG_DIR / "robustness_mock_preview.pdf")
    print(FIG_DIR / "robustness_mock_preview.png")
    print(FIG_DIR / "robustness_mock_preview.csv")


if __name__ == "__main__":
    main()
