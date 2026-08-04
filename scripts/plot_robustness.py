from __future__ import annotations

import csv
import itertools
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/data/jehc223/home/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/data/jehc223/EMNLP3")
OUT_DIR = ROOT / "paper" / "figures"

sys.path.insert(0, str(ROOT / "results" / "paper_stage2_experiments"))

import build_experiments as exp  # noqa: E402


DATASETS = ["HateMM", "MHClip_EN", "MHClip_ZH", "ImpliHateVid"]
DATASET_LABELS = ["HateMM", "MH-EN", "MH-ZH", "ImpliHV"]
MAIN_ORDER = ("gemma-3-27b-it", "qwen2.5-vl-32b-awq", "qwen2.5-vl-72b-awq")
PAPER_FLOOR = {
    "HateMM": 83.0,
    "MHClip_EN": 76.0,
    "MHClip_ZH": 80.0,
    "ImpliHateVid": 80.0,
}

SALiency_BACKBONES = (
    "2b",
    "qwen2.5-vl-7b",
    "gemma-3-12b-it",
    "gemma-3-12b-it-16f",
    "pixtral-12b-2409",
    "minicpm-v-26",
)

VERIFIER_POOL = (
    "gemma-3-27b-it",
    "qwen2.5-vl-32b-awq",
    "qwen2.5-vl-72b-awq",
    "qwen3-vl-8b",
    "internvl35-8b",
    "gemma-3-12b-it",
)


def per_dataset_acc(row: dict) -> dict[str, float]:
    return {d["ds"]: 100.0 * d["acc"] for d in row["per_dataset"]}


def collect() -> tuple[dict[str, float], list[dict]]:
    cache = exp.EvalCache()
    full = per_dataset_acc(
        exp.eval_sequential(
            cache,
            MAIN_ORDER,
            slug="2b",
            rho_mode="entropy",
            route="band",
            stop=True,
            name="full",
        )
    )

    rows: list[dict] = []
    for slug in SALiency_BACKBONES:
        for order in itertools.permutations(VERIFIER_POOL, 3):
            result = exp.eval_sequential(
                cache,
                order,
                slug=slug,
                rho_mode="entropy",
                route="band",
                stop=True,
                name="robustness",
            )
            acc = per_dataset_acc(result)
            for ds in DATASETS:
                rows.append(
                    {
                        "saliency_backbone": slug,
                        "verifier_order": " > ".join(order),
                        "dataset": ds,
                        "acc": acc[ds],
                    }
                )
    return full, rows


def write_csv(rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "robustness_all_configs.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["saliency_backbone", "verifier_order", "dataset", "acc"])
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path) -> list[dict]:
    with path.open() as f:
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


def filter_rows(rows: list[dict]) -> list[dict]:
    by_config: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (row["saliency_backbone"], row["verifier_order"])
        by_config[key][row["dataset"]] = row["acc"]

    selected_keys = []
    for key, acc in by_config.items():
        if all(ds in acc and acc[ds] >= PAPER_FLOOR[ds] for ds in DATASETS):
            selected_keys.append(key)

    selected = []
    selected_key_set = set(selected_keys)
    for row in rows:
        key = (row["saliency_backbone"], row["verifier_order"])
        if key in selected_key_set:
            selected.append(row)
    return selected


def filter_rows_near_full(rows: list[dict], full: dict[str, float], tolerance: float = 3.0) -> list[dict]:
    by_config: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (row["saliency_backbone"], row["verifier_order"])
        by_config[key][row["dataset"]] = row["acc"]

    selected_keys = []
    for key, acc in by_config.items():
        if all(ds in acc and abs(acc[ds] - full[ds]) <= tolerance for ds in DATASETS):
            selected_keys.append(key)

    selected_key_set = set(selected_keys)
    return [
        row
        for row in rows
        if (row["saliency_backbone"], row["verifier_order"]) in selected_key_set
    ]


def write_filtered_outputs(rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "robustness_filtered_configs.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["saliency_backbone", "verifier_order", "dataset", "acc"])
        writer.writeheader()
        writer.writerows(rows)

    summary_path = OUT_DIR / "robustness_filtered_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "n", "mean", "std", "min", "max"])
        writer.writeheader()
        for ds in DATASETS:
            vals = np.array([r["acc"] for r in rows if r["dataset"] == ds], dtype=float)
            writer.writerow(
                {
                    "dataset": ds,
                    "n": len(vals),
                    "mean": f"{vals.mean():.2f}",
                    "std": f"{vals.std():.2f}",
                    "min": f"{vals.min():.2f}",
                    "max": f"{vals.max():.2f}",
                }
            )


def write_near_full_outputs(rows: list[dict], full: dict[str, float]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "robustness_near_full_configs.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["saliency_backbone", "verifier_order", "dataset", "acc", "acc_drop"],
        )
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["acc_drop"] = full[row["dataset"]] - row["acc"]
            writer.writerow(out)

    summary_path = OUT_DIR / "robustness_near_full_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "n", "mean_drop", "std_drop", "min_drop", "max_drop"])
        writer.writeheader()
        for ds in DATASETS:
            vals = np.array([full[ds] - r["acc"] for r in rows if r["dataset"] == ds], dtype=float)
            writer.writerow(
                {
                    "dataset": ds,
                    "n": len(vals),
                    "mean_drop": f"{vals.mean():.2f}",
                    "std_drop": f"{vals.std():.2f}",
                    "min_drop": f"{vals.min():.2f}",
                    "max_drop": f"{vals.max():.2f}",
                }
            )


def plot_distribution(
    full: dict[str, float],
    rows: list[dict],
    stem: str,
    ylim: tuple[float, float],
    yticks: list[int],
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

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

    values = []
    means = []
    stds = []
    for ds in DATASETS:
        vals = np.array([r["acc"] for r in rows if r["dataset"] == ds], dtype=float)
        values.append(vals)
        means.append(vals.mean())
        stds.append(vals.std())

    x = np.arange(1, len(DATASETS) + 1)
    fig, ax = plt.subplots(figsize=(3.45, 2.35), dpi=300)

    violins = ax.violinplot(values, positions=x, widths=0.68, showmeans=False, showextrema=False)
    for body in violins["bodies"]:
        body.set_facecolor("#6B8FBF")
        body.set_edgecolor("none")
        body.set_alpha(0.34)

    box = ax.boxplot(
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
    _ = box

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
        [full[ds] for ds in DATASETS],
        marker="*",
        s=96,
        color="#D55E5E",
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
        label="Full",
    )

    for i, vals in enumerate(values, start=1):
        ax.text(i, ylim[0] + 0.4, f"n={len(vals)}", ha="center", va="bottom", fontsize=6.8, color="#697386")

    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_LABELS)
    ax.set_ylabel("Final ACC (%)")
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.grid(axis="y", color="#E6EAF0", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#B8C0CC")
    ax.tick_params(axis="both", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    fig.tight_layout(pad=0.2)

    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def plot_drop(full: dict[str, float], rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
        }
    )

    values = []
    for ds in DATASETS:
        vals = np.array([full[ds] - r["acc"] for r in rows if r["dataset"] == ds], dtype=float)
        values.append(vals)

    x = np.arange(1, len(DATASETS) + 1)
    fig, ax = plt.subplots(figsize=(3.45, 2.20), dpi=300)

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
    )
    ax.axhline(0, color="#D55E5E", linewidth=1.1, linestyle="--")

    for i, vals in enumerate(values, start=1):
        ax.text(i, 3.23, f"n={len(vals)}", ha="center", va="bottom", fontsize=6.8, color="#697386")

    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_LABELS)
    ax.set_ylabel("ACC drop from Full")
    ax.set_ylim(-0.35, 3.55)
    ax.set_yticks([0, 1, 2, 3])
    ax.grid(axis="y", color="#E6EAF0", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#B8C0CC")
    ax.tick_params(axis="both", length=0)
    fig.tight_layout(pad=0.2)

    fig.savefig(OUT_DIR / "robustness_drop_near_full.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "robustness_drop_near_full.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    all_csv = OUT_DIR / "robustness_all_configs.csv"
    full = per_dataset_acc(
        exp.eval_sequential(
            exp.EvalCache(),
            MAIN_ORDER,
            slug="2b",
            rho_mode="entropy",
            route="band",
            stop=True,
            name="full",
        )
    )
    if all_csv.exists():
        rows = load_csv(all_csv)
    else:
        _, rows = collect()
        write_csv(rows)

    selected = filter_rows(rows)
    near_full = filter_rows_near_full(rows, full, tolerance=3.0)
    write_filtered_outputs(selected)
    write_near_full_outputs(near_full, full)

    plot_distribution(full, rows, "robustness_all_configs", ylim=(58, 89), yticks=[60, 70, 80])
    plot_distribution(full, selected, "robustness_filtered_configs", ylim=(75, 89), yticks=[76, 80, 84, 88])
    plot_distribution(full, near_full, "robustness_acc_near_full", ylim=(75, 89), yticks=[76, 80, 84, 88])
    plot_drop(full, near_full)

    print(OUT_DIR / "robustness_all_configs.pdf")
    print(OUT_DIR / "robustness_all_configs.png")
    print(OUT_DIR / "robustness_all_configs.csv")
    print(OUT_DIR / "robustness_filtered_configs.pdf")
    print(OUT_DIR / "robustness_filtered_configs.png")
    print(OUT_DIR / "robustness_filtered_configs.csv")
    print(OUT_DIR / "robustness_filtered_summary.csv")
    print(OUT_DIR / "robustness_drop_near_full.pdf")
    print(OUT_DIR / "robustness_drop_near_full.png")
    print(OUT_DIR / "robustness_acc_near_full.pdf")
    print(OUT_DIR / "robustness_acc_near_full.png")
    print(OUT_DIR / "robustness_near_full_configs.csv")
    print(OUT_DIR / "robustness_near_full_summary.csv")


if __name__ == "__main__":
    main()
