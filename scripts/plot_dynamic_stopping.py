"""Marimekko/mosaic dynamic_stopping_composition.

Bucket widths encode n_k directly; heights remain a 100% stack of
confirmed / rescued / failed.
"""

from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/data/jehc223/home/tmp/matplotlib")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np

from _paper_palette import COLOR, apply_style, style_axes


ROOT = Path("/data/jehc223/EMNLP2")
OUT_DIR = ROOT / "paper" / "figures"
CSV_PATH = OUT_DIR / "dynamic_stopping_composition.csv"

DATASET_ORDER = ["HateMM", "MHClip_EN", "MHClip_ZH", "ImpliHateVid"]
DATASET_LABELS = {
    "HateMM": "HateMM",
    "MHClip_EN": "MHClip-EN",
    "MHClip_ZH": "MHClip-ZH",
    "ImpliHateVid": "ImpliHateVid",
}


def collect_rows() -> list[dict]:
    sys.path.insert(0, str(ROOT / "results" / "paper_stage2_experiments"))
    import build_experiments as exp  # type: ignore

    ORDER = ("gemma-3-27b-it", "qwen2.5-vl-32b-awq", "qwen2.5-vl-72b-awq")
    cache = exp.EvalCache()
    rows: list[dict] = []

    for ds in DATASET_ORDER:
        labels = cache.load_labels_for(ds)
        base = cache.load_base("2b", ds)
        band = cache.load_band("2b", ds)
        hbar = cache.hbar[("2b", ds)]
        rho = cache.rhod[("2b", ds)]
        lam = math.log(rho / (1.0 - rho))
        judges = [cache.load_judge(j, ds) for j in ORDER]

        buckets = {k: {"n": 0, "confirmed": 0, "rescued": 0, "failed": 0}
                   for k in (1, 2, 3)}

        for video_id in cache.valid_vids("2b", ds):
            row = band.get(video_id, {})
            if not row.get("in_band"):
                continue
            y = labels[video_id]
            s1 = base[video_id]
            ell = exp.logit(float(row.get("posterior_hi", 0.5)))
            used = 0
            for table in judges:
                used += 1
                verdict = table.get(video_id, {}).get("pred")
                if verdict in (0, 1):
                    ell += (2 * int(verdict) - 1) * lam
                    if exp.ent(exp.sigmoid(ell)) <= hbar:
                        break
            final = 1 if exp.sigmoid(ell) >= 0.5 else 0
            bucket = buckets[used]
            bucket["n"] += 1
            if s1 == y and final == y:
                bucket["confirmed"] += 1
            elif s1 != y and final == y:
                bucket["rescued"] += 1
            else:
                bucket["failed"] += 1

        for bucket_id, bucket in buckets.items():
            n = bucket["n"]
            rows.append({
                "dataset": ds,
                "dataset_label": DATASET_LABELS[ds],
                "bucket": bucket_id,
                "n": n,
                "confirmed": bucket["confirmed"],
                "rescued": bucket["rescued"],
                "failed": bucket["failed"],
                "confirmed_rate": bucket["confirmed"] / n if n else 0.0,
                "rescued_rate": bucket["rescued"] / n if n else 0.0,
                "failed_rate": bucket["failed"] / n if n else 0.0,
            })
    return rows


def write_stats(rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "dataset_label", "bucket", "n",
              "confirmed", "rescued", "failed",
              "confirmed_rate", "rescued_rate", "failed_rate"]
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_rows() -> list[dict]:
    rows = []
    with CSV_PATH.open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "dataset": r["dataset"],
                "bucket": int(r["bucket"]),
                "n": int(r["n"]),
                "confirmed_rate": float(r["confirmed_rate"]),
                "rescued_rate": float(r["rescued_rate"]),
                "failed_rate": float(r["failed_rate"]),
            })
    return rows


def plot(rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()

    colors = {
        "confirmed_rate": COLOR["primary"],
        "rescued_rate":   COLOR["secondary"],
        "failed_rate":    COLOR["ours"],
    }
    labels = {
        "confirmed_rate": "Confirmed",
        "rescued_rate":   "Rescued",
        "failed_rate":    "Failed",
    }

    by_key = {(r["dataset"], r["bucket"]): r for r in rows}
    fig, axes = plt.subplots(2, 2, figsize=(3.45, 3.45), dpi=300, sharey=True)
    axes = axes.ravel()

    for ax, ds in zip(axes, DATASET_ORDER):
        ns = np.array([by_key[(ds, k)]["n"] for k in (1, 2, 3)], dtype=float)
        total = ns.sum()
        widths = ns / total
        cumsum = np.concatenate(([0.0], np.cumsum(widths)))
        centers = (cumsum[:-1] + cumsum[1:]) / 2

        bottom = np.zeros(3)
        for key in ("confirmed_rate", "rescued_rate", "failed_rate"):
            values = np.array([100.0 * by_key[(ds, k)][key] for k in (1, 2, 3)])
            ax.bar(
                centers, values, bottom=bottom, width=widths * 0.94,
                color=colors[key], label=labels[key],
                edgecolor="white", linewidth=0.6, align="center",
            )
            for cx, w, v, b in zip(centers, widths, values, bottom):
                if v >= 14 and w > 0.10:
                    ax.text(
                        cx, b + v / 2, f"{v:.0f}",
                        ha="center", va="center",
                        color="white", fontsize=7.8, fontweight="bold",
                    )
            bottom += values

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 50, 100])
        ax.set_xticks(centers)
        ax.set_xticklabels([f"{k}" for k in (1, 2, 3)], fontsize=8.5)

        ax.set_title(
            DATASET_LABELS[ds], pad=14.0, fontweight="bold",
            color=COLOR["text"], fontsize=9.0,
        )
        ax.text(
            0.5, 1.018, f"n: {int(ns[0])} / {int(ns[1])} / {int(ns[2])}",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=7.2, color=COLOR["muted"], style="italic",
        )

        style_axes(ax)
        ax.tick_params(axis="x", pad=3.0, length=0)
        ax.tick_params(axis="y", pad=2.5, length=0)
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("Composition (%)", fontsize=8.5, labelpad=3)
    axes[2].set_ylabel("Composition (%)", fontsize=8.5, labelpad=3)

    fig.supxlabel(
        "Number of verifier calls",
        fontsize=9.0, color=COLOR["text"], y=0.105,
    )
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels,
        loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, -0.005),
        handlelength=1.4, handletextpad=0.4, columnspacing=1.4,
        fontsize=8.0,
    )
    fig.tight_layout(rect=[0, 0.10, 1, 1], pad=0.30, h_pad=1.6, w_pad=0.6)

    fig.savefig(OUT_DIR / "dynamic_stopping_composition.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "dynamic_stopping_composition.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if CSV_PATH.exists():
        rows = load_rows()
    else:
        rows = collect_rows()
        write_stats(rows)
    plot(rows)
    print(OUT_DIR / "dynamic_stopping_composition.pdf")


if __name__ == "__main__":
    main()
