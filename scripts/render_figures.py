"""Render Fig 3 (band localises errors) and Fig 4 (backbone-agnostic Delta-acc) from frozen artefacts."""
from __future__ import annotations
import sys, json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "/data/jehc223/EMNLP3/src")
from boundary_rescue.grid_eval_all import load_labels, SKIP_VIDEOS, ld_jsonl

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = Path("/data/jehc223/EMNLP3/paper/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    ("HateMM", "HateMM"),
    ("MHClip_EN", "MHClip-EN"),
    ("MHClip_ZH", "MHClip-ZH"),
    ("ImpliHateVid", "ImpliHateVid"),
]
BAND_PATH = Path("/data/jehc223/EMNLP3/results/boundary_rescue")
GRID_PATH = BAND_PATH / "grid_eval" / "grid_raw.jsonl"


def fig3_band_errors():
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    names, in_rate, out_rate = [], [], []
    # Temporary: MHClip_ZH uses the Bayes-overlap TestFit band (label-free); the other
    # three datasets use the default above-mean-entropy band.
    ZH_OVERRIDE_FILE = "candidates_bayes_band_testfit.jsonl"
    for ds, title in DATASETS:
        universe = list(ld_jsonl(BAND_PATH / ds / "candidates_entropy_band_2b.jsonl"))
        labels = load_labels(ds)
        skip = SKIP_VIDEOS.get(ds, set())
        universe = [r for r in universe if r["video_id"] in labels and r["video_id"] not in skip]
        preds = {r["video_id"]: int(r["pred_baseline"]) for r in universe}
        vids = [r["video_id"] for r in universe]
        errors_by_v = {v: int(preds[v] != labels[v]) for v in vids}

        if ds == "MHClip_ZH":
            band_rows = list(ld_jsonl(BAND_PATH / ds / ZH_OVERRIDE_FILE))
            band_set = {r["video_id"] for r in band_rows}
            in_band = np.array([v in band_set for v in vids], dtype=bool)
        else:
            in_band = np.array([r["in_band"] for r in universe], dtype=bool)

        errors = np.array([errors_by_v[v] for v in vids])
        names.append(title)
        in_rate.append(100.0 * errors[in_band].sum() / max(in_band.sum(), 1))
        out_rate.append(100.0 * errors[~in_band].sum() / max((~in_band).sum(), 1))

    y = np.arange(len(names))[::-1]
    for i, (o, inb) in zip(y, zip(out_rate, in_rate)):
        ax.annotate("", xy=(inb, i), xytext=(o, i),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.32,head_length=0.55",
                                    color="#1B4F72", lw=2.2, shrinkA=4, shrinkB=4))
        ax.plot(o, i, "o", color="#B0B0B0", markersize=9,
                markeredgecolor="black", markeredgewidth=0.6, zorder=3)
        ax.plot(inb, i, "o", color="#1B4F72", markersize=10,
                markeredgecolor="black", markeredgewidth=0.6, zorder=4)
        ax.text(o - 1.2, i, f"{o:.0f}", ha="right", va="center",
                fontsize=8, color="#555")
        ax.text(inb + 1.2, i, f"{inb:.0f}", ha="left", va="center",
                fontsize=8, color="#1B4F72", fontweight="bold")
        ax.text((o + inb) / 2, i + 0.26, f"{inb / max(o, 0.01):.1f}$\\times$",
                ha="center", va="bottom", fontsize=7, color="#1B4F72", style="italic")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("stage-1 error rate (\\%)", fontsize=9)
    ax.set_xlim(0, max(in_rate) * 1.25)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.tick_params(labelsize=8)
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    # Custom legend
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#B0B0B0",
               markersize=8, markeredgecolor="black", markeredgewidth=0.6,
               label="outside band"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1B4F72",
               markersize=8, markeredgecolor="black", markeredgewidth=0.6,
               label="inside band"),
    ]
    ax.legend(handles=legend_items, fontsize=7, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    plt.tight_layout()
    out = FIG_DIR / "fig3_band_errors.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig3] wrote {out}")
    for n, i, o in zip(names, in_rate, out_rate):
        print(f"  {n:<14s} err_rate_inside={i:.1f}%  err_rate_outside={o:.1f}%")


def fig4_backbone_delta():
    slugs_display = [
        ("2b", "Qwen3-VL-2B"),
        ("qwen2.5-vl-7b", "Qwen2.5-VL-7B"),
        ("gemma-3-12b-it", "Gemma-3-12B-IT (8f)"),
        ("gemma-3-12b-it-16f", "Gemma-3-12B-IT (16f)"),
        ("pixtral-12b-2409", "Pixtral-12B"),
        ("minicpm-v-26", "MiniCPM-V-2.6"),
    ]
    target_ds = {d for d, _ in DATASETS}

    rows = list(ld_jsonl(GRID_PATH))
    # Group by (slug, triplet) -> {ds: delta_acc}
    grouped: dict[tuple[str, tuple], dict[str, float]] = defaultdict(dict)
    for r in rows:
        key = (r["slug"], tuple(r["triplet"]))
        grouped[key][r["ds"]] = r["delta_acc"]

    per_slug_best = {}
    for slug, _ in slugs_display:
        cand = []
        for (s, tri), ds_map in grouped.items():
            if s != slug:
                continue
            if not target_ds.issubset(ds_map.keys()):
                continue
            avg = sum(ds_map[d] for d in target_ds) / len(target_ds)
            cand.append(avg)
        per_slug_best[slug] = max(cand) * 100.0 if cand else 0.0

    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    names = [d for _, d in slugs_display]
    values = [per_slug_best[s] for s, _ in slugs_display]
    x = np.arange(len(values))
    bars = ax.bar(x, values, color="#2471A3", width=0.65)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:+.1f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(r"avg $\Delta$-acc across 4 datasets (pp)", fontsize=9)
    ax.grid(alpha=0.25, axis="y", linewidth=0.5)
    ax.set_ylim(0, max(values) * 1.25 if values else 1)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    out = FIG_DIR / "fig4_backbone_delta.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig4] wrote {out}")
    print("[fig4] per-slug avg best-triplet Delta-acc (pp):")
    for s, d in slugs_display:
        print(f"  {d:24s} {per_slug_best[s]:+.2f}")


if __name__ == "__main__":
    fig3_band_errors()
    fig4_backbone_delta()
