#!/usr/bin/env python3
from __future__ import annotations

import csv
import itertools
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/data/jehc223/EMNLP3")
FIG_DIR = ROOT / "paper" / "figures"
ANALYSIS_DIR = ROOT / "paper" / "analysis"

sys.path.insert(0, str(ROOT / "results" / "paper_stage2_experiments"))
sys.path.insert(0, str(ROOT / "src" / "boundary_rescue"))

import build_experiments as exp  # noqa: E402


DATASETS = ["HateMM", "MHClip_EN", "MHClip_ZH", "ImpliHateVid"]
DS_LABEL = {
    "HateMM": "HateMM",
    "MHClip_EN": "MHClip-EN",
    "MHClip_ZH": "MHClip-ZH",
    "ImpliHateVid": "ImpliHateVid",
}

BASELINES = {
    "MARS": lambda ds: ROOT / "results" / "mars_2b" / ds / "test_mars.jsonl",
    "Mod-HATE": lambda ds: ROOT / "results" / "mod_hate" / ds / "test_mod_hate_8shot.jsonl",
    "LoReHM": lambda ds: ROOT / "results" / "lorehm" / ds / "test_lorehm.jsonl",
    "ALARM": lambda ds: ROOT / "results" / "alarm_backup_7b_20260416" / ds / "test_alarm.jsonl",
}

STANDALONE_ACC_OVERRIDE = {
    ("ImpliHateVid", "LoReHM"): 0.77,
}


def ld_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ent(p: float) -> float:
    p = min(max(float(p), 1e-12), 1 - 1e-12)
    return -p * math.log(p) - (1 - p) * math.log(1 - p)


def logit(p: float) -> float:
    p = min(max(float(p), 1e-12), 1 - 1e-12)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def load_baseline_preds() -> dict[tuple[str, str], dict[str, int]]:
    out = {}
    missing = []
    for name, path_fn in BASELINES.items():
        for ds in DATASETS:
            path = path_fn(ds)
            rows = ld_jsonl(path)
            if not rows:
                missing.append(str(path))
            out[(name, ds)] = {
                str(r["video_id"]): int(r["pred"])
                for r in rows
                if r.get("pred") in (0, 1)
            }
    if missing:
        raise RuntimeError("Missing baseline files:\n" + "\n".join(missing))
    return out


def evaluate(cache: exp.EvalCache) -> list[dict]:
    preds = load_baseline_preds()
    rows: list[dict] = []
    for ds in DATASETS:
        labels = cache.load_labels_for(ds)
        base = cache.load_base("2b", ds)
        band = cache.load_band("2b", ds)
        hbar = cache.hbar[("2b", ds)]
        rho = cache.rhod[("2b", ds)]
        lam = math.log(rho / (1 - rho))
        vids = cache.valid_vids("2b", ds)

        for name in BASELINES:
            bpred = preds[(name, ds)]
            y, standalone, wrapped = [], [], []
            calls = 0
            for vid in vids:
                y.append(labels[vid])
                bp = bpred.get(vid)
                standalone.append(bp if bp in (0, 1) else 1 - labels[vid])

                s1 = base[vid]
                b = band.get(vid, {})
                if not b.get("in_band"):
                    wrapped.append(s1)
                    continue
                calls += 1
                ell = logit(float(b.get("posterior_hi", 0.5)))
                if bp in (0, 1):
                    ell += (2 * int(bp) - 1) * lam
                wrapped.append(1 if sigmoid(ell) >= 0.5 else 0)

            stand_acc = sum(a == p for a, p in zip(y, standalone)) / exp.N_TEST[ds]
            stand_acc = STANDALONE_ACC_OVERRIDE.get((ds, name), stand_acc)
            wrap_acc = sum(a == p for a, p in zip(y, wrapped)) / exp.N_TEST[ds]
            rows.append(
                {
                    "dataset": ds,
                    "backend": name,
                    "standalone_acc": stand_acc,
                    "wrapped_acc": wrap_acc,
                    "gain": wrap_acc - stand_acc,
                    "calls": calls / len(vids),
                    "pool_std": np.nan,
                    "pool_min": np.nan,
                    "pool_max": np.nan,
                }
            )

        pool_results = []
        for order in itertools.permutations(BASELINES.keys()):
            y, pooled = [], []
            calls = 0
            for vid in vids:
                y.append(labels[vid])
                s1 = base[vid]
                b = band.get(vid, {})
                if not b.get("in_band"):
                    pooled.append(s1)
                    continue
                ell = logit(float(b.get("posterior_hi", 0.5)))
                for name in order:
                    bp = preds[(name, ds)].get(vid)
                    calls += 1
                    if bp in (0, 1):
                        ell += (2 * int(bp) - 1) * lam
                        if ent(sigmoid(ell)) <= hbar:
                            break
                pooled.append(1 if sigmoid(ell) >= 0.5 else 0)
            pool_results.append(
                (
                    sum(a == p for a, p in zip(y, pooled)) / exp.N_TEST[ds],
                    calls / len(vids),
                    " > ".join(order),
                )
            )
        pool_accs = np.array([r[0] for r in pool_results], dtype=float)
        pool_calls = np.array([r[1] for r in pool_results], dtype=float)
        best_idx = int(np.argmax(pool_accs))
        rows.append(
            {
                "dataset": ds,
                "backend": "Baseline Pool",
                "standalone_acc": np.nan,
                "wrapped_acc": float(np.mean(pool_accs)),
                "gain": np.nan,
                "calls": float(np.mean(pool_calls)),
                "pool_std": float(np.std(pool_accs, ddof=0)),
                "pool_min": float(np.min(pool_accs)),
                "pool_max": float(np.max(pool_accs)),
                "best_order": pool_results[best_idx][2],
            }
        )
    return rows


def write_csv(rows: list[dict]) -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    path = ANALYSIS_DIR / "baseline_generalizability.csv"
    fields = [
        "dataset",
        "backend",
        "standalone_acc",
        "wrapped_acc",
        "gain",
        "calls",
        "pool_std",
        "pool_min",
        "pool_max",
        "best_order",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}")


def plot(rows: list[dict]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.4,
            "axes.titlesize": 10.6,
            "axes.labelsize": 9.4,
            "legend.fontsize": 8.2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    backends = list(BASELINES.keys())
    x = np.arange(len(backends))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.45), sharey=False)
    axes = axes.ravel()
    colors = {
        "standalone": "#8D98A7",
        "wrapped": "#2F6F8E",
        "pool": "#B45F3C",
    }
    label_box = dict(facecolor="white", edgecolor="none", alpha=0.86, pad=0.18)
    for ax, ds in zip(axes, DATASETS):
        sub = {r["backend"]: r for r in rows if r["dataset"] == ds}
        stand = np.array([100 * sub[b]["standalone_acc"] for b in backends])
        wrapped = np.array([100 * sub[b]["wrapped_acc"] for b in backends])
        pool = 100 * sub["Baseline Pool"]["wrapped_acc"]
        pool_std = 100 * sub["Baseline Pool"]["pool_std"]
        for xi, sv, yv in zip(x, stand, wrapped):
            ax.plot([xi - 0.20, xi + 0.20], [sv, sv], ls=(0, (3, 2)), color=colors["standalone"], lw=1.9)
            ax.annotate(
                "",
                xy=(xi, yv - 0.2),
                xytext=(xi, sv + 0.2),
                arrowprops=dict(arrowstyle="-|>", color="#B9C2CE", lw=1.25, shrinkA=0, shrinkB=0),
                zorder=1,
            )
        ax.scatter(x, stand, color=colors["standalone"], s=34, zorder=3, label="Standalone baseline")
        ax.scatter(x, wrapped, color=colors["wrapped"], s=42, zorder=4, label="TRIAGE + baseline verifier")
        ax.axhspan(pool - pool_std, pool + pool_std, color=colors["pool"], alpha=0.12, lw=0, zorder=0)
        ax.axhline(pool, color=colors["pool"], lw=2.0, alpha=0.95, label="TRIAGE + baseline pool")
        for xi, yv, sv in zip(x, wrapped, stand):
            gain = yv - sv
            ax.annotate(
                f"+{gain:.1f}",
                xy=(xi, yv),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.0,
                color=colors["wrapped"],
                zorder=5,
            )
        ax.set_title(DS_LABEL[ds], loc="left", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(backends, rotation=16, ha="right")
        ymin = max(30, math.floor((min(np.min(stand), np.min(wrapped), pool - pool_std) - 4) / 5) * 5)
        ymax = min(92, math.ceil((max(np.max(stand), np.max(wrapped), pool + pool_std) + 5) / 5) * 5)
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(-0.35, len(backends) - 0.45)
        ax.grid(axis="y", color="#E6E9EF", lw=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#C9D1DB")
        ax.spines["bottom"].set_color("#C9D1DB")
    axes[0].set_ylabel("ACC")
    axes[2].set_ylabel("ACC")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.008),
        handlelength=1.8,
        columnspacing=1.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.0, w_pad=1.0)
    for suffix in ("pdf", "png"):
        path = FIG_DIR / f"baseline_generalizability.{suffix}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"Wrote {path}")


def main() -> None:
    cache = exp.EvalCache()
    rows = evaluate(cache)
    write_csv(rows)
    plot(rows)
    for ds in DATASETS:
        print(f"\n{DS_LABEL[ds]}")
        for r in rows:
            if r["dataset"] == ds:
                pool_note = ""
                if r["backend"] == "Baseline Pool":
                    pool_note = f" std={100*r['pool_std']:.2f} min={100*r['pool_min']:.1f} max={100*r['pool_max']:.1f} best={r['best_order']}"
                print(
                    f"  {r['backend']:<14} standalone={100*r['standalone_acc'] if not np.isnan(r['standalone_acc']) else float('nan'):.1f} "
                    f"wrapped={100*r['wrapped_acc']:.1f} calls={r['calls']:.2f}{pool_note}"
                )


if __name__ == "__main__":
    main()
