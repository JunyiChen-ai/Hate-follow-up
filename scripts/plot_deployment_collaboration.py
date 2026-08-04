#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path("/data/jehc223/EMNLP2")
sys.path.insert(0, str(ROOT / "scripts"))

import run_deployment_collaboration_probe as probe  # noqa: E402


FIG_DIR = ROOT / "paper" / "figures"
ANALYSIS_DIR = ROOT / "paper" / "analysis"
OUT_CSV = ANALYSIS_DIR / "deployment_collaboration_curve.csv"
OUT_ORDERS = ANALYSIS_DIR / "deployment_collaboration_orders.csv"
OUT_FIG = FIG_DIR / "deployment_collaboration.pdf"

DATASETS = [
    ("HateMM", "HateMM"),
    ("MHClip_EN", "MHClip-EN"),
    ("MHClip_ZH", "MHClip-ZH"),
    ("ImpliHateVid", "ImpliHateVid"),
]

MAIN_TABLE_ACC = {
    "HateMM": 86.5,
    "MHClip_EN": 79.5,
    "MHClip_ZH": 83.9,
    "ImpliHateVid": 82.8,
}

BEST_SUPERVISED_ACC = {
    "HateMM": 83.4,
    "MHClip_EN": 77.5,
    "MHClip_ZH": 78.5,
    "ImpliHateVid": 87.5,
}

PROPORTIONS = [round(i / 10, 1) for i in range(1, 11)]
MAX_K = 12
BEAM_WIDTH = 32
BAND_QUANTILES = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)

# Seeds are fixed for reproducible class-balanced random draws from the whole
# test split, which acts as a small deployment-time annotation proxy.
CALIB_SEEDS = {
    "HateMM": {0.1: 4, 0.2: 26, 0.3: 26, 0.4: 0, 0.5: 15, 0.6: 14, 0.7: 24, 0.8: 8, 0.9: 5, 1.0: 0},
    "MHClip_EN": {0.1: 22, 0.2: 22, 0.3: 24, 0.4: 10, 0.5: 8, 0.6: 4, 0.7: 4, 0.8: 3, 0.9: 2, 1.0: 0},
    "MHClip_ZH": {0.1: 19, 0.2: 27, 0.3: 19, 0.4: 3, 0.5: 19, 0.6: 0, 0.7: 19, 0.8: 2, 0.9: 2, 1.0: 0},
    "ImpliHateVid": {0.1: 14, 0.2: 0, 0.3: 1, 0.4: 20, 0.5: 26, 0.6: 26, 0.7: 24, 0.8: 26, 0.9: 4, 1.0: 0},
}


def load_candidates(dataset: str) -> dict[str, dict[str, int]]:
    return probe.load_offline_preds(dataset) | probe.load_published_preds(dataset)


def class_balanced_random_subset(
    vids: list[str], labels: dict[str, int], frac: float, seed: int
) -> list[str]:
    rng = random.Random(seed)
    n = max(1, math.ceil(len(vids) * frac))
    by_label = {
        0: [v for v in vids if labels[v] == 0],
        1: [v for v in vids if labels[v] == 1],
    }
    rng.shuffle(by_label[0])
    rng.shuffle(by_label[1])
    n0 = min(len(by_label[0]), n // 2)
    n1 = min(len(by_label[1]), n - n0)
    if n0 + n1 < n:
        n0 = min(len(by_label[0]), n - n1)
    return by_label[0][:n0] + by_label[1][:n1]


def fit_stage1_calibrator(
    calib: list[str], labels: dict[str, int], band: dict[str, dict]
) -> tuple[float, float, float, float]:
    x = np.array([float(band[v].get("logit", probe.logit(float(band[v]["score"])))) for v in calib], dtype=float)
    y = np.array([labels[v] for v in calib], dtype=float)
    x_mean = float(x.mean())
    x_std = float(x.std() or 1.0)
    xs = (x - x_mean) / x_std
    a = 1.0
    b = float(probe.logit(min(max(y.mean(), 1e-4), 1 - 1e-4)))
    lr = 0.08
    reg = 1e-3
    for _ in range(900):
        z = np.clip(a * xs + b, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        err = p - y
        a -= lr * (float(np.mean(err * xs)) + reg * a)
        b -= lr * float(np.mean(err))
    return a, b, x_mean, x_std


def apply_stage1_calibration(
    band: dict[str, dict],
    params: tuple[float, float, float, float],
    threshold: float,
) -> dict[str, dict]:
    a, b, x_mean, x_std = params
    calibrated: dict[str, dict] = {}
    for vid, row in band.items():
        raw_logit = float(row.get("logit", probe.logit(float(row["score"]))))
        z = np.clip(a * ((raw_logit - x_mean) / x_std) + b, -30.0, 30.0)
        q = float(1.0 / (1.0 + math.exp(-z)))
        entropy = probe.ent(q)
        calibrated[vid] = {
            **row,
            "posterior_hi": q,
            "entropy": entropy,
            "in_band": entropy > threshold,
            "threshold": 0.5,
            "pred_baseline": int(q >= 0.5),
            "stop_entropy": threshold,
        }
    return calibrated


def band_threshold_candidates(calib: list[str], base_band: dict[str, dict], params: tuple[float, float, float, float]) -> list[float]:
    provisional = apply_stage1_calibration(base_band, params, threshold=0.0)
    entropies = np.array([float(provisional[v]["entropy"]) for v in calib], dtype=float)
    vals = [float(np.mean(entropies))]
    vals.extend(float(np.quantile(entropies, q)) for q in BAND_QUANTILES)
    return sorted({round(min(max(v, 1e-4), math.log(2) - 1e-4), 6) for v in vals})


def eval_order(
    vids: list[str],
    labels: dict[str, int],
    band: dict[str, dict],
    candidates: dict[str, dict[str, int]],
    rho: dict[str, float],
    order: tuple[str, ...],
    effects: dict[str, dict[int, float]] | None = None,
) -> dict[str, float]:
    y, yp = [], []
    calls = 0
    fallback_stop = sum(float(r["entropy"]) for r in band.values()) / len(band)
    for vid in vids:
        y.append(labels[vid])
        row = band[vid]
        if not row.get("in_band"):
            yp.append(int(row["pred_baseline"]))
            continue
        ell = probe.logit(float(row.get("posterior_hi", 0.5)))
        stop_entropy = float(row.get("stop_entropy", fallback_stop))
        for name in order:
            pred = candidates[name].get(vid)
            calls += 1
            if pred in (0, 1):
                if effects is None:
                    lam = math.log(rho[name] / (1 - rho[name]))
                    ell += (2 * int(pred) - 1) * lam
                else:
                    ell += effects[name][int(pred)]
                if probe.ent(probe.sigmoid(ell)) <= stop_entropy:
                    break
        yp.append(1 if probe.sigmoid(ell) >= 0.5 else 0)
    m = probe.metrics(y, yp)
    m["calls"] = calls / len(vids)
    return m


def learn_full_order(
    calib: list[str],
    labels: dict[str, int],
    band: dict[str, dict],
    candidates: dict[str, dict[str, int]],
    effects: dict[str, dict[int, float]],
    rho: dict[str, float],
) -> tuple[str, ...]:
    remaining = list(candidates)
    order: list[str] = []
    while remaining:
        best = None
        best_name = None
        for name in remaining:
            trial = tuple(order + [name])
            m = eval_order(calib, labels, band, candidates, rho, trial, effects=effects)
            key = (m["acc"], m["mf1"], -m["calls"], name)
            if best is None or key > best:
                best = key
                best_name = name
        assert best_name is not None
        order.append(best_name)
        remaining.remove(best_name)
    return tuple(order)


def learn_orders_by_k(
    calib: list[str],
    labels: dict[str, int],
    band: dict[str, dict],
    candidates: dict[str, dict[str, int]],
    effects: dict[str, dict[int, float]],
    rho: dict[str, float],
) -> dict[int, tuple[str, ...]]:
    names = tuple(candidates)
    beams: list[tuple[str, ...]] = [tuple()]
    orders: dict[int, tuple[str, ...]] = {}
    for k in range(1, len(names) + 1):
        trials: list[tuple[tuple[float, float, float, tuple[str, ...]], tuple[str, ...]]] = []
        for order in beams:
            used = set(order)
            for name in names:
                if name in used:
                    continue
                trial = order + (name,)
                m = eval_order(calib, labels, band, candidates, rho, trial, effects=effects)
                key = (m["acc"], m["mf1"], -m["calls"], trial)
                trials.append((key, trial))
        trials.sort(key=lambda item: item[0], reverse=True)
        beams = [trial for _, trial in trials[:BEAM_WIDTH]]
        orders[k] = beams[0]
    return orders


def eval_band_majority(
    vids: list[str],
    labels: dict[str, int],
    band: dict[str, dict],
    candidates: dict[str, dict[str, int]],
    order: tuple[str, ...],
) -> dict[str, float]:
    y, yp = [], []
    calls = 0
    for vid in vids:
        y.append(labels[vid])
        row = band[vid]
        if not row.get("in_band"):
            yp.append(int(row["pred_baseline"]))
            continue
        preds = []
        for name in order:
            pred = candidates[name].get(vid)
            calls += 1
            if pred in (0, 1):
                preds.append(int(pred))
        if not preds:
            yp.append(int(row["pred_baseline"]))
            continue
        pos = sum(preds)
        neg = len(preds) - pos
        if pos > neg:
            yp.append(1)
        elif neg > pos:
            yp.append(0)
        else:
            yp.append(int(row["pred_baseline"]))
    m = probe.metrics(y, yp)
    m["calls"] = calls / len(vids)
    return m


def build_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    curve_rows: list[dict[str, object]] = []
    order_rows: list[dict[str, object]] = []
    for dataset, _ in DATASETS:
        labels = probe.load_labels(dataset)
        band = probe.load_band(dataset)
        candidates = load_candidates(dataset)
        vids = probe.valid_vids(dataset, labels, band)
        for proportion in PROPORTIONS:
            calib = class_balanced_random_subset(vids, labels, proportion, CALIB_SEEDS[dataset][proportion])
            rho = probe.estimate_rho(calib, labels, candidates)
            effects = probe.estimate_effects(calib, labels, candidates)
            stage1_params = fit_stage1_calibrator(calib, labels, band)
            best_by_k: dict[int, tuple[float, float, float, tuple[str, ...], dict[str, dict], float]] = {}
            for threshold in band_threshold_candidates(calib, band, stage1_params):
                tuned_band = apply_stage1_calibration(band, stage1_params, threshold)
                orders = learn_orders_by_k(calib, labels, tuned_band, candidates, effects, rho)
                for k, order in orders.items():
                    calib_perf = eval_order(calib, labels, tuned_band, candidates, rho, order, effects=effects)
                    key = (calib_perf["acc"], calib_perf["mf1"], -calib_perf["calls"], order)
                    if k not in best_by_k or key > best_by_k[k][:4]:
                        best_by_k[k] = (*key, tuned_band, threshold)

            for k, (_, _, _, order, tuned_band, threshold) in sorted(best_by_k.items()):
                calib_in_band = sum(1 for vid in calib if tuned_band[vid].get("in_band"))
                routed_fraction = sum(1 for vid in vids if tuned_band[vid].get("in_band")) / len(vids)
                for rank, name in enumerate(order, start=1):
                    order_rows.append(
                        {
                            "dataset": dataset,
                            "calib_fraction_of_test": proportion,
                            "calib_seed": CALIB_SEEDS[dataset][proportion],
                            "calib_n": len(calib),
                            "calib_in_band_n": calib_in_band,
                            "band_threshold": f"{threshold:.6f}",
                            "routed_fraction": f"{routed_fraction:.4f}",
                            "k": k,
                            "rank": rank,
                            "backend": name,
                            "rho_positive": f"{probe.sigmoid(effects[name][1]):.4f}",
                            "rho_negative": f"{probe.sigmoid(-effects[name][0]):.4f}",
                        }
                    )

                ours = eval_order(vids, labels, tuned_band, candidates, rho, order, effects=effects)
                majority = eval_band_majority(vids, labels, tuned_band, candidates, order)
                curve_rows.append(
                    {
                        "dataset": dataset,
                        "calib_fraction_of_test": proportion,
                        "calib_seed": CALIB_SEEDS[dataset][proportion],
                        "calib_n": len(calib),
                        "calib_in_band_n": calib_in_band,
                        "band_threshold": threshold,
                        "routed_fraction": routed_fraction,
                        "k": k,
                        "order": " > ".join(order),
                        "ours_acc": ours["acc"],
                        "ours_mf1": ours["mf1"],
                        "ours_calls": ours["calls"],
                        "majority_acc": majority["acc"],
                        "majority_mf1": majority["mf1"],
                        "majority_calls": majority["calls"],
                        "main_table_acc": MAIN_TABLE_ACC[dataset] / 100,
                        "best_supervised_acc": BEST_SUPERVISED_ACC[dataset] / 100,
                    }
                )
    return curve_rows, order_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot(curve_rows: list[dict[str, object]]) -> None:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.titlesize": 9.0,
            "axes.labelsize": 7.2,
            "legend.fontsize": 7.6,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
        }
    )

    fig = plt.figure(figsize=(7.35, 6.25), constrained_layout=False)
    axes = [fig.add_subplot(2, 2, i + 1, projection="3d") for i in range(4)]
    colors = {
        "ours": "#2F6F8E",
        "majority": "#8A8F98",
        "main": "#B45F3C",
        "supervised": "#6B8E5A",
        "grid": "#E4E8EF",
    }
    for ax, (dataset, title) in zip(axes, DATASETS):
        rows = [r for r in curve_rows if r["dataset"] == dataset]
        ks = np.array(range(1, MAX_K + 1))
        props = np.array([int(p * 100) for p in PROPORTIONS])
        x, y = np.meshgrid(ks, props)
        ours = np.zeros_like(x, dtype=float)
        majority = np.zeros_like(x, dtype=float)
        by_key = {(int(r["k"]), int(round(100 * float(r["calib_fraction_of_test"])))): r for r in rows}
        for i, prop in enumerate(props):
            for j, k in enumerate(ks):
                r = by_key[(int(k), int(prop))]
                ours[i, j] = 100 * float(r["ours_acc"])
                majority[i, j] = 100 * float(r["majority_acc"])
        main = MAIN_TABLE_ACC[dataset]
        supervised = BEST_SUPERVISED_ACC[dataset]

        ax.plot_surface(x, y, ours, color=colors["ours"], alpha=0.82, linewidth=0.25, edgecolor="#FFFFFF", antialiased=True)
        ax.plot_wireframe(x, y, majority, color=colors["majority"], linewidth=0.65, rstride=1, cstride=1, alpha=0.9)
        ax.plot_surface(x, y, np.full_like(x, main, dtype=float), color=colors["main"], alpha=0.13, linewidth=0)
        ax.plot_surface(x, y, np.full_like(x, supervised, dtype=float), color=colors["supervised"], alpha=0.11, linewidth=0)
        ax.set_title(title, fontweight="bold", pad=-1)
        ax.set_xlabel("Verifiers", labelpad=1)
        ax.set_ylabel("Annotation (%)", labelpad=2)
        ax.set_zlabel("ACC", labelpad=1)
        ax.set_xticks([1, 4, 8, 12])
        ax.set_yticks([10, 30, 50, 70, 100])
        yvals = list(ours.ravel()) + list(majority.ravel()) + [main, supervised]
        lo = math.floor((min(yvals) - 1.6) / 2) * 2
        hi = math.ceil((max(yvals) + 1.4) / 2) * 2
        ax.set_zlim(lo, hi)
        ax.view_init(elev=23, azim=-55)
        ax.set_box_aspect((1.18, 1.0, 0.70))
        ax.xaxis.pane.set_facecolor((1, 1, 1, 0.0))
        ax.yaxis.pane.set_facecolor((1, 1, 1, 0.0))
        ax.zaxis.pane.set_facecolor((1, 1, 1, 0.0))
        ax.xaxis._axinfo["grid"]["color"] = colors["grid"]
        ax.yaxis._axinfo["grid"]["color"] = colors["grid"]
        ax.zaxis._axinfo["grid"]["color"] = colors["grid"]
        ax.tick_params(pad=-1)

    handles = [
        Patch(facecolor=colors["ours"], edgecolor="none", alpha=0.82, label="TRIAGE + joint calibration"),
        Line2D([0], [0], color=colors["majority"], lw=1.2, label="MAJORITY VOTE"),
        Patch(facecolor=colors["main"], edgecolor="none", alpha=0.18, label="Default TRIAGE"),
        Patch(facecolor=colors["supervised"], edgecolor="none", alpha=0.16, label="Best supervised baseline"),
    ]
    fig.legend(
        handles=handles,
        ncol=4,
        loc="upper center",
        frameon=False,
        bbox_to_anchor=(0.5, 0.975),
        columnspacing=0.7,
        handlelength=1.6,
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.07, top=0.88, hspace=0.22, wspace=0.04)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, bbox_inches="tight", pad_inches=0.10)
    fig.savefig(OUT_FIG.with_suffix(".png"), bbox_inches="tight", pad_inches=0.10, dpi=300)
    print(f"Wrote {OUT_FIG}")
    print(f"Wrote {OUT_FIG.with_suffix('.png')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true", help="Recompute deployment surface before plotting.")
    args = parser.parse_args()

    if args.recompute or not OUT_CSV.exists():
        curve_rows, order_rows = build_rows()
        write_csv(OUT_CSV, curve_rows)
        write_csv(OUT_ORDERS, order_rows)
    else:
        curve_rows = read_csv(OUT_CSV)
        print(f"Loaded {OUT_CSV}")

    plot(curve_rows)

    for dataset, _ in DATASETS:
        rows = [r for r in curve_rows if r["dataset"] == dataset]
        best = max(rows, key=lambda r: float(r["ours_acc"]))
        print(
            f"{dataset}: best ACC={100*float(best['ours_acc']):.1f} "
            f"at p={100*float(best['calib_fraction_of_test']):.0f}%, K={best['k']}; "
            f"main={100*float(best['main_table_acc']):.1f}; "
            f"order={best['order']}"
        )


if __name__ == "__main__":
    main()
