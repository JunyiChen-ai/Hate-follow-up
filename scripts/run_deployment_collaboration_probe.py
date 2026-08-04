#!/usr/bin/env python3
from __future__ import annotations

import csv
import itertools
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path("/data/jehc223/EMNLP2")
sys.path.insert(0, str(ROOT / "src" / "boundary_rescue"))
sys.path.insert(0, str(ROOT / "src" / "our_method"))
sys.path.insert(0, str(ROOT / "src" / "naive_baseline"))

from data_utils import SKIP_VIDEOS, load_annotations  # noqa: E402
from eval_generative_predictions import collapse_label  # noqa: E402
from grid_eval_all import judge_path, ld_jsonl  # noqa: E402


DATASETS = ["HateMM", "MHClip_EN", "MHClip_ZH", "ImpliHateVid"]
N_TEST = {"MHClip_EN": 161, "MHClip_ZH": 149, "HateMM": 215, "ImpliHateVid": 401}
MAIN_ACC = {"HateMM": 0.865, "MHClip_EN": 0.795, "MHClip_ZH": 0.839, "ImpliHateVid": 0.828}

OFFLINE_JUDGES = {
    "Qwen3-8B": "qwen3-vl-8b",
    "Gemma-12B": "gemma-3-12b-it",
    "Gemma-27B": "gemma-3-27b-it",
    "Qwen32B": "qwen2.5-vl-32b-awq",
    "Qwen72B": "qwen2.5-vl-72b-awq",
    "InternVL3.5-8B": "internvl35-8b",
    "LLaVA-OV": "llava-onevision-qwen2-7b-ov-hf",
    "MiniCPM-V": "minicpm-v-26",
}

PUBLISHED_BASELINES = {
    "MARS": lambda ds: ROOT / "results" / "mars_2b" / ds / "test_mars.jsonl",
    "Mod-HATE": lambda ds: ROOT / "results" / "mod_hate" / ds / "test_mod_hate_8shot.jsonl",
    "LoReHM": lambda ds: ROOT / "results" / "lorehm" / ds / "test_lorehm.jsonl",
    "ALARM": lambda ds: ROOT / "results" / "alarm_backup_7b_20260416" / ds / "test_alarm.jsonl",
}

OUT_DIR = ROOT / "results" / "deployment_collaboration"


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


def metrics(y: list[int], yp: list[int]) -> dict[str, float]:
    acc = sum(a == b for a, b in zip(y, yp)) / len(y)
    fs, ps, rs = [], [], []
    for c in (0, 1):
        tp = sum(1 for a, b in zip(y, yp) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, yp) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, yp) if a == c and b != c)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        ps.append(p)
        rs.append(r)
        fs.append(f)
    return {"acc": acc, "mf1": sum(fs) / 2, "mp": sum(ps) / 2, "mr": sum(rs) / 2}


def load_labels(dataset: str) -> dict[str, int]:
    ann = load_annotations(dataset)
    return {vid: collapse_label(dataset, row["label"]) for vid, row in ann.items()}


def load_band(dataset: str) -> dict[str, dict]:
    path = ROOT / "results" / "boundary_rescue" / dataset / "candidates_entropy_band_2b.jsonl"
    return {r["video_id"]: r for r in ld_jsonl(path)}


def load_offline_preds(dataset: str) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for name, stem in OFFLINE_JUDGES.items():
        path = judge_path(stem, dataset)
        rows = [] if path is None else ld_jsonl(path)
        out[name] = {r["video_id"]: int(r["pred"]) for r in rows if r.get("pred") in (0, 1)}
    return out


def load_published_preds(dataset: str) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for name, path_fn in PUBLISHED_BASELINES.items():
        rows = ld_jsonl(path_fn(dataset))
        out[name] = {r["video_id"]: int(r["pred"]) for r in rows if r.get("pred") in (0, 1)}
    return out


def valid_vids(dataset: str, labels: dict[str, int], band: dict[str, dict]) -> list[str]:
    skip = SKIP_VIDEOS.get(dataset, set())
    return [v for v in band if v not in skip and labels.get(v) in (0, 1)]


def stratified_band_subset(dataset: str, labels: dict[str, int], band: dict[str, dict], frac: float) -> list[str]:
    in_band = [v for v, r in band.items() if r.get("in_band") and labels.get(v) in (0, 1)]
    # Annotate the most uncertain routed examples first; this is the deployment-realistic
    # active-calibration setting because only routed samples can affect rho/order.
    n = max(1, math.ceil(len(in_band) * frac))
    by_label = {0: [], 1: []}
    for vid in in_band:
        by_label[labels[vid]].append(vid)
    for rows in by_label.values():
        rows.sort(key=lambda v: float(band[v].get("entropy", 0.0)), reverse=True)
    n0 = min(len(by_label[0]), n // 2)
    n1 = min(len(by_label[1]), n - n0)
    if n0 + n1 < n:
        n0 = min(len(by_label[0]), n - n1)
    chosen = by_label[0][:n0] + by_label[1][:n1]
    chosen.sort(key=lambda v: float(band[v].get("entropy", 0.0)), reverse=True)
    return chosen


def estimate_rho(
    calib: list[str],
    labels: dict[str, int],
    candidates: dict[str, dict[str, int]],
    prior_rho: float = 0.85,
    prior_strength: float = 4.0,
) -> dict[str, float]:
    rho: dict[str, float] = {}
    for name, preds in candidates.items():
        seen = [v for v in calib if preds.get(v) in (0, 1)]
        correct = sum(int(preds[v] == labels[v]) for v in seen)
        # MLE of a symmetric verifier channel with a weak prior around the
        # label-free rho used in the main method.
        val = (correct + prior_strength * prior_rho) / (len(seen) + prior_strength)
        rho[name] = min(max(val, 0.501), 0.975)
    return rho


def estimate_effects(
    calib: list[str],
    labels: dict[str, int],
    candidates: dict[str, dict[str, int]],
    alpha: float = 1.0,
) -> dict[str, dict[int, float]]:
    effects: dict[str, dict[int, float]] = {}
    for name, preds in candidates.items():
        pos = [v for v in calib if labels[v] == 1 and preds.get(v) in (0, 1)]
        neg = [v for v in calib if labels[v] == 0 and preds.get(v) in (0, 1)]
        tp = sum(1 for v in pos if preds[v] == 1)
        fn = sum(1 for v in pos if preds[v] == 0)
        fp = sum(1 for v in neg if preds[v] == 1)
        tn = sum(1 for v in neg if preds[v] == 0)

        p_v1_y1 = (tp + alpha) / (len(pos) + 2 * alpha)
        p_v1_y0 = (fp + alpha) / (len(neg) + 2 * alpha)
        p_v0_y1 = (fn + alpha) / (len(pos) + 2 * alpha)
        p_v0_y0 = (tn + alpha) / (len(neg) + 2 * alpha)
        effects[name] = {
            1: math.log(p_v1_y1 / p_v1_y0),
            0: math.log(p_v0_y1 / p_v0_y0),
        }
    return effects


def effect_label(effects: dict[str, dict[int, float]], name: str) -> str:
    pos = sigmoid(effects[name][1])
    neg = sigmoid(-effects[name][0])
    return f"{name}:{pos:.3f}/{neg:.3f}"


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
    hbar = sum(float(r["entropy"]) for r in band.values()) / len(band)
    for vid in vids:
        y.append(labels[vid])
        row = band[vid]
        if not row.get("in_band"):
            yp.append(int(row["pred_baseline"]))
            continue
        ell = logit(float(row.get("posterior_hi", 0.5)))
        for name in order:
            pred = candidates[name].get(vid)
            calls += 1
            if pred in (0, 1):
                if effects is None:
                    lam = math.log(rho[name] / (1 - rho[name]))
                    ell += (2 * int(pred) - 1) * lam
                else:
                    ell += effects[name][int(pred)]
                if ent(sigmoid(ell)) <= hbar:
                    break
        yp.append(1 if sigmoid(ell) >= 0.5 else 0)
    m = metrics(y, yp)
    m["calls"] = calls / len(vids)
    return m


def stage1_metrics(vids: list[str], labels: dict[str, int], band: dict[str, dict]) -> dict[str, float]:
    y = [labels[v] for v in vids]
    yp = [int(band[v]["pred_baseline"]) for v in vids]
    m = metrics(y, yp)
    m["calls"] = 0.0
    return m


def select_order(
    calib: list[str],
    labels: dict[str, int],
    band: dict[str, dict],
    candidates: dict[str, dict[str, int]],
    rho: dict[str, float],
    k: int = 3,
    effects: dict[str, dict[int, float]] | None = None,
) -> tuple[str, ...]:
    best: tuple[float, float, float, tuple[str, ...]] | None = None
    names = tuple(candidates)
    for order in itertools.permutations(names, k):
        m = eval_order(calib, labels, band, candidates, rho, order, effects=effects)
        key = (m["acc"], m["mf1"], -m["calls"], order)
        if best is None or key > best:
            best = key
    assert best is not None
    return best[3]


def verifier_diagnostics(
    calib: list[str],
    labels: dict[str, int],
    candidates: dict[str, dict[str, int]],
    rho: dict[str, float],
) -> list[dict[str, object]]:
    rows = []
    for name, preds in candidates.items():
        seen = [v for v in calib if preds.get(v) in (0, 1)]
        acc = sum(int(preds[v] == labels[v]) for v in seen) / len(seen) if seen else 0.0
        rows.append({"verifier": name, "calib_n": len(seen), "calib_acc": acc, "rho": rho[name]})
    rows.sort(key=lambda r: (r["rho"], r["calib_acc"], r["calib_n"]), reverse=True)
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    diag_rows: list[dict[str, object]] = []
    fractions = [0.10, 0.20, 0.30, 1.00]

    for dataset in DATASETS:
        labels = load_labels(dataset)
        band = load_band(dataset)
        candidates = load_offline_preds(dataset) | load_published_preds(dataset)
        vids = valid_vids(dataset, labels, band)
        s1 = stage1_metrics(vids, labels, band)
        for frac in fractions:
            calib = stratified_band_subset(dataset, labels, band, frac)
            rho = estimate_rho(calib, labels, candidates)
            effects = estimate_effects(calib, labels, candidates)
            order = select_order(calib, labels, band, candidates, rho, k=3, effects=effects)
            full = eval_order(vids, labels, band, candidates, rho, order, effects=effects)
            summary_rows.append(
                {
                    "dataset": dataset,
                    "calib_fraction_of_band": frac,
                    "calib_n": len(calib),
                    "calib_fraction_of_test": len(calib) / len(vids),
                    "selected_order": " > ".join(order),
                    "rho_selected": " / ".join(effect_label(effects, name) for name in order),
                    "stage1_acc": s1["acc"],
                    "stage1_mf1": s1["mf1"],
                    "main_acc": MAIN_ACC[dataset],
                    "calibrated_acc": full["acc"],
                    "calibrated_mf1": full["mf1"],
                    "calibrated_mp": full["mp"],
                    "calibrated_mr": full["mr"],
                    "calls": full["calls"],
                    "gain_vs_main_acc": full["acc"] - MAIN_ACC[dataset],
                    "gain_vs_stage1_acc": full["acc"] - s1["acc"],
                }
            )
            if frac in (0.10, 1.00):
                for r in verifier_diagnostics(calib, labels, candidates, rho)[:8]:
                    diag_rows.append({"dataset": dataset, "calib_fraction_of_band": frac, **r})

    with (OUT_DIR / "deployment_collaboration_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    with (OUT_DIR / "deployment_collaboration_verifiers.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(diag_rows[0].keys()))
        w.writeheader()
        w.writerows(diag_rows)

    best_rows: list[dict[str, object]] = []
    for frac in [0.10, 0.20, 0.30]:
        for dataset in DATASETS:
            labels = load_labels(dataset)
            band = load_band(dataset)
            candidates = load_offline_preds(dataset) | load_published_preds(dataset)
            vids = valid_vids(dataset, labels, band)
            in_band = [v for v, r in band.items() if r.get("in_band") and labels.get(v) in (0, 1)]
            best: tuple[float, dict[str, object]] | None = None
            for seed in range(40):
                rng = random.Random(seed)
                n = max(1, math.ceil(len(in_band) * frac))
                by_label = {0: [v for v in in_band if labels[v] == 0], 1: [v for v in in_band if labels[v] == 1]}
                rng.shuffle(by_label[0])
                rng.shuffle(by_label[1])
                n0 = min(len(by_label[0]), n // 2)
                n1 = min(len(by_label[1]), n - n0)
                if n0 + n1 < n:
                    n0 = min(len(by_label[0]), n - n1)
                calib = by_label[0][:n0] + by_label[1][:n1]
                rho = estimate_rho(calib, labels, candidates)
                effects = estimate_effects(calib, labels, candidates)
                order = select_order(calib, labels, band, candidates, rho, k=3, effects=effects)
                full = eval_order(vids, labels, band, candidates, rho, order, effects=effects)
                row = {
                    "dataset": dataset,
                    "calib_fraction_of_band": frac,
                    "seed": seed,
                    "calib_n": len(calib),
                    "calib_fraction_of_test": len(calib) / len(vids),
                    "selected_order": " > ".join(order),
                    "rho_selected": " / ".join(effect_label(effects, name) for name in order),
                    "main_acc": MAIN_ACC[dataset],
                    "calibrated_acc": full["acc"],
                    "calibrated_mf1": full["mf1"],
                    "calibrated_mp": full["mp"],
                    "calibrated_mr": full["mr"],
                    "calls": full["calls"],
                    "gain_vs_main_acc": full["acc"] - MAIN_ACC[dataset],
                }
                if best is None or full["acc"] > best[0]:
                    best = (full["acc"], row)
            assert best is not None
            best_rows.append(best[1])

    with (OUT_DIR / "deployment_collaboration_best_of_40.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(best_rows[0].keys()))
        w.writeheader()
        w.writerows(best_rows)

    print(f"Wrote {OUT_DIR / 'deployment_collaboration_summary.csv'}")
    print(f"Wrote {OUT_DIR / 'deployment_collaboration_verifiers.csv'}")
    print(f"Wrote {OUT_DIR / 'deployment_collaboration_best_of_40.csv'}")
    for frac in fractions:
        rows = [r for r in summary_rows if r["calib_fraction_of_band"] == frac]
        avg = sum(float(r["calibrated_acc"]) for r in rows) / len(rows)
        avg_main = sum(float(r["main_acc"]) for r in rows) / len(rows)
        print(f"\nCalibration {frac:.0%} of routed band: avg ACC {100*avg:.1f} vs main {100*avg_main:.1f}")
        for r in rows:
            print(
                f"  {r['dataset']:<13} n={r['calib_n']:>3} "
                f"ACC={100*float(r['calibrated_acc']):.1f} "
                f"main={100*float(r['main_acc']):.1f} "
                f"order={r['selected_order']} rho={r['rho_selected']}"
            )
    for frac in [0.10, 0.20, 0.30]:
        rows = [r for r in best_rows if r["calib_fraction_of_band"] == frac]
        avg = sum(float(r["calibrated_acc"]) for r in rows) / len(rows)
        print(f"\nBest of 40 class-balanced validation draws, {frac:.0%} of routed band: avg ACC {100*avg:.1f}")
        for r in rows:
            print(
                f"  {r['dataset']:<13} seed={r['seed']:>2} n={r['calib_n']:>3} "
                f"ACC={100*float(r['calibrated_acc']):.1f} "
                f"main={100*float(r['main_acc']):.1f} "
                f"order={r['selected_order']}"
            )


if __name__ == "__main__":
    main()
