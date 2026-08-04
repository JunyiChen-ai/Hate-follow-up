#!/usr/bin/env python3
from __future__ import annotations

import csv
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path("/data/jehc223/EMNLP2")
sys.path.insert(0, str(ROOT / "src" / "boundary_rescue"))
sys.path.insert(0, str(ROOT / "src" / "our_method"))
sys.path.insert(0, str(ROOT / "src" / "naive_baseline"))

from data_utils import SKIP_VIDEOS, load_annotations  # noqa: E402
from eval_generative_predictions import collapse_label  # noqa: E402
from grid_eval_all import judge_path, ld_jsonl  # noqa: E402


DATASETS = ["HateMM", "MHClip_EN", "MHClip_ZH", "ImpliHateVid"]
JUDGES = [
    "gemma-3-27b-it",
    "qwen2.5-vl-32b-awq",
    "internvl35-8b",
    "qwen2.5-vl-72b-awq",
    "qwen3-vl-8b",
    "gemma-3-12b-it",
]
OUT_DIR = ROOT / "results" / "boundary_rescue" / "k_sensitivity"
RAW_OUT = OUT_DIR / "k_sensitivity_raw.csv"
SUMMARY_OUT = OUT_DIR / "k_sensitivity_summary.csv"


def entropy(p: float) -> float:
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


def rho_from_hbar(hbar: float) -> float:
    lo, hi = 0.5, 1 - 1e-12
    for _ in range(100):
        mid = (lo + hi) / 2
        if entropy(mid) > hbar:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def metrics(y: list[int], yh: list[int]) -> dict[str, float]:
    acc = sum(a == b for a, b in zip(y, yh)) / len(y)
    fs, ps, rs = [], [], []
    for c in (0, 1):
        tp = sum(1 for a, b in zip(y, yh) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, yh) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, yh) if a == c and b != c)
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


def load_base(dataset: str) -> dict[str, dict[str, float | int | bool]]:
    band_path = ROOT / "results" / "boundary_rescue" / dataset / "candidates_entropy_band_2b.jsonl"
    rows = {}
    with open(band_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rows[r["video_id"]] = {
                "pred": int(r["pred_baseline"]),
                "posterior_hi": float(r["posterior_hi"]),
                "in_band": bool(r["in_band"]),
                "hbar": float(r["mean_entropy_dataset"]),
            }
    return rows


def load_judges(dataset: str) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for judge in JUDGES:
        table = {}
        path = judge_path(judge, dataset)
        for r in ld_jsonl(path):
            if r.get("pred") in (0, 1):
                table[r["video_id"]] = int(r["pred"])
        out[judge] = table
    return out


def eval_order(dataset: str, order: tuple[str, ...], base, labels, judges) -> dict[str, float]:
    valid = [v for v in base if v not in SKIP_VIDEOS.get(dataset, set()) and labels.get(v) in (0, 1)]
    hbar = next(iter(base.values()))["hbar"]
    rho = rho_from_hbar(float(hbar))
    lam = math.log(rho / (1 - rho))
    y, yh = [], []
    calls = 0
    for vid in valid:
        y.append(labels[vid])
        b = base[vid]
        if not b["in_band"]:
            yh.append(int(b["pred"]))
            continue
        ell = logit(float(b["posterior_hi"]))
        for judge in order:
            pred = judges[judge].get(vid)
            if pred in (0, 1):
                calls += 1
                ell += (2 * pred - 1) * lam
                if entropy(sigmoid(ell)) <= hbar:
                    break
        yh.append(1 if sigmoid(ell) >= 0.5 else 0)
    m = metrics(y, yh)
    m["calls"] = calls / len(valid)
    return m


def eval_majority(dataset: str, judges_subset: tuple[str, ...], base, labels, judges) -> dict[str, float]:
    valid = [v for v in labels if v not in SKIP_VIDEOS.get(dataset, set()) and labels.get(v) in (0, 1)]
    y, yh = [], []
    calls = 0
    for vid in valid:
        preds = []
        for judge in judges_subset:
            pred = judges[judge].get(vid)
            if pred in (0, 1):
                preds.append(pred)
                calls += 1
        if not preds:
            if vid not in base:
                continue
            y.append(labels[vid])
            yh.append(int(base[vid]["pred"]))
            continue
        y.append(labels[vid])
        n_pos = sum(preds)
        n_neg = len(preds) - n_pos
        if n_pos > n_neg:
            yh.append(1)
        elif n_neg > n_pos:
            yh.append(0)
        else:
            yh.append(int(preds[0]))
    m = metrics(y, yh)
    m["calls"] = calls / len(valid)
    return m


def eval_band_majority(dataset: str, judges_subset: tuple[str, ...], base, labels, judges) -> dict[str, float]:
    valid = [v for v in base if v not in SKIP_VIDEOS.get(dataset, set()) and labels.get(v) in (0, 1)]
    y, yh = [], []
    calls = 0
    for vid in valid:
        y.append(labels[vid])
        b = base[vid]
        if not b["in_band"]:
            yh.append(int(b["pred"]))
            continue
        preds = []
        for judge in judges_subset:
            pred = judges[judge].get(vid)
            if pred in (0, 1):
                preds.append(pred)
                calls += 1
        if not preds:
            yh.append(int(b["pred"]))
            continue
        n_pos = sum(preds)
        n_neg = len(preds) - n_pos
        if n_pos > n_neg:
            yh.append(1)
        elif n_neg > n_pos:
            yh.append(0)
        else:
            yh.append(int(b["pred"]))
    m = metrics(y, yh)
    m["calls"] = calls / len(valid)
    return m


def trimmed_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    lo, hi = np.percentile(arr, [5, 95])
    kept = arr[(arr >= lo) & (arr <= hi)]
    return {
        "mean": float(kept.mean()),
        "std": float(kept.std(ddof=0)),
        "median": float(np.median(kept)),
        "p25": float(np.percentile(kept, 25)),
        "p75": float(np.percentile(kept, 75)),
        "p10": float(np.percentile(kept, 10)),
        "p90": float(np.percentile(kept, 90)),
        "min": float(kept.min()),
        "max": float(kept.max()),
        "n": int(len(kept)),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for dataset in DATASETS:
        base = load_base(dataset)
        labels = load_labels(dataset)
        judges = load_judges(dataset)
        for k in range(1, len(JUDGES) + 1):
            triage_acc, triage_mf1 = [], []
            maj_acc, maj_mf1 = [], []
            band_maj_acc, band_maj_mf1 = [], []
            pool = tuple(JUDGES[:k])
            default_order_metrics = None
            default_majority_metrics = None
            default_band_majority_metrics = None
            for order in itertools.permutations(JUDGES, k):
                tm = eval_order(dataset, order, base, labels, judges)
                raw_rows.append(
                    {
                        "dataset": dataset,
                        "k": k,
                        "method": "triage",
                        "order": ">".join(order),
                        **tm,
                    }
                )
                triage_acc.append(tm["acc"])
                triage_mf1.append(tm["mf1"])
                if order == pool:
                    default_order_metrics = tm
            for subset in itertools.combinations(JUDGES, k):
                mm = eval_majority(dataset, subset, base, labels, judges)
                raw_rows.append(
                    {
                        "dataset": dataset,
                        "k": k,
                        "method": "majority",
                        "order": ">".join(subset),
                        **mm,
                    }
                )
                maj_acc.append(mm["acc"])
                maj_mf1.append(mm["mf1"])
                bm = eval_band_majority(dataset, subset, base, labels, judges)
                raw_rows.append(
                    {
                        "dataset": dataset,
                        "k": k,
                        "method": "band_majority",
                        "order": ">".join(subset),
                        **bm,
                    }
                )
                band_maj_acc.append(bm["acc"])
                band_maj_mf1.append(bm["mf1"])
                if subset == pool:
                    default_majority_metrics = mm
                    default_band_majority_metrics = bm

            for method, accs, mf1s in [
                ("triage", triage_acc, triage_mf1),
                ("majority", maj_acc, maj_mf1),
                ("band_majority", band_maj_acc, band_maj_mf1),
            ]:
                acc_stat = trimmed_stats(accs)
                mf1_stat = trimmed_stats(mf1s)
                if method == "triage":
                    center = default_order_metrics
                elif method == "majority":
                    center = default_majority_metrics
                else:
                    center = default_band_majority_metrics
                summary_rows.append(
                    {
                        "dataset": dataset,
                        "k": k,
                        "method": method,
                        "acc_mean": acc_stat["mean"],
                        "acc_std": acc_stat["std"],
                        "acc_default": center["acc"],
                        "acc_median": acc_stat["median"],
                        "acc_p25": acc_stat["p25"],
                        "acc_p75": acc_stat["p75"],
                        "acc_p10": acc_stat["p10"],
                        "acc_p90": acc_stat["p90"],
                        "mf1_mean": mf1_stat["mean"],
                        "mf1_std": mf1_stat["std"],
                        "mf1_default": center["mf1"],
                        "mf1_median": mf1_stat["median"],
                        "mf1_p25": mf1_stat["p25"],
                        "mf1_p75": mf1_stat["p75"],
                        "mf1_p10": mf1_stat["p10"],
                        "mf1_p90": mf1_stat["p90"],
                        "n_kept": acc_stat["n"],
                    }
                )

    with open(RAW_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)
    with open(SUMMARY_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote {RAW_OUT}")
    print(f"Wrote {SUMMARY_OUT}")
    for dataset in DATASETS:
        vals = [r for r in summary_rows if r["dataset"] == dataset and r["method"] == "triage"]
        print(dataset, "triage ACC:", " ".join(f"K{r['k']}={r['acc_default']*100:.1f}" for r in vals))


if __name__ == "__main__":
    main()
