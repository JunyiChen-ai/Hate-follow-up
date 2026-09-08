#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture


ROOT = Path("/data/jehc223/EMNLP3")
sys.path.insert(0, str(ROOT / "src" / "boundary_rescue"))
sys.path.insert(0, str(ROOT / "src" / "our_method"))
sys.path.insert(0, str(ROOT / "src" / "naive_baseline"))

from data_utils import SKIP_VIDEOS, load_annotations  # noqa: E402
from eval_generative_predictions import collapse_label  # noqa: E402
from grid_eval_all import judge_path, ld_jsonl  # noqa: E402
from quick_eval_all import load_scores_file  # noqa: E402
from thresholds import gmm_threshold, li_lee_threshold, otsu_threshold  # noqa: E402


DATASETS = ["HateMM", "MHClip_EN", "MHClip_ZH", "ImpliHateVid"]
JUDGES = ("gemma-3-27b-it", "qwen2.5-vl-32b-awq", "qwen2.5-vl-72b-awq")
PROPORTIONS = [0.10, 0.20, 0.40, 0.60, 0.80, 1.00]
N_POOL_SEEDS = 30
N_BOOT = 10000
OUT_DIR = ROOT / "results" / "boundary_rescue" / "reviewer_followups"

BASELINE_PATHS = {
    "HateMM": ("ALARM", ROOT / "results" / "alarm_backup_7b_20260416" / "HateMM" / "test_alarm.jsonl"),
    "MHClip_EN": ("LOREHM", ROOT / "results" / "lorehm" / "MHClip_EN" / "test_lorehm.jsonl"),
    "MHClip_ZH": ("QWEN3-VL-2B", ROOT / "results" / "naive_2b" / "MHClip_ZH" / "test_naive.jsonl"),
    "ImpliHateVid": ("MARS-32B", ROOT / "results" / "mars_32b_awq" / "ImpliHateVid" / "test_mars.jsonl"),
}
THRESHOLD_FNS = {"gmm": gmm_threshold, "li_lee": li_lee_threshold, "otsu": otsu_threshold}


def entropy(p: float) -> float:
    p = min(max(float(p), 1e-12), 1 - 1e-12)
    return -p * math.log(p) - (1 - p) * math.log(1 - p)


def logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
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


def metrics(y: list[int] | np.ndarray, yh: list[int] | np.ndarray) -> dict[str, float]:
    y = list(map(int, y))
    yh = list(map(int, yh))
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


def load_scores(dataset: str) -> list[dict[str, float | str]]:
    path = ROOT / "results" / "boundary_rescue" / dataset / "baseline_preds_v2.jsonl"
    rows = []
    for r in ld_jsonl(path):
        rows.append({"video_id": r["video_id"], "score": float(r["score"])})
    return rows


def load_main_protocol() -> dict[str, dict[str, object]]:
    return json.load(open(ROOT / "results" / "boundary_rescue" / "v2_baseline.json"))


def ordered_test_scores(dataset: str) -> list[dict[str, float | str]]:
    path = ROOT / "results" / "boundary_rescue" / dataset / "baseline_preds_v2.jsonl"
    return [{"video_id": r["video_id"], "score": float(r["score"])} for r in ld_jsonl(path)]


def load_default_base(dataset: str) -> dict[str, dict[str, float | int | bool]]:
    path = ROOT / "results" / "boundary_rescue" / dataset / "candidates_entropy_band_2b.jsonl"
    rows = {}
    for r in ld_jsonl(path):
        rows[r["video_id"]] = {
            "pred": int(r["pred_baseline"]),
            "posterior_hi": float(r["posterior_hi"]),
            "in_band": bool(r["in_band"]),
            "hbar": float(r["mean_entropy_dataset"]),
        }
    return rows


def load_judges(dataset: str) -> dict[str, dict[str, int]]:
    out = {}
    for judge in JUDGES:
        path = judge_path(judge, dataset)
        out[judge] = {r["video_id"]: int(r["pred"]) for r in ld_jsonl(path) if r.get("pred") in (0, 1)}
    return out


def fit_gmm(scores: np.ndarray) -> tuple[GaussianMixture, int]:
    z = np.asarray([logit(s) for s in scores], dtype=float).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42, max_iter=300, reg_covar=1e-6)
    gmm.fit(z)
    hi = int(np.argmax(gmm.means_.flatten()))
    return gmm, hi


def make_base_from_pool(test_rows, fit_scores, pool_indices, criterion) -> dict[str, dict[str, float | int | bool]]:
    test_scores = np.asarray([float(r["score"]) for r in test_rows], dtype=float)
    pool_scores = fit_scores[np.asarray(pool_indices, dtype=int)]
    threshold = THRESHOLD_FNS[criterion](pool_scores)
    gmm, hi = fit_gmm(pool_scores)
    all_z = np.asarray([logit(s) for s in test_scores], dtype=float).reshape(-1, 1)
    post = gmm.predict_proba(all_z)[:, hi]

    pool_z = np.asarray([logit(s) for s in pool_scores], dtype=float).reshape(-1, 1)
    pool_post = gmm.predict_proba(pool_z)[:, hi]
    hbar = float(np.mean([entropy(p) for p in pool_post]))

    base = {}
    for row, score, p in zip(test_rows, test_scores, post):
        h = entropy(float(p))
        base[row["video_id"]] = {
            "pred": int(float(score) >= threshold),
            "posterior_hi": float(p),
            "in_band": bool(h > hbar),
            "hbar": hbar,
        }
    return base


def run_triage(dataset: str, base, labels, judges) -> tuple[np.ndarray, np.ndarray, float]:
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
        for judge in JUDGES:
            pred = judges[judge].get(vid)
            if pred in (0, 1):
                calls += 1
                ell += (2 * pred - 1) * lam
                if entropy(sigmoid(ell)) <= hbar:
                    break
        yh.append(1 if sigmoid(ell) >= 0.5 else 0)
    return np.asarray(y), np.asarray(yh), calls / len(valid)


def load_baseline_preds(dataset: str, path: Path) -> dict[str, int]:
    preds = {}
    for r in ld_jsonl(path):
        if r.get("pred") in (0, 1):
            preds[r["video_id"]] = int(r["pred"])
        elif r.get("pred_baseline") in (0, 1):
            preds[r["video_id"]] = int(r["pred_baseline"])
    return preds


def run_pool_sensitivity() -> list[dict[str, object]]:
    rows = []
    protocol = load_main_protocol()
    for dataset in DATASETS:
        labels = load_labels(dataset)
        test_rows = [r for r in ordered_test_scores(dataset) if r["video_id"] not in SKIP_VIDEOS.get(dataset, set())]
        fit_scores = np.asarray(list(load_scores_file(protocol[dataset]["fit_path"]).values()), dtype=float)
        criterion = str(protocol[dataset]["criterion"])
        judges = load_judges(dataset)
        n = len(fit_scores)
        for prop in PROPORTIONS:
            seeds = [0] if prop >= 1.0 else list(range(N_POOL_SEEDS))
            for seed in seeds:
                rng = np.random.default_rng(seed)
                k = n if prop >= 1.0 else max(10, int(round(n * prop)))
                pool_idx = np.arange(n) if prop >= 1.0 else np.sort(rng.choice(n, size=min(k, n), replace=False))
                if prop >= 1.0:
                    base = load_default_base(dataset)
                else:
                    base = make_base_from_pool(test_rows, fit_scores, pool_idx, criterion)
                y, yh, calls = run_triage(dataset, base, labels, judges)
                m = metrics(y, yh)
                n_band = sum(1 for b in base.values() if b["in_band"])
                rows.append(
                    {
                        "dataset": dataset,
                        "pool_prop": prop,
                        "seed": seed,
                        "n_pool": len(pool_idx),
                        "n_eval": len(y),
                        "n_band": n_band,
                        "band_rate": n_band / len(y),
                        "calls_per_video": calls,
                        **m,
                    }
                )
    return rows


def summarize_pool(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for dataset in DATASETS:
        for prop in PROPORTIONS:
            vals = [r for r in rows if r["dataset"] == dataset and r["pool_prop"] == prop]
            for metric in ("acc", "mf1", "band_rate", "calls_per_video"):
                arr = np.asarray([float(r[metric]) for r in vals])
                if metric == "acc":
                    rec = {
                        "dataset": dataset,
                        "pool_prop": prop,
                        "n_pool_mean": float(np.mean([float(r["n_pool"]) for r in vals])),
                    }
                rec[f"{metric}_mean"] = float(arr.mean())
                rec[f"{metric}_std"] = float(arr.std(ddof=0))
            out.append(rec)
    return out


def run_bootstrap() -> list[dict[str, object]]:
    rows = []
    rng = np.random.default_rng(20260505)
    for dataset in DATASETS:
        labels = load_labels(dataset)
        judges = load_judges(dataset)
        base = load_default_base(dataset)
        y, triage_pred, _ = run_triage(dataset, base, labels, judges)
        valid = [v for v in base if v not in SKIP_VIDEOS.get(dataset, set()) and labels.get(v) in (0, 1)]

        baseline_name, baseline_path = BASELINE_PATHS[dataset]
        baseline_map = load_baseline_preds(dataset, baseline_path)
        keep = [i for i, v in enumerate(valid) if v in baseline_map]
        yy = y[keep]
        tt = triage_pred[keep]
        bb = np.asarray([baseline_map[valid[i]] for i in keep], dtype=int)

        triage_m = metrics(yy, tt)
        base_m = metrics(yy, bb)
        n = len(yy)
        acc_diffs, mf1_diffs = [], []
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, size=n)
            acc_diffs.append(metrics(yy[idx], tt[idx])["acc"] - metrics(yy[idx], bb[idx])["acc"])
            mf1_diffs.append(metrics(yy[idx], tt[idx])["mf1"] - metrics(yy[idx], bb[idx])["mf1"])
        for metric, diffs in [("acc", acc_diffs), ("mf1", mf1_diffs)]:
            arr = np.asarray(diffs, dtype=float)
            rows.append(
                {
                    "dataset": dataset,
                    "baseline": baseline_name,
                    "metric": metric,
                    "n": n,
                    "triage": triage_m[metric],
                    "baseline_value": base_m[metric],
                    "delta": triage_m[metric] - base_m[metric],
                    "ci_low": float(np.percentile(arr, 2.5)),
                    "ci_high": float(np.percentile(arr, 97.5)),
                    "win_rate": float(np.mean(arr > 0)),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pool_raw = run_pool_sensitivity()
    pool_summary = summarize_pool(pool_raw)
    boot = run_bootstrap()
    write_csv(OUT_DIR / "unlabeled_pool_sensitivity_raw.csv", pool_raw)
    write_csv(OUT_DIR / "unlabeled_pool_sensitivity_summary.csv", pool_summary)
    write_csv(OUT_DIR / "bootstrap_significance.csv", boot)

    print("UNLABELED POOL SENSITIVITY (ACC mean±std)")
    for dataset in DATASETS:
        vals = [r for r in pool_summary if r["dataset"] == dataset]
        print(dataset, " ".join(f"{int(r['pool_prop']*100)}%={r['acc_mean']*100:.1f}±{r['acc_std']*100:.1f}" for r in vals))

    print("\nBOOTSTRAP SIGNIFICANCE (delta in pp; win-rate)")
    for r in boot:
        if r["metric"] != "acc":
            continue
        print(
            f"{r['dataset']:<14} vs {r['baseline']:<12} "
            f"ACC Δ={r['delta']*100:+.1f} "
            f"CI=[{r['ci_low']*100:+.1f},{r['ci_high']*100:+.1f}] "
            f"win={r['win_rate']*100:.1f}%"
        )


if __name__ == "__main__":
    main()
