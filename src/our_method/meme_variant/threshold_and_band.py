from __future__ import annotations

import argparse
import json
import math

import numpy as np
from sklearn.mixture import GaussianMixture

try:
    from .common import DATASETS, RESULT_ROOT, entropy, read_jsonl, write_jsonl
except ImportError:
    from common import DATASETS, RESULT_ROOT, entropy, read_jsonl, write_jsonl


def to_logit(scores: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    s = np.clip(np.asarray(scores, dtype=float), eps, 1.0 - eps)
    return np.log(s / (1.0 - s))


def otsu_threshold(scores: np.ndarray, nbins: int = 128) -> float:
    hist, edges = np.histogram(scores, bins=nbins, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = hist.sum()
    if total <= 0:
        return 0.5
    sum_total = float((hist * centers).sum())
    weight_bg = 0.0
    sum_bg = 0.0
    best_var = -1.0
    best_i = nbins // 2
    for i in range(nbins):
        weight_bg += hist[i]
        if weight_bg <= 0 or weight_bg >= total:
            continue
        sum_bg += hist[i] * centers[i]
        weight_fg = total - weight_bg
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > best_var:
            best_var = var_between
            best_i = i
    return float(centers[best_i])


def fit_gmm(scores: np.ndarray):
    z = to_logit(scores).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42, max_iter=300)
    gmm.fit(z)
    means = gmm.means_.flatten()
    hi_idx = int(np.argmax(means))
    lo_idx = 1 - hi_idx
    threshold_logit = float((means[hi_idx] + means[lo_idx]) / 2.0)
    threshold = 1.0 / (1.0 + math.exp(-threshold_logit))
    return gmm, hi_idx, threshold


def build_dataset(dataset: str, slug: str, criterion: str) -> dict:
    in_root = RESULT_ROOT / f"holistic_{slug}" / dataset
    train_rows = [r for r in read_jsonl(in_root / "train_binary.jsonl") if r.get("score") is not None]
    test_rows = [r for r in read_jsonl(in_root / "test_binary.jsonl") if r.get("score") is not None]
    if len(train_rows) < 4:
        raise RuntimeError(f"{dataset}: not enough train stage-1 scores at {in_root}")
    if not test_rows:
        raise RuntimeError(f"{dataset}: no test stage-1 scores at {in_root}")

    train_scores = np.array([float(r["score"]) for r in train_rows], dtype=float)
    test_scores = np.array([float(r["score"]) for r in test_rows], dtype=float)
    gmm, hi_idx, gmm_threshold = fit_gmm(train_scores)
    threshold = otsu_threshold(train_scores) if criterion == "otsu" else gmm_threshold
    posteriors = gmm.predict_proba(to_logit(test_scores).reshape(-1, 1))[:, hi_idx]
    entropies = np.array([entropy(p) for p in posteriors], dtype=float)
    hbar = float(entropies.mean())

    out_root = RESULT_ROOT / "boundary" / dataset
    base_rows = []
    band_rows = []
    for row, posterior, ent in zip(test_rows, posteriors, entropies):
        pred = int(float(row["score"]) >= threshold)
        base = {
            "id": row["id"],
            "dataset": dataset,
            "split": "test",
            "label": row["label"],
            "score": float(row["score"]),
            "threshold": float(threshold),
            "pred_baseline": pred,
            "posterior_hi": float(posterior),
            "entropy": float(ent),
            "model_slug": slug,
            "criterion": criterion,
        }
        base_rows.append(base)
        band_rows.append({**base, "in_band": bool(ent > hbar), "hbar": hbar})

    write_jsonl(out_root / "baseline_preds.jsonl", base_rows)
    write_jsonl(out_root / "candidates_entropy_band.jsonl", band_rows)
    summary = {
        "dataset": dataset,
        "criterion": criterion,
        "model_slug": slug,
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "threshold": float(threshold),
        "hbar": hbar,
        "n_band": int(sum(r["in_band"] for r in band_rows)),
        "gmm_means_logit": gmm.means_.flatten().tolist(),
        "gmm_weights": gmm.weights_.flatten().tolist(),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build baseline predictions and entropy band")
    parser.add_argument("--dataset", default="all", choices=("all", *DATASETS))
    parser.add_argument("--model-slug", default="2b")
    parser.add_argument("--criterion", default="gmm", choices=("gmm", "otsu"))
    args = parser.parse_args()
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    summaries = [build_dataset(ds, args.model_slug, args.criterion) for ds in datasets]
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

