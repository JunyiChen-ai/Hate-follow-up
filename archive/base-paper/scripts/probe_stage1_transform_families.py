#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import sys
import argparse
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import beta as beta_dist
from scipy.stats import t as student_t_dist
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.neighbors import KernelDensity

ROOT = Path("/data/jehc223/EMNLP3")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src" / "boundary_rescue"))

import run_deployment_collaboration_probe as probe  # noqa: E402
from thresholds import li_lee_threshold, load_scores_file, otsu_threshold  # noqa: E402


DEFAULT_ORDER = ("Gemma-27B", "Qwen32B", "InternVL3.5-8B")
OUT_DIR = ROOT / "results" / "boundary_rescue" / "stage1_transform_families"
OUT_CSV = OUT_DIR / "stage1_transform_family_probe.csv"


def entropy(p: np.ndarray | float) -> np.ndarray | float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -p * np.log(p) - (1 - p) * np.log(1 - p)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def edge_from_entropy(hbar: float) -> float:
    lo, hi = 0.5, 1.0 - 1e-9
    for _ in range(80):
        mid = (lo + hi) / 2
        if float(entropy(mid)) > hbar:
            lo = mid
        else:
            hi = mid
    return hi


def fit_gaussian_mixture(x: np.ndarray) -> np.ndarray:
    gmm = GaussianMixture(
        n_components=2, random_state=42, max_iter=500, reg_covar=1e-6
    ).fit(x.reshape(-1, 1))
    hi = int(np.argmax(gmm.means_.reshape(-1)))
    return gmm.predict_proba(x.reshape(-1, 1))[:, hi]


def fit_bayesian_gaussian_mixture(x: np.ndarray) -> np.ndarray:
    bgmm = BayesianGaussianMixture(
        n_components=2,
        random_state=42,
        max_iter=1000,
        reg_covar=1e-6,
        weight_concentration_prior_type="dirichlet_distribution",
    ).fit(x.reshape(-1, 1))
    hi = int(np.argmax(bgmm.means_.reshape(-1)))
    return bgmm.predict_proba(x.reshape(-1, 1))[:, hi]


def fit_kde_mixture(x: np.ndarray) -> np.ndarray:
    x2 = x.reshape(-1, 1)
    labels = KMeans(n_clusters=2, n_init=20, random_state=42).fit_predict(x2)
    centers = np.array([x[labels == k].mean() for k in range(2)])
    hi, lo = int(np.argmax(centers)), int(np.argmin(centers))
    log_terms = []
    for k in (lo, hi):
        vals = x[labels == k]
        if len(vals) < 3:
            return fit_gaussian_mixture(x)
        bandwidth = max(0.05, 1.06 * float(vals.std() + 1e-6) * len(vals) ** (-0.2))
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(vals.reshape(-1, 1))
        log_weight = math.log(len(vals) / len(x))
        log_terms.append(log_weight + kde.score_samples(x2))
    stacked = np.vstack(log_terms)
    return np.exp(stacked[1] - logsumexp(stacked, axis=0))


def weighted_median(x: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(x)
    xs = x[order]
    ws = weights[order]
    cutoff = 0.5 * float(ws.sum())
    return float(xs[np.searchsorted(np.cumsum(ws), cutoff, side="left")])


def beta_moments(x: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    wsum = float(weights.sum()) + 1e-12
    mean = float(np.sum(weights * x) / wsum)
    var = float(np.sum(weights * (x - mean) ** 2) / wsum)
    mean = min(max(mean, 1e-4), 1 - 1e-4)
    max_var = mean * (1 - mean) * 0.95
    var = min(max(var, 1e-5), max_var)
    common = mean * (1 - mean) / var - 1
    return max(0.2, mean * common), max(0.2, (1 - mean) * common)


def fit_beta_mixture(
    fit_scores: np.ndarray, eval_scores: np.ndarray
) -> np.ndarray:
    x = np.clip(fit_scores, 1e-5, 1 - 1e-5)
    y = np.clip(eval_scores, 1e-5, 1 - 1e-5)
    labels = KMeans(n_clusters=2, n_init=20, random_state=42).fit_predict(
        x.reshape(-1, 1)
    )
    weights = np.array([(labels == k).mean() for k in range(2)], dtype=float)
    params = [beta_moments(x, (labels == k).astype(float) + 1e-3) for k in range(2)]
    for _ in range(100):
        logp = np.vstack(
            [
                math.log(weights[k] + 1e-12)
                + beta_dist.logpdf(x, params[k][0], params[k][1])
                for k in range(2)
            ]
        )
        resp = np.exp(logp - logsumexp(logp, axis=0))
        weights = resp.mean(axis=1)
        params = [beta_moments(x, resp[k]) for k in range(2)]
    means = np.array([a / (a + b) for a, b in params])
    hi = int(np.argmax(means))
    logp_eval = np.vstack(
        [
            math.log(weights[k] + 1e-12)
            + beta_dist.logpdf(y, params[k][0], params[k][1])
            for k in range(2)
        ]
    )
    resp_eval = np.exp(logp_eval - logsumexp(logp_eval, axis=0))
    return resp_eval[hi]


def fit_student_t_mixture(
    fit_logits: np.ndarray, eval_logits: np.ndarray, df: float = 4.0
) -> np.ndarray:
    x = np.asarray(fit_logits, dtype=float)
    labels = KMeans(n_clusters=2, n_init=20, random_state=42).fit_predict(
        x.reshape(-1, 1)
    )
    weights = np.array([(labels == k).mean() for k in range(2)], dtype=float)
    locs = np.array([x[labels == k].mean() for k in range(2)], dtype=float)
    scales = np.array([max(0.05, x[labels == k].std()) for k in range(2)], dtype=float)
    for _ in range(100):
        logp = np.vstack(
            [
                math.log(weights[k] + 1e-12)
                + student_t_dist.logpdf(x, df=df, loc=locs[k], scale=scales[k])
                for k in range(2)
            ]
        )
        resp = np.exp(logp - logsumexp(logp, axis=0))
        weights = resp.mean(axis=1)
        for k in range(2):
            delta = ((x - locs[k]) / scales[k]) ** 2
            u = (df + 1.0) / (df + delta)
            denom = float(np.sum(resp[k] * u)) + 1e-12
            locs[k] = float(np.sum(resp[k] * u * x) / denom)
            var = float(np.sum(resp[k] * u * (x - locs[k]) ** 2) / (resp[k].sum() + 1e-12))
            scales[k] = math.sqrt(max(var, 0.05**2))
    hi = int(np.argmax(locs))
    logp_eval = np.vstack(
        [
            math.log(weights[k] + 1e-12)
            + student_t_dist.logpdf(eval_logits, df=df, loc=locs[k], scale=scales[k])
            for k in range(2)
        ]
    )
    resp_eval = np.exp(logp_eval - logsumexp(logp_eval, axis=0))
    return resp_eval[hi]


def fit_laplace_mixture(fit_logits: np.ndarray, eval_logits: np.ndarray) -> np.ndarray:
    x = np.asarray(fit_logits, dtype=float)
    labels = KMeans(n_clusters=2, n_init=20, random_state=42).fit_predict(
        x.reshape(-1, 1)
    )
    weights = np.array([(labels == k).mean() for k in range(2)], dtype=float)
    locs = np.array([np.median(x[labels == k]) for k in range(2)], dtype=float)
    scales = np.array(
        [max(0.05, np.mean(np.abs(x[labels == k] - locs[k]))) for k in range(2)],
        dtype=float,
    )
    for _ in range(100):
        logp = np.vstack(
            [
                math.log(weights[k] + 1e-12)
                - math.log(2 * scales[k])
                - np.abs(x - locs[k]) / scales[k]
                for k in range(2)
            ]
        )
        resp = np.exp(logp - logsumexp(logp, axis=0))
        weights = resp.mean(axis=1)
        for k in range(2):
            locs[k] = weighted_median(x, resp[k])
            scales[k] = max(0.05, float(np.sum(resp[k] * np.abs(x - locs[k])) / (resp[k].sum() + 1e-12)))
    hi = int(np.argmax(locs))
    logp_eval = np.vstack(
        [
            math.log(weights[k] + 1e-12)
            - math.log(2 * scales[k])
            - np.abs(eval_logits - locs[k]) / scales[k]
            for k in range(2)
        ]
    )
    resp_eval = np.exp(logp_eval - logsumexp(logp_eval, axis=0))
    return resp_eval[hi]


def fit_histogram_mixture(fit_logits: np.ndarray, eval_logits: np.ndarray) -> np.ndarray:
    x = np.asarray(fit_logits, dtype=float)
    labels = KMeans(n_clusters=2, n_init=20, random_state=42).fit_predict(
        x.reshape(-1, 1)
    )
    edges = np.histogram_bin_edges(x, bins="fd")
    if len(edges) < 8:
        edges = np.linspace(x.min(), x.max(), 16)
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    weights = np.array([(labels == k).mean() for k in range(2)], dtype=float)
    centers = np.array([x[labels == k].mean() for k in range(2)])
    hi = int(np.argmax(centers))
    log_terms = []
    bins = np.clip(np.searchsorted(edges, eval_logits, side="right") - 1, 0, len(edges) - 2)
    widths = np.diff(edges)
    for k in range(2):
        counts, _ = np.histogram(x[labels == k], bins=edges)
        probs = (counts + 1.0) / (counts.sum() + len(counts))
        density = probs[bins] / widths[bins]
        log_terms.append(math.log(weights[k] + 1e-12) + np.log(density + 1e-12))
    stacked = np.vstack(log_terms)
    resp = np.exp(stacked - logsumexp(stacked, axis=0))
    return resp[hi]


def threshold_sigmoid(scores: np.ndarray, threshold: float) -> np.ndarray:
    # This is intentionally only a control: thresholding criteria do not define
    # a posterior, so the temperature must be supplied externally.
    scale = max(0.02, float(np.std(scores)) / 3.0)
    return sigmoid((scores - threshold) / scale)


def threshold_partition_gaussian_calibration(
    fit_scores: np.ndarray,
    eval_scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    x = to_logit(fit_scores)
    y = to_logit(eval_scores)
    low = x[fit_scores < threshold]
    high = x[fit_scores >= threshold]
    if len(low) < 3 or len(high) < 3:
        return threshold_sigmoid(eval_scores, threshold)
    priors = np.array([len(low), len(high)], dtype=float) / (len(low) + len(high))
    means = np.array([float(low.mean()), float(high.mean())], dtype=float)
    stds = np.array([max(0.05, float(low.std())), max(0.05, float(high.std()))])
    hi = int(np.argmax(means))
    logp = np.vstack(
        [
            math.log(priors[k] + 1e-12)
            - np.log(stds[k])
            - 0.5 * ((y - means[k]) / stds[k]) ** 2
            for k in range(2)
        ]
    )
    resp = np.exp(logp - logsumexp(logp, axis=0))
    return resp[hi]


def threshold_partition_laplace_calibration(
    fit_scores: np.ndarray,
    eval_scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    x = to_logit(fit_scores)
    y = to_logit(eval_scores)
    low = x[fit_scores < threshold]
    high = x[fit_scores >= threshold]
    if len(low) < 3 or len(high) < 3:
        return threshold_sigmoid(eval_scores, threshold)
    priors = np.array([len(low), len(high)], dtype=float) / (len(low) + len(high))
    locs = np.array([float(np.median(low)), float(np.median(high))], dtype=float)
    scales = np.array(
        [
            max(0.05, float(np.mean(np.abs(low - locs[0])))),
            max(0.05, float(np.mean(np.abs(high - locs[1])))),
        ]
    )
    hi = int(np.argmax(locs))
    logp = np.vstack(
        [
            math.log(priors[k] + 1e-12)
            - math.log(2 * scales[k])
            - np.abs(y - locs[k]) / scales[k]
            for k in range(2)
        ]
    )
    resp = np.exp(logp - logsumexp(logp, axis=0))
    return resp[hi]


def threshold_partition_kde_calibration(
    fit_scores: np.ndarray,
    eval_scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    x = to_logit(fit_scores)
    y = to_logit(eval_scores)
    groups = [x[fit_scores < threshold], x[fit_scores >= threshold]]
    if min(len(g) for g in groups) < 3:
        return threshold_sigmoid(eval_scores, threshold)
    priors = np.array([len(g) for g in groups], dtype=float) / len(x)
    means = np.array([float(g.mean()) for g in groups])
    hi = int(np.argmax(means))
    log_terms = []
    for k, vals in enumerate(groups):
        bandwidth = max(0.05, 1.06 * float(vals.std() + 1e-6) * len(vals) ** (-0.2))
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(
            vals.reshape(-1, 1)
        )
        log_terms.append(
            math.log(priors[k] + 1e-12) + kde.score_samples(y.reshape(-1, 1))
        )
    stacked = np.vstack(log_terms)
    resp = np.exp(stacked - logsumexp(stacked, axis=0))
    return resp[hi]


def load_rows(dataset: str) -> tuple[dict[str, dict], list[str], dict[str, int]]:
    labels = probe.load_labels(dataset)
    band = probe.load_band(dataset)
    vids = probe.valid_vids(dataset, labels, band)
    return band, vids, labels


def build_band(
    base_band: dict[str, dict],
    q_by_vid: dict[str, float],
    pred_threshold: float = 0.5,
    pred_by_vid: dict[str, int] | None = None,
) -> dict[str, dict]:
    entropies = np.array([float(entropy(q)) for q in q_by_vid.values()], dtype=float)
    hbar = float(entropies.mean())
    tuned: dict[str, dict] = {}
    for vid, old in base_band.items():
        q = float(q_by_vid[vid])
        e = float(entropy(q))
        row = dict(old)
        row["posterior_hi"] = q
        row["entropy"] = e
        row["mean_entropy_dataset"] = hbar
        row["in_band"] = bool(e > hbar)
        if pred_by_vid is None:
            row["pred_baseline"] = int(q >= pred_threshold)
        else:
            row["pred_baseline"] = int(pred_by_vid[vid])
        row["side"] = "above" if row["pred_baseline"] == 1 else "below"
        tuned[vid] = row
    return tuned


def band_diagnostics(
    vids: list[str], labels: dict[str, int], band: dict[str, dict]
) -> dict[str, float]:
    total_err = 0
    captured_err = 0
    n_band = 0
    for vid in vids:
        err = int(int(band[vid]["pred_baseline"]) != labels[vid])
        routed = bool(band[vid]["in_band"])
        total_err += err
        captured_err += int(err and routed)
        n_band += int(routed)
    return {
        "band_rate": n_band / len(vids),
        "stage1_errors": total_err,
        "error_capture": captured_err / total_err if total_err else 0.0,
    }


def to_logit(scores: np.ndarray) -> np.ndarray:
    s = np.clip(np.asarray(scores, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(s / (1.0 - s))


def transform_scores(
    method: str, fit_scores: np.ndarray, eval_scores: np.ndarray
) -> tuple[np.ndarray, float]:
    fit_logits = to_logit(fit_scores)
    eval_logits = to_logit(eval_scores)
    if method == "GMM-logit":
        gmm = GaussianMixture(
            n_components=2, random_state=42, max_iter=500, reg_covar=1e-6
        ).fit(fit_logits.reshape(-1, 1))
        hi = int(np.argmax(gmm.means_.reshape(-1)))
        return gmm.predict_proba(eval_logits.reshape(-1, 1))[:, hi], 0.5
    if method == "GMMpost-Otsu-threshold":
        gmm = GaussianMixture(
            n_components=2, random_state=42, max_iter=500, reg_covar=1e-6
        ).fit(fit_logits.reshape(-1, 1))
        hi = int(np.argmax(gmm.means_.reshape(-1)))
        fit_q = gmm.predict_proba(fit_logits.reshape(-1, 1))[:, hi]
        eval_q = gmm.predict_proba(eval_logits.reshape(-1, 1))[:, hi]
        return eval_q, otsu_threshold(fit_q)
    if method == "GMMpost-LiLee-threshold":
        gmm = GaussianMixture(
            n_components=2, random_state=42, max_iter=500, reg_covar=1e-6
        ).fit(fit_logits.reshape(-1, 1))
        hi = int(np.argmax(gmm.means_.reshape(-1)))
        fit_q = gmm.predict_proba(fit_logits.reshape(-1, 1))[:, hi]
        eval_q = gmm.predict_proba(eval_logits.reshape(-1, 1))[:, hi]
        return eval_q, li_lee_threshold(fit_q)
    if method == "BayesianGMM-logit":
        bgmm = BayesianGaussianMixture(
            n_components=2,
            random_state=42,
            max_iter=1000,
            reg_covar=1e-6,
            weight_concentration_prior_type="dirichlet_distribution",
        ).fit(fit_logits.reshape(-1, 1))
        hi = int(np.argmax(bgmm.means_.reshape(-1)))
        return bgmm.predict_proba(eval_logits.reshape(-1, 1))[:, hi], 0.5
    if method == "StudentT-mixture-logit":
        return fit_student_t_mixture(fit_logits, eval_logits), 0.5
    if method == "Laplace-mixture-logit":
        return fit_laplace_mixture(fit_logits, eval_logits), 0.5
    if method == "KDE-mixture-logit":
        x = fit_logits
        labels = KMeans(n_clusters=2, n_init=20, random_state=42).fit_predict(
            x.reshape(-1, 1)
        )
        centers = np.array([x[labels == k].mean() for k in range(2)])
        hi, lo = int(np.argmax(centers)), int(np.argmin(centers))
        log_terms = []
        for k in (lo, hi):
            vals = x[labels == k]
            if len(vals) < 3:
                return transform_scores("GMM-logit", fit_scores, eval_scores)
            bandwidth = max(0.05, 1.06 * float(vals.std() + 1e-6) * len(vals) ** (-0.2))
            kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(
                vals.reshape(-1, 1)
            )
            log_weight = math.log(len(vals) / len(x))
            log_terms.append(log_weight + kde.score_samples(eval_logits.reshape(-1, 1)))
        stacked = np.vstack(log_terms)
        return np.exp(stacked[1] - logsumexp(stacked, axis=0)), 0.5
    if method == "Histogram-mixture-logit":
        return fit_histogram_mixture(fit_logits, eval_logits), 0.5
    if method == "GMM-raw":
        gmm = GaussianMixture(
            n_components=2, random_state=42, max_iter=500, reg_covar=1e-6
        ).fit(fit_scores.reshape(-1, 1))
        hi = int(np.argmax(gmm.means_.reshape(-1)))
        return gmm.predict_proba(eval_scores.reshape(-1, 1))[:, hi], 0.5
    if method == "Beta-mixture-raw":
        return fit_beta_mixture(fit_scores, eval_scores), 0.5
    if method == "Otsu-GaussianCal-logit":
        threshold = otsu_threshold(fit_scores)
        return threshold_partition_gaussian_calibration(fit_scores, eval_scores, threshold), 0.5
    if method == "LiLee-GaussianCal-logit":
        threshold = li_lee_threshold(fit_scores)
        return threshold_partition_gaussian_calibration(fit_scores, eval_scores, threshold), 0.5
    if method == "Otsu-LaplaceCal-logit":
        threshold = otsu_threshold(fit_scores)
        return threshold_partition_laplace_calibration(fit_scores, eval_scores, threshold), 0.5
    if method == "LiLee-LaplaceCal-logit":
        threshold = li_lee_threshold(fit_scores)
        return threshold_partition_laplace_calibration(fit_scores, eval_scores, threshold), 0.5
    if method == "Otsu-KDECal-logit":
        threshold = otsu_threshold(fit_scores)
        return threshold_partition_kde_calibration(fit_scores, eval_scores, threshold), 0.5
    if method == "LiLee-KDECal-logit":
        threshold = li_lee_threshold(fit_scores)
        return threshold_partition_kde_calibration(fit_scores, eval_scores, threshold), 0.5
    if method == "Otsu-sigmoid-control":
        return threshold_sigmoid(eval_scores, otsu_threshold(fit_scores)), 0.5
    if method == "LiLee-sigmoid-control":
        return threshold_sigmoid(eval_scores, li_lee_threshold(fit_scores)), 0.5
    raise ValueError(method)


def gmm_posterior_from_fit(fit_scores: np.ndarray, eval_scores: np.ndarray) -> np.ndarray:
    fit_logits = to_logit(fit_scores)
    eval_logits = to_logit(eval_scores)
    gmm = GaussianMixture(
        n_components=2, random_state=42, max_iter=500, reg_covar=1e-6
    ).fit(fit_logits.reshape(-1, 1))
    hi = int(np.argmax(gmm.means_.reshape(-1)))
    return gmm.predict_proba(eval_logits.reshape(-1, 1))[:, hi]


def get_fit_scores(dataset: str, source: str, eval_scores: np.ndarray) -> np.ndarray:
    if source == "train set":
        return np.array(
            list(
                load_scores_file(
                    ROOT / "results" / "holistic_2b" / dataset / "train_binary.jsonl"
                ).values()
            ),
            dtype=float,
        )
    if source == "test set":
        return np.asarray(eval_scores, dtype=float)
    raise ValueError(source)


def run_one(dataset: str, method: str, source: str) -> dict[str, object]:
    base_band, vids, labels = load_rows(dataset)
    ordered_vids = list(base_band)
    scores = np.array([float(base_band[v]["score"]) for v in ordered_vids], dtype=float)
    fit_scores = get_fit_scores(dataset, source, scores)
    pred_by_vid = None
    if method == "Otsu-initial-threshold":
        q = gmm_posterior_from_fit(fit_scores, scores)
        pred_threshold = otsu_threshold(fit_scores)
        pred_by_vid = {
            vid: int(scores[i] >= pred_threshold) for i, vid in enumerate(ordered_vids)
        }
    elif method == "LiLee-initial-threshold":
        q = gmm_posterior_from_fit(fit_scores, scores)
        pred_threshold = li_lee_threshold(fit_scores)
        pred_by_vid = {
            vid: int(scores[i] >= pred_threshold) for i, vid in enumerate(ordered_vids)
        }
    else:
        q, pred_threshold = transform_scores(method, fit_scores, scores)
    q = np.clip(q, 1e-6, 1 - 1e-6)
    q_by_vid = {vid: float(q[i]) for i, vid in enumerate(ordered_vids)}
    band = build_band(
        base_band, q_by_vid, pred_threshold=pred_threshold, pred_by_vid=pred_by_vid
    )

    stage = probe.stage1_metrics(vids, labels, band)
    diag = band_diagnostics(vids, labels, band)
    candidates = probe.load_offline_preds(dataset)
    hbar = sum(float(row["entropy"]) for row in band.values()) / len(band)
    edge = edge_from_entropy(hbar)
    rho = {name: edge for name in candidates}
    full = probe.eval_order(vids, labels, band, candidates, rho, DEFAULT_ORDER)

    return {
        "dataset": dataset,
        "method": method,
        "source": source,
        "n": len(vids),
        "band_rate": diag["band_rate"],
        "error_capture": diag["error_capture"],
        "stage1_errors": diag["stage1_errors"],
        "stage1_acc": stage["acc"],
        "stage1_mf1": stage["mf1"],
        "stage1_mp": stage["mp"],
        "stage1_mr": stage["mr"],
        "full_acc": full["acc"],
        "full_mf1": full["mf1"],
        "full_mp": full["mp"],
        "full_mr": full["mr"],
        "avg_calls": full["calls"],
        "rho_edge": edge,
        "pred_threshold": pred_threshold,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        choices=["train", "test", "both"],
        default="both",
        help="Unlabeled score source used to fit the Stage-1 transform.",
    )
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    methods = [
        "GMM-logit",
        "GMMpost-Otsu-threshold",
        "GMMpost-LiLee-threshold",
        "Otsu-initial-threshold",
        "LiLee-initial-threshold",
        "BayesianGMM-logit",
        "StudentT-mixture-logit",
        "Laplace-mixture-logit",
        "KDE-mixture-logit",
        "Histogram-mixture-logit",
        "GMM-raw",
        "Beta-mixture-raw",
        "Otsu-GaussianCal-logit",
        "LiLee-GaussianCal-logit",
        "Otsu-LaplaceCal-logit",
        "LiLee-LaplaceCal-logit",
        "Otsu-KDECal-logit",
        "LiLee-KDECal-logit",
        "Otsu-sigmoid-control",
        "LiLee-sigmoid-control",
    ]
    rows = []
    sources = {
        "train": ["train set"],
        "test": ["test set"],
        "both": ["train set", "test set"],
    }[args.source]
    for dataset in probe.DATASETS:
        for source in sources:
            for method in methods:
                rows.append(run_one(dataset, method, source))
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {OUT_CSV}")
    print(
        "dataset,method,stage1_acc,stage1_mf1,full_acc,full_mf1,band_rate,error_capture,avg_calls"
    )
    for row in rows:
        print(
            f"{row['dataset']},{row['source']},{row['method']},"
            f"{row['stage1_acc']:.3f},{row['stage1_mf1']:.3f},"
            f"{row['full_acc']:.3f},{row['full_mf1']:.3f},"
            f"{row['band_rate']:.3f},{row['error_capture']:.3f},{row['avg_calls']:.2f}"
        )

    by_method = {}
    for method in methods:
        vals = [r for r in rows if r["method"] == method]
        by_method[method] = (
            float(np.mean([r["stage1_acc"] for r in vals])),
            float(np.mean([r["full_acc"] for r in vals])),
            float(np.mean([r["error_capture"] for r in vals])),
        )
    print("\naverage stage1_acc/full_acc/error_capture")
    for method, vals in by_method.items():
        print(f"{method}: {vals[0]:.3f}/{vals[1]:.3f}/{vals[2]:.3f}")


if __name__ == "__main__":
    main()
