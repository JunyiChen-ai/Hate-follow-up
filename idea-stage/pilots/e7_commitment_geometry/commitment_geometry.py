"""E7 pilot: is the 8B judge's raw-z bimodality a commitment geometry?

Gatekeeper analysis for the "commitment geometry" story. The question is whether
the two modes of the 8B judge's raw-z distribution are corpus-independent
COMMITMENT STATES of the model -- locations pinned by the model, only mixture
weights set by the corpus -- or corpus-dependent class/content clusters.

CPU only. Reads score files that already exist on disk. No model is loaded, no
GPU is touched, nothing is written outside this pilot directory.

Labels enter the analysis only. They are never used to fit a mixture, to choose
a threshold, or to select a cell. Analysis 1 and analysis 5 condition on labels
because the pre-registered question is explicitly about within-class geometry.

Content policy: no transcript text is read or emitted. Only per-video scalar
statistics that the score and gate files already carry.

Pre-registration lives in findings.md; the verdict rule was frozen before any
number in this file was computed.
"""

import json
import os
import sys
import time

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))
os.environ.setdefault("HVD_DATA_ROOT", "/home/jehc223/data")
sys.path.insert(0, os.path.join(ROOT, "src", "our_method"))

from data_utils import load_annotations, load_clean_split_ids  # noqa: E402

# --------------------------------------------------------------------------
# Frozen constants
# --------------------------------------------------------------------------

# KDE recipe, inherited verbatim from scripts/duplex/crossbench_analyze.py so
# that "mode" means here exactly what it means everywhere else in this repo.
N_GRID = 4001
GRID_PAD = 2.0

# The judge's raw z is quantised at 0.25 (logprob readout granularity). A
# mixture component narrower than the quantisation step is an artefact of the
# grid, not a state of the model, so component sd is floored there.
Z_QUANTUM = 0.25
VAR_FLOOR = Z_QUANTUM ** 2

EM_MAX_ITER = 1000
EM_TOL = 1e-9
N_BOOT = 1000
N_RESAMPLE_DRAWS = 200
N_SILVERMAN_BOOT = 300
SEED = 0

LABEL_MAP = {
    "HateMM": {"Hate": 1, "Non Hate": 0},
    "MHClip_EN": {"Hateful": 1, "Offensive": 1, "Normal": 0},
    "MHClip_ZH": {"Hateful": 1, "Offensive": 1, "Normal": 0},
    "ImpliHateVid": {"Hateful": 1, "Normal": 0},
}

# Cell inventory. `condition` records how the audio channel reached the judge:
#   C0  no transcript at all (the original duplex readout)
#   C1  the dataset's own shipped transcript, uncapped
#   C2  fresh gated Whisper large-v3 transcript (channel restoration)
# ImpliHateVid train exists as both C0 and C2, which is the within-corpus
# condition contrast; the crossbench train cells are all C1 because the fresh
# ASR pass was never run there (gate_outcomes = no_fresh_pass for every video);
# every test cell is C2.
CELLS = [
    # dataset, split, condition, model, path
    ("ImpliHateVid", "train", "C0", "8b", "results/duplex_readout/ImpliHateVid/scores.jsonl"),
    ("ImpliHateVid", "train", "C0", "2b", "results/duplex_readout/ImpliHateVid_Qwen3-VL-2B-Instruct/scores.jsonl"),
    ("ImpliHateVid", "train", "C1", "8b", "results/channel_restoration/c1_8b/scores.jsonl"),
    ("ImpliHateVid", "train", "C1", "2b", "results/channel_restoration/c1_2b/scores.jsonl"),
    ("ImpliHateVid", "train", "C2", "8b", "results/c2_fullcorpus/judge_8b/scores.jsonl"),
    ("ImpliHateVid", "train", "C2", "2b", "results/c2_fullcorpus/judge_2b/scores.jsonl"),
    ("HateMM", "train", "C1", "8b", "results/crossbench/hatemm/judge_8b/scores.jsonl"),
    ("HateMM", "train", "C1", "2b", "results/crossbench/hatemm/judge_2b/scores.jsonl"),
    ("MHClip_EN", "train", "C1", "8b", "results/crossbench/mhclip_en/judge_8b/scores.jsonl"),
    ("MHClip_EN", "train", "C1", "2b", "results/crossbench/mhclip_en/judge_2b/scores.jsonl"),
    ("MHClip_ZH", "train", "C1", "8b", "results/crossbench/mhclip_zh/judge_8b/scores.jsonl"),
    ("MHClip_ZH", "train", "C1", "2b", "results/crossbench/mhclip_zh/judge_2b/scores.jsonl"),
    ("ImpliHateVid", "test", "C2", "8b", "results/testruns/implihatevid/judge_8b/scores.jsonl"),
    ("ImpliHateVid", "test", "C2", "2b", "results/testruns/implihatevid/judge_2b/scores.jsonl"),
    ("HateMM", "test", "C2", "8b", "results/testruns/hatemm/judge_8b/scores.jsonl"),
    ("HateMM", "test", "C2", "2b", "results/testruns/hatemm/judge_2b/scores.jsonl"),
    ("MHClip_EN", "test", "C2", "8b", "results/testruns/mhclip_en/judge_8b/scores.jsonl"),
    ("MHClip_EN", "test", "C2", "2b", "results/testruns/mhclip_en/judge_2b/scores.jsonl"),
    ("MHClip_ZH", "test", "C2", "8b", "results/testruns/mhclip_zh/judge_8b/scores.jsonl"),
    ("MHClip_ZH", "test", "C2", "2b", "results/testruns/mhclip_zh/judge_2b/scores.jsonl"),
]

GATE_FILES = {
    ("ImpliHateVid", "train", "C2"): "results/c2_fullcorpus/gate_outcomes.jsonl",
    ("ImpliHateVid", "train", "C1"): "results/channel_restoration/gate_outcomes.jsonl",
    ("HateMM", "train", "C1"): "results/crossbench/hatemm/gate_outcomes.jsonl",
    ("MHClip_EN", "train", "C1"): "results/crossbench/mhclip_en/gate_outcomes.jsonl",
    ("MHClip_ZH", "train", "C1"): "results/crossbench/mhclip_zh/gate_outcomes.jsonl",
    ("ImpliHateVid", "test", "C2"): "results/testruns/implihatevid/gate_outcomes.jsonl",
    ("HateMM", "test", "C2"): "results/testruns/hatemm/gate_outcomes.jsonl",
    ("MHClip_EN", "test", "C2"): "results/testruns/mhclip_en/gate_outcomes.jsonl",
    ("MHClip_ZH", "test", "C2"): "results/testruns/mhclip_zh/gate_outcomes.jsonl",
}

COVARIATES = ["n_transcript_chars", "n_tokens_total", "n_image_tokens_per_frame",
              "vad_speech_frac", "gzip_ratio_raw", "raw_chars", "collapse_shrink",
              "gate_accepted"]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def read_jsonl(path):
    recs = []
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        return recs
    with open(full) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


_ANN_CACHE, _SPLIT_CACHE = {}, {}


def labels_for(dataset, split):
    if dataset not in _ANN_CACHE:
        _ANN_CACHE[dataset] = load_annotations(dataset)
    key = (dataset, split)
    if key not in _SPLIT_CACHE:
        _SPLIT_CACHE[key] = set(load_clean_split_ids(dataset, split))
    ann, lmap = _ANN_CACHE[dataset], LABEL_MAP[dataset]
    return {v: lmap[ann[v]["label"]] for v in _SPLIT_CACHE[key]
            if v in ann and ann[v]["label"] in lmap}


def load_cell(dataset, split, condition, model, path):
    recs = read_jsonl(path)
    lab = labels_for(dataset, split)
    gate = {}
    gpath = GATE_FILES.get((dataset, split, condition))
    if gpath:
        for r in read_jsonl(gpath):
            gate[r["video_id"]] = r
    rows, n_unlabelled = [], 0
    for r in recs:
        v = r["video_id"]
        z = r.get("z")
        if not isinstance(z, (int, float)) or not np.isfinite(z):
            continue
        if v not in lab:
            n_unlabelled += 1
            continue
        row = {"video_id": v, "z": float(z), "y": lab[v]}
        for k in ("n_transcript_chars", "n_tokens_total", "n_image_tokens_per_frame"):
            if isinstance(r.get(k), (int, float)):
                row[k] = float(r[k])
        g = gate.get(v)
        if g:
            for k in ("vad_speech_frac", "gzip_ratio_raw", "raw_chars", "collapse_shrink"):
                if isinstance(g.get(k), (int, float)):
                    row[k] = float(g[k])
            row["gate_accepted"] = 1.0 if g.get("outcome") == "accepted" else 0.0
        rows.append(row)
    name = f"{dataset}|{split}|{condition}|{model}"
    return {
        "name": name, "dataset": dataset, "split": split,
        "condition": condition, "model": model, "path": path,
        "n": len(rows), "n_dropped_unlabelled": n_unlabelled,
        "rows": rows,
        "z": np.array([r["z"] for r in rows]),
        "y": np.array([r["y"] for r in rows], dtype=int),
    }


# --------------------------------------------------------------------------
# Analysis 1 helpers: KDE modes and a unimodality test
# --------------------------------------------------------------------------

def _binned_kde(x, h, grid, nbins=384):
    """Gaussian KDE evaluated on `grid`, with the sample pre-binned.

    Binning keeps the Silverman bootstrap affordable. The bin width is at most
    (range/nbins), always far below the critical bandwidths reported here, so
    the mode count is unaffected.
    """
    lo, hi = x.min(), x.max()
    if hi <= lo:
        hi = lo + 1e-6
    edges = np.linspace(lo, hi, nbins + 1)
    cnt, _ = np.histogram(x, bins=edges)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    m = cnt > 0
    ctr, w = ctr[m], cnt[m] / cnt.sum()
    u = (grid[:, None] - ctr[None, :]) / h
    return (np.exp(-0.5 * u * u) @ w) / (h * np.sqrt(2 * np.pi))


def _n_modes(x, h, grid):
    d = _binned_kde(x, h, grid)
    return int(np.sum((d[1:-1] > d[:-2]) & (d[1:-1] > d[2:])))


def critical_bandwidth(x, grid, iters=24):
    """Silverman's critical bandwidth: smallest h giving a unimodal KDE."""
    s = float(np.std(x, ddof=1)) or 1.0
    lo, hi = 1e-4 * s, 5.0 * s
    while _n_modes(x, hi, grid) > 1 and hi < 200 * s:
        hi *= 2
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        if _n_modes(x, mid, grid) > 1:
            lo = mid
        else:
            hi = mid
    return float(hi)


def silverman_test(x, n_boot=N_SILVERMAN_BOOT, seed=SEED):
    """Silverman (1981) smoothed-bootstrap test of H0: the density is unimodal.

    Small p = the data resist being smoothed into one mode, i.e. evidence for
    at least two modes. The test is known to be conservative, so a small p is
    strong evidence and a large p is weak evidence of unimodality.
    """
    x = np.asarray(x, float)
    if len(x) < 30 or np.std(x, ddof=1) == 0:
        return {"n": int(len(x)), "p_value": None,
                "note": "too few points or zero variance"}
    grid = np.linspace(x.min() - GRID_PAD, x.max() + GRID_PAD, 1025)
    h_crit = critical_bandwidth(x, grid)
    s = float(np.std(x, ddof=1))
    mu = float(np.mean(x))
    rng = np.random.default_rng(seed)
    n = len(x)
    exceed = 0
    for _ in range(n_boot):
        xs = x[rng.integers(0, n, n)]
        eps = rng.standard_normal(n)
        ys = mu + (xs - mu + h_crit * eps) / np.sqrt(1.0 + h_crit ** 2 / s ** 2)
        g2 = np.linspace(ys.min() - GRID_PAD, ys.max() + GRID_PAD, 1025)
        if critical_bandwidth(ys, g2) > h_crit:
            exceed += 1
    return {"n": int(n), "h_crit": round(h_crit, 4),
            "h_crit_over_sd": round(h_crit / s, 4),
            "n_boot": n_boot, "p_value": round(exceed / n_boot, 4)}


def kde_modes(z):
    """The repo's frozen KDE mode/valley readout."""
    from scipy.stats import gaussian_kde
    z = np.asarray(z, float)
    if len(z) < 10 or np.std(z, ddof=1) == 0:
        return {"n": int(len(z)), "n_modes": None}
    kde = gaussian_kde(z, bw_method="scott")
    grid = np.linspace(z.min() - GRID_PAD, z.max() + GRID_PAD, N_GRID)
    d = kde(grid)
    loc = [i for i in range(1, N_GRID - 1) if d[i] > d[i - 1] and d[i] > d[i + 1]]
    out = {"n": int(len(z)), "n_modes": len(loc),
           "bandwidth_scott": round(float(kde.factor * np.std(z, ddof=1)), 4),
           "all_mode_locations": [round(float(grid[i]), 3) for i in loc]}
    if len(loc) < 2:
        return out
    a, b = sorted(sorted(loc, key=lambda i: -d[i])[:2])
    j = a + 1 + int(np.argmin(d[a + 1:b]))
    out.update({
        "top2_mode_locations": [round(float(grid[a]), 3), round(float(grid[b]), 3)],
        "valley": round(float(grid[j]), 3),
        "relative_trough_depth": round(float(1.0 - d[j] / min(d[a], d[b])), 4),
    })
    return out


# --------------------------------------------------------------------------
# Analysis 2 helpers: deterministic 2-component unequal-variance Gaussian EM
# --------------------------------------------------------------------------

def _norm_pdf(x, mu, sd):
    u = (x - mu) / sd
    return np.exp(-0.5 * u * u) / (sd * np.sqrt(2 * np.pi))


def fit_gmm(z, k=2, init="quantile", seed=SEED):
    """EM for a k-component 1-D Gaussian mixture with unequal variances.

    Initialisation is deterministic: means at evenly spaced sample quantiles,
    common sd = sample sd / k, equal weights. No restarts, no label input.
    """
    z = np.asarray(z, float)
    n = len(z)
    if n < 4 * k:
        return None
    qs = [(i + 0.5) / k for i in range(k)]
    mu = np.quantile(z, qs).astype(float)
    mu = mu + np.linspace(-1e-3, 1e-3, k)  # break exact ties deterministically
    sd = np.full(k, max(np.std(z, ddof=1) / k, Z_QUANTUM))
    w = np.full(k, 1.0 / k)
    ll_prev = -np.inf
    for _ in range(EM_MAX_ITER):
        comp = np.stack([w[j] * _norm_pdf(z, mu[j], sd[j]) for j in range(k)], 1)
        tot = comp.sum(1)
        tot = np.maximum(tot, 1e-300)
        ll = float(np.log(tot).sum())
        r = comp / tot[:, None]
        nk = r.sum(0) + 1e-12
        w = nk / n
        mu = (r * z[:, None]).sum(0) / nk
        var = (r * (z[:, None] - mu) ** 2).sum(0) / nk
        sd = np.sqrt(np.maximum(var, VAR_FLOOR))
        if abs(ll - ll_prev) < EM_TOL * max(1.0, abs(ll_prev)):
            ll_prev = ll
            break
        ll_prev = ll
    order = np.argsort(mu)
    mu, sd, w = mu[order], sd[order], w[order]
    npar = 3 * k - 1
    return {"k": k, "mu": mu, "sd": sd, "w": w, "loglik": ll_prev,
            "bic": float(-2 * ll_prev + npar * np.log(n)),
            "aic": float(-2 * ll_prev + 2 * npar), "n": n, "n_params": npar}


def gmm_responsibility_high(z, fit):
    """Posterior of the highest-mean component."""
    z = np.asarray(z, float)
    comp = np.stack([fit["w"][j] * _norm_pdf(z, fit["mu"][j], fit["sd"][j])
                     for j in range(fit["k"])], 1)
    return comp[:, -1] / np.maximum(comp.sum(1), 1e-300)


def gmm_boot(z, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    z = np.asarray(z, float)
    n = len(z)
    mus, sds, ws = [], [], []
    for _ in range(n_boot):
        f = fit_gmm(z[rng.integers(0, n, n)])
        if f is None:
            continue
        mus.append(f["mu"]); sds.append(f["sd"]); ws.append(f["w"])
    if not mus:
        return None
    mus, sds, ws = np.array(mus), np.array(sds), np.array(ws)
    ci = lambda a: [[round(float(np.quantile(a[:, j], 0.025)), 3),
                     round(float(np.quantile(a[:, j], 0.975)), 3)]
                    for j in range(a.shape[1])]
    return {"n_boot": len(mus), "mu_ci95": ci(mus), "sd_ci95": ci(sds),
            "w_ci95": ci(ws),
            "mu_sd": [round(float(np.std(mus[:, j], ddof=1)), 3)
                      for j in range(mus.shape[1])]}


def summarise_fit(fit, boot=None):
    if fit is None:
        return None
    sep = float(fit["mu"][-1] - fit["mu"][0])
    pooled = float(np.sqrt((fit["sd"] ** 2 * fit["w"]).sum()))
    out = {"n": fit["n"], "k": fit["k"],
           "mu": [round(float(v), 3) for v in fit["mu"]],
           "sd": [round(float(v), 3) for v in fit["sd"]],
           "w": [round(float(v), 4) for v in fit["w"]],
           "loglik": round(fit["loglik"], 2), "bic": round(fit["bic"], 2),
           "inter_mode_distance": round(sep, 3),
           "separation_over_pooled_sd": round(sep / pooled, 3) if pooled else None}
    if boot:
        out["bootstrap"] = boot
    return out


# --------------------------------------------------------------------------
# Analysis 5 helpers: AUC and ridge logistic regression
# --------------------------------------------------------------------------

def auc(scores, binary):
    scores, binary = np.asarray(scores, float), np.asarray(binary, int)
    m = np.isfinite(scores)
    scores, binary = scores[m], binary[m]
    p, q = scores[binary == 1], scores[binary == 0]
    if len(p) == 0 or len(q) == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    s, lab = scores[order], binary[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    n1 = len(p)
    return float((ranks[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(q)))


def logistic_cv_auc(X, y, folds=5, ridge=1.0, seed=SEED):
    """Cross-validated AUC of a ridge logistic fit. Newton/IRLS, standardised X."""
    X, y = np.asarray(X, float), np.asarray(y, int)
    n, d = X.shape
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    pred = np.zeros(n)
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        if len(np.unique(y[tr])) < 2:
            return None
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1.0
        A = np.hstack([np.ones((len(tr), 1)), (X[tr] - mu) / sd])
        B = np.hstack([np.ones((len(te), 1)), (X[te] - mu) / sd])
        b = np.zeros(d + 1)
        pen = ridge * np.eye(d + 1)
        pen[0, 0] = 0.0
        for _ in range(60):
            p = 1 / (1 + np.exp(-np.clip(A @ b, -30, 30)))
            W = np.maximum(p * (1 - p), 1e-6)
            H = A.T @ (A * W[:, None]) + pen
            g = A.T @ (y[tr] - p) - pen @ b
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break
            b = b + step
            if np.max(np.abs(step)) < 1e-8:
                break
        pred[te] = B @ b
    return auc(pred, y)


# --------------------------------------------------------------------------
# Analyses
# --------------------------------------------------------------------------

def analysis1_within_class(cells):
    out = {}
    for c in cells:
        blocks = {}
        for cls, name in ((0, "normal_only"), (1, "hateful_only"), (None, "all")):
            z = c["z"] if cls is None else c["z"][c["y"] == cls]
            if len(z) < 30:
                blocks[name] = {"n": int(len(z)), "note": "n<30, skipped"}
                continue
            b = {"kde": kde_modes(z),
                 "mean": round(float(np.mean(z)), 3),
                 "sd": round(float(np.std(z, ddof=1)), 3)}
            b["silverman_unimodality"] = silverman_test(z)
            f = fit_gmm(z, 2)
            b["gmm2"] = summarise_fit(f)
            blocks[name] = b
        if "normal_only" in blocks and "hateful_only" in blocks:
            mn, mh = blocks["normal_only"].get("mean"), blocks["hateful_only"].get("mean")
            if mn is not None and mh is not None:
                blocks["class_mean_gap"] = round(mh - mn, 3)
        out[c["name"]] = blocks
    return out


def analysis2_mode_invariance(cells):
    out = {}
    for c in cells:
        f = fit_gmm(c["z"], 2)
        if f is None:
            out[c["name"]] = {"n": c["n"], "note": "too few points"}
            continue
        out[c["name"]] = summarise_fit(f, gmm_boot(c["z"]))
    return out


def analysis3_composition(cells, prevalences=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
    out = {}
    rng = np.random.default_rng(SEED)
    for c in cells:
        pos, neg = c["z"][c["y"] == 1], c["z"][c["y"] == 0]
        if len(pos) < 40 or len(neg) < 40:
            out[c["name"]] = {"note": "class too small for composition sweep",
                              "n_pos": int(len(pos)), "n_neg": int(len(neg))}
            continue
        n_fixed = int(min(len(pos), len(neg)) / max(prevalences))
        n_fixed = min(n_fixed, 400)
        base = fit_gmm(c["z"], 2)
        boot = gmm_boot(c["z"], n_boot=400, seed=SEED + 1)
        rows = {}
        for p in prevalences:
            npos, nneg = int(round(n_fixed * p)), n_fixed - int(round(n_fixed * p))
            if npos < 5 or nneg < 5:
                continue
            mus, ws = [], []
            for _ in range(N_RESAMPLE_DRAWS):
                s = np.concatenate([rng.choice(pos, npos, replace=True),
                                    rng.choice(neg, nneg, replace=True)])
                f = fit_gmm(s, 2)
                if f is not None:
                    mus.append(f["mu"]); ws.append(f["w"])
            if not mus:
                continue
            mus, ws = np.array(mus), np.array(ws)
            rows[f"{p:.1f}"] = {
                "n_draws": len(mus), "n_per_draw": n_fixed,
                "mu_median": [round(float(np.median(mus[:, j])), 3) for j in range(2)],
                "mu_iqr": [round(float(np.quantile(mus[:, j], .75)
                                       - np.quantile(mus[:, j], .25)), 3) for j in range(2)],
                "w_median": [round(float(np.median(ws[:, j])), 4) for j in range(2)],
            }
        if not rows:
            out[c["name"]] = {"note": "no feasible prevalence cells"}
            continue
        med_lo = [v["mu_median"][0] for v in rows.values()]
        med_hi = [v["mu_median"][1] for v in rows.values()]
        w_hi = [v["w_median"][1] for v in rows.values()]
        d = base["mu"][1] - base["mu"][0]
        drift = [max(med_lo) - min(med_lo), max(med_hi) - min(med_hi)]
        prev = np.array([float(k) for k in rows])
        out[c["name"]] = {
            "full_corpus_fit": summarise_fit(base),
            "bootstrap_ci_full_corpus": boot,
            "inter_mode_distance": round(float(d), 3),
            "drift_threshold_one_third": round(float(d / 3), 3),
            "mu_drift_across_prevalence": [round(float(v), 3) for v in drift],
            "drift_over_intermode": [round(float(v / d), 3) for v in drift],
            "clause_locations_stable": bool(max(drift) < d / 3),
            "weight_tracks_prevalence_pearson_r":
                round(float(np.corrcoef(prev, np.array(w_hi))[0, 1]), 4),
            "per_prevalence": rows,
        }
    return out


def _pool_ll_shared(z_list, mu, sd, w_list):
    ll = 0.0
    for z, w in zip(z_list, w_list):
        comp = np.stack([w[j] * _norm_pdf(z, mu[j], sd[j]) for j in range(len(mu))], 1)
        ll += float(np.log(np.maximum(comp.sum(1), 1e-300)).sum())
    return ll


def fit_shared_location_mixture(z_list, k=2):
    """k components with means/sds SHARED across corpora, weights free per corpus.

    This is the commitment hypothesis written as a model: the model owns the
    locations, the corpus owns only the mixing.
    """
    zc = np.concatenate(z_list)
    qs = [(i + 0.5) / k for i in range(k)]
    mu = np.quantile(zc, qs).astype(float) + np.linspace(-1e-3, 1e-3, k)
    sd = np.full(k, max(np.std(zc, ddof=1) / k, Z_QUANTUM))
    ws = [np.full(k, 1.0 / k) for _ in z_list]
    ll_prev = -np.inf
    for _ in range(EM_MAX_ITER):
        rs, ll = [], 0.0
        for z, w in zip(z_list, ws):
            comp = np.stack([w[j] * _norm_pdf(z, mu[j], sd[j]) for j in range(k)], 1)
            tot = np.maximum(comp.sum(1), 1e-300)
            ll += float(np.log(tot).sum())
            rs.append(comp / tot[:, None])
        ws = [r.sum(0) / len(z) for r, z in zip(rs, z_list)]
        nk = np.zeros(k); sm = np.zeros(k)
        for r, z in zip(rs, z_list):
            nk += r.sum(0); sm += (r * z[:, None]).sum(0)
        mu = sm / np.maximum(nk, 1e-12)
        sv = np.zeros(k)
        for r, z in zip(rs, z_list):
            sv += (r * (z[:, None] - mu) ** 2).sum(0)
        sd = np.sqrt(np.maximum(sv / np.maximum(nk, 1e-12), VAR_FLOOR))
        if abs(ll - ll_prev) < EM_TOL * max(1.0, abs(ll_prev)):
            ll_prev = ll
            break
        ll_prev = ll
    order = np.argsort(mu)
    mu, sd = mu[order], sd[order]
    ws = [w[order] for w in ws]
    n = sum(len(z) for z in z_list)
    npar = 2 * k + len(z_list) * (k - 1)
    return {"k": k, "mu": mu, "sd": sd, "w_per_corpus": ws, "loglik": ll_prev,
            "bic": float(-2 * ll_prev + npar * np.log(n)), "n": n, "n_params": npar}


def analysis4_pooling(cells_by_name, pools):
    out = {}
    for pool_name, members in pools.items():
        cs = [cells_by_name[m] for m in members if m in cells_by_name]
        if len(cs) < 2:
            continue
        z_list = [c["z"] for c in cs]
        zc = np.concatenate(z_list)
        per_k = {}
        for k in (1, 2, 3, 4):
            f = fit_gmm(zc, k)
            if f is None:
                continue
            per_k[f"K={k}"] = {"loglik": round(f["loglik"], 2),
                               "bic": round(f["bic"], 2),
                               "n_params": f["n_params"],
                               "mu": [round(float(v), 3) for v in f["mu"]],
                               "sd": [round(float(v), 3) for v in f["sd"]],
                               "w": [round(float(v), 4) for v in f["w"]]}
        best_k = min(per_k, key=lambda kk: per_k[kk]["bic"]) if per_k else None

        # Structural comparison, the sharper form of the same question.
        shared = fit_shared_location_mixture(z_list, 2)
        free = [fit_gmm(z, 2) for z in z_list]
        ll_free = sum(f["loglik"] for f in free)
        npar_free = sum(f["n_params"] for f in free)
        n = len(zc)
        bic_free = -2 * ll_free + npar_free * np.log(n)
        loc_gap = [round(float(free[i]["mu"][j] - shared["mu"][j]), 3)
                   for i in range(len(free)) for j in range(2)]
        out[pool_name] = {
            "members": [c["name"] for c in cs],
            "n_total": int(n),
            "pooled_bic_by_k": per_k,
            "pooled_bic_best_k": best_k,
            "shared_location_model": {
                "mu": [round(float(v), 3) for v in shared["mu"]],
                "sd": [round(float(v), 3) for v in shared["sd"]],
                "w_per_corpus": [[round(float(v), 4) for v in w]
                                 for w in shared["w_per_corpus"]],
                "loglik": round(shared["loglik"], 2),
                "n_params": shared["n_params"],
                "bic": round(shared["bic"], 2)},
            "free_location_model": {
                "mu_per_corpus": [[round(float(v), 3) for v in f["mu"]] for f in free],
                "sd_per_corpus": [[round(float(v), 3) for v in f["sd"]] for f in free],
                "loglik": round(ll_free, 2), "n_params": npar_free,
                "bic": round(float(bic_free), 2)},
            "delta_bic_shared_minus_free": round(float(shared["bic"] - bic_free), 2),
            "bic_prefers": "shared" if shared["bic"] < bic_free else "free",
            "free_minus_shared_mu": loc_gap,
        }
    return out


def analysis5_covariates(cells, target_names):
    out = {}
    for c in cells:
        if c["name"] not in target_names:
            continue
        f = fit_gmm(c["z"], 2)
        if f is None:
            continue
        resp = gmm_responsibility_high(c["z"], f)
        mem = (resp > 0.5).astype(int)
        avail = [k for k in COVARIATES
                 if sum(1 for r in c["rows"] if k in r) == len(c["rows"])
                 and len({r[k] for r in c["rows"]}) > 1]
        per = {}
        for k in avail:
            v = np.array([r[k] for r in c["rows"]])
            per[k] = {"auc_vs_high_mode_membership": round(auc(v, mem), 4),
                      "auc_vs_gold_label": round(auc(v, c["y"]), 4)}
        lab_auc = auc(c["y"].astype(float), mem)
        X = np.array([[r[k] for k in avail] for r in c["rows"]]) if avail else None
        multi = logistic_cv_auc(X, mem) if X is not None and X.shape[1] else None
        # Within the high-z component only: does the label still separate?
        hi = mem == 1
        out[c["name"]] = {
            "gmm2": summarise_fit(f),
            "n_high_mode": int(hi.sum()),
            "high_mode_label_composition": {
                "n_hateful": int(c["y"][hi].sum()),
                "n_normal": int((1 - c["y"][hi]).sum()),
                "prevalence_in_high_mode": round(float(c["y"][hi].mean()), 4) if hi.sum() else None,
                "prevalence_in_corpus": round(float(c["y"].mean()), 4)},
            "gold_label_auc_for_high_mode_membership": round(lab_auc, 4),
            "labelfree_covariates_available": avail,
            "per_covariate": per,
            "labelfree_multivariate_cv_auc_for_membership":
                round(multi, 4) if multi is not None else None,
        }
    return out


# --------------------------------------------------------------------------
# Supplementary diagnostics. POST-HOC: written after reading analyses 1-6, not
# pre-registered, and they carry no weight in the frozen verdict. They exist to
# separate two things the pre-registered drift statistic cannot tell apart --
# a component whose location moved, and a component whose state was emptied by
# the resampling so that EM repurposed it.
# --------------------------------------------------------------------------

COMMIT_BAND = 13.0


def supp_class_conditional_locations(cells):
    """Fit the same 2-component mixture separately inside each class.

    Commitment reading: within one corpus the two class-conditional fits should
    land on the same pair of locations and differ only in weight.
    """
    out = {}
    for c in cells:
        zs = {0: c["z"][c["y"] == 0], 1: c["z"][c["y"] == 1]}
        if min(len(zs[0]), len(zs[1])) < 40:
            continue
        f0, f1 = fit_gmm(zs[0], 2), fit_gmm(zs[1], 2)
        fa = fit_gmm(c["z"], 2)
        if f0 is None or f1 is None or fa is None:
            continue
        d = float(fa["mu"][1] - fa["mu"][0])
        gap = [abs(float(f1["mu"][j] - f0["mu"][j])) for j in range(2)]
        out[c["name"]] = {
            "normal_mu": [round(float(v), 3) for v in f0["mu"]],
            "normal_w": [round(float(v), 4) for v in f0["w"]],
            "hateful_mu": [round(float(v), 3) for v in f1["mu"]],
            "hateful_w": [round(float(v), 4) for v in f1["w"]],
            "corpus_inter_mode_distance": round(d, 3),
            "class_conditional_location_gap": [round(v, 3) for v in gap],
            "gap_over_intermode": [round(v / d, 3) for v in gap],
            "locations_shared_across_classes": bool(max(gap) < d / 3),
        }
    return out


def supp_restricted_drift(analysis3):
    """Drift of each component's location over the prevalence cells in which
    that component still holds at least 25 percent of the mass."""
    out = {}
    for name, v in analysis3.items():
        if "per_prevalence" not in v:
            continue
        d = v["inter_mode_distance"]
        res = {}
        for j, side in ((0, "low_component"), (1, "high_component")):
            keep = [(p, row) for p, row in v["per_prevalence"].items()
                    if row["w_median"][j] >= 0.25]
            if len(keep) < 3:
                res[side] = {"n_prevalence_cells_retained": len(keep),
                             "note": "component emptied in most cells"}
                continue
            mus = [row["mu_median"][j] for _, row in keep]
            res[side] = {
                "prevalence_cells_retained": [p for p, _ in keep],
                "mu_range": [round(min(mus), 3), round(max(mus), 3)],
                "drift": round(max(mus) - min(mus), 3),
                "drift_over_intermode": round((max(mus) - min(mus)) / d, 3),
                "stable_under_one_third_rule": bool((max(mus) - min(mus)) < d / 3)}
        res["inter_mode_distance"] = d
        out[name] = res
    return out


def supp_shared_location_k(cells_by_name, pools, ks=(2, 3, 4)):
    out = {}
    for pool_name, members in pools.items():
        cs = [cells_by_name[m] for m in members if m in cells_by_name]
        if len(cs) < 2:
            continue
        z_list = [c["z"] for c in cs]
        n = sum(len(z) for z in z_list)
        rows = {}
        for k in ks:
            sh = fit_shared_location_mixture(z_list, k)
            fr = [fit_gmm(z, k) for z in z_list]
            if any(f is None for f in fr):
                continue
            ll_f = sum(f["loglik"] for f in fr)
            np_f = sum(f["n_params"] for f in fr)
            bic_f = -2 * ll_f + np_f * np.log(n)
            rows[f"K={k}"] = {
                "shared_mu": [round(float(v), 3) for v in sh["mu"]],
                "shared_sd": [round(float(v), 3) for v in sh["sd"]],
                "shared_w_per_corpus": [[round(float(v), 4) for v in w]
                                        for w in sh["w_per_corpus"]],
                "shared_bic": round(sh["bic"], 2),
                "free_mu_per_corpus": [[round(float(v), 3) for v in f["mu"]] for f in fr],
                "free_bic": round(float(bic_f), 2),
                "delta_bic_shared_minus_free": round(float(sh["bic"] - bic_f), 2)}
        out[pool_name] = {"members": [c["name"] for c in cs], "by_k": rows}
    return out


def supp_commitment_bands(cells, band=COMMIT_BAND):
    """Occupancy and conditional location of the two saturation bands.

    The band edges are fixed at the same value for every corpus. If the model
    owns the states, the conditional mean inside a band should be the same
    everywhere and only the occupancy should move with the corpus.
    """
    out = {}
    for c in cells:
        z, y = c["z"], c["y"]
        rec = {"band_edge": band}
        for tag, m in (("committed_yes", z >= band), ("committed_no", z <= -band),
                       ("uncommitted", np.abs(z) < band)):
            rec[tag] = {
                "occupancy": round(float(m.mean()), 4),
                "mean_z_in_band": round(float(z[m].mean()), 3) if m.any() else None,
                "sd_z_in_band": round(float(np.std(z[m], ddof=1)), 3) if m.sum() > 1 else None,
                "prevalence_in_band": round(float(y[m].mean()), 4) if m.any() else None,
                "n": int(m.sum())}
        rec["occupancy_by_class"] = {
            "normal": {"committed_yes": round(float((z[y == 0] >= band).mean()), 4),
                       "committed_no": round(float((z[y == 0] <= -band).mean()), 4)},
            "hateful": {"committed_yes": round(float((z[y == 1] >= band).mean()), 4),
                        "committed_no": round(float((z[y == 1] <= -band).mean()), 4)}}
        out[c["name"]] = rec
    return out


def supp_within_corpus_stability(a2):
    """Same corpus, different split and different audio condition."""
    pairs = [("ImpliHateVid|train|C2|8b", "ImpliHateVid|test|C2|8b", "same condition C2, train vs test"),
             ("HateMM|train|C1|8b", "HateMM|test|C2|8b", "C1 train vs C2 test"),
             ("MHClip_EN|train|C1|8b", "MHClip_EN|test|C2|8b", "C1 train vs C2 test"),
             ("MHClip_ZH|train|C1|8b", "MHClip_ZH|test|C2|8b", "C1 train vs C2 test"),
             ("ImpliHateVid|train|C0|8b", "ImpliHateVid|train|C2|8b", "same corpus and split, C0 vs C2")]
    out = {}
    for a, b, note in pairs:
        if a not in a2 or b not in a2 or "mu" not in a2[a] or "mu" not in a2[b]:
            continue
        d = a2[a]["inter_mode_distance"]
        gap = [abs(a2[a]["mu"][j] - a2[b]["mu"][j]) for j in range(2)]
        out[f"{a} vs {b}"] = {
            "note": note, "mu_a": a2[a]["mu"], "mu_b": a2[b]["mu"],
            "location_gap": [round(v, 3) for v in gap],
            "gap_over_intermode": [round(v / d, 3) for v in gap],
            "stable_under_one_third_rule": bool(max(gap) < d / 3)}
    return out


def main():
    t0 = time.time()
    cells = []
    for ds, sp, cond, mdl, path in CELLS:
        c = load_cell(ds, sp, cond, mdl, path)
        if c["n"] == 0:
            print(f"  SKIP (empty) {c['name']}")
            continue
        cells.append(c)
        print(f"  loaded {c['name']:34s} n={c['n']:5d} "
              f"pos={int(c['y'].sum()):4d} dropped={c['n_dropped_unlabelled']}")
    by_name = {c["name"]: c for c in cells}
    c8 = [c for c in cells if c["model"] == "8b"]
    c2 = [c for c in cells if c["model"] == "2b"]

    res = {
        "pilot": "e7_commitment_geometry",
        "question": ("Are the two modes of the 8B judge's raw-z distribution "
                     "corpus-independent commitment states, or corpus-dependent "
                     "class/content clusters?"),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "compute": "CPU only; reads existing score files; no model loaded",
        "frozen_settings": {
            "kde": {"bw": "scott", "n_grid": N_GRID, "grid_pad": GRID_PAD},
            "gmm": {"components": 2, "variance": "unequal", "init": "quantile",
                    "sd_floor": Z_QUANTUM, "max_iter": EM_MAX_ITER, "tol": EM_TOL},
            "bootstrap": N_BOOT, "resample_draws": N_RESAMPLE_DRAWS,
            "silverman_boot": N_SILVERMAN_BOOT, "seed": SEED},
        "cells": [{"name": c["name"], "path": c["path"], "n": c["n"],
                   "n_hateful": int(c["y"].sum()),
                   "prevalence": round(float(c["y"].mean()), 4),
                   "condition": c["condition"], "model": c["model"],
                   "z_min": round(float(c["z"].min()), 3),
                   "z_max": round(float(c["z"].max()), 3),
                   "z_sd": round(float(np.std(c["z"], ddof=1)), 3)} for c in cells],
    }

    print("[1/6] within-class geometry (8B)")
    res["analysis1_within_class_8b"] = analysis1_within_class(c8)
    print("[6/6] within-class geometry + fits (2B contrast)")
    res["analysis6_2b_contrast_within_class"] = analysis1_within_class(c2)
    print("[2/6] mode location invariance")
    res["analysis2_mode_invariance_8b"] = analysis2_mode_invariance(c8)
    res["analysis6_2b_contrast_mixture"] = analysis2_mode_invariance(c2)
    print("[3/6] composition resampling")
    res["analysis3_composition_resampling_8b"] = analysis3_composition(c8)
    print("[4/6] pooling test")
    pools = {
        "IHV_C1_plus_HateMM_C1_8b_matched_dataset_transcript": [
            "ImpliHateVid|train|C1|8b", "HateMM|train|C1|8b"],
        "IHV_C2_plus_HateMM_C2_8b_matched_fresh_transcript_test": [
            "ImpliHateVid|test|C2|8b", "HateMM|test|C2|8b"],
        "all_four_train_C1_8b": [
            "ImpliHateVid|train|C1|8b", "HateMM|train|C1|8b",
            "MHClip_EN|train|C1|8b", "MHClip_ZH|train|C1|8b"],
        "all_four_test_C2_8b": [
            "ImpliHateVid|test|C2|8b", "HateMM|test|C2|8b",
            "MHClip_EN|test|C2|8b", "MHClip_ZH|test|C2|8b"],
        "IHV_C1_plus_HateMM_C1_2b_matched": [
            "ImpliHateVid|train|C1|2b", "HateMM|train|C1|2b"],
    }
    res["analysis4_pooling"] = analysis4_pooling(by_name, pools)
    print("[5/6] covariate accounting")
    res["analysis5_covariates"] = analysis5_covariates(
        cells, {"HateMM|train|C1|8b", "HateMM|test|C2|8b",
                "ImpliHateVid|train|C2|8b"})

    print("[S] supplementary post-hoc diagnostics")
    res["supplementary_post_hoc"] = {
        "status": ("POST-HOC. Written after reading analyses 1-6. Carries no "
                   "weight in the frozen verdict; reported so the failure of "
                   "clause 2 can be read correctly."),
        "s1_class_conditional_locations_8b": supp_class_conditional_locations(c8),
        "s2_restricted_composition_drift_8b": supp_restricted_drift(
            res["analysis3_composition_resampling_8b"]),
        "s3_shared_location_by_k": supp_shared_location_k(by_name, pools),
        "s4_commitment_band_occupancy_8b": supp_commitment_bands(c8),
        "s4_commitment_band_occupancy_2b": supp_commitment_bands(c2, band=1.5),
        "s5_within_corpus_stability_8b": supp_within_corpus_stability(
            res["analysis2_mode_invariance_8b"]),
    }

    res["runtime_seconds"] = round(time.time() - t0, 1)
    out_path = os.path.join(_THIS, "results.json")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {out_path} in {res['runtime_seconds']}s")


if __name__ == "__main__":
    main()
