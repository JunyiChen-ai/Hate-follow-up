"""Anchored operating point: a one-sided positive-saturation threshold rule.

Preregistration: docs/duplex/PREREG_anchored_operating_point.md (frozen
2026-08-08 before any mixture was fitted to these four score files).

The experiment fits a two-component one-dimensional Gaussian mixture to each
test corpus's raw-z scores and decides hateful by the posterior of the positive
component. In the primary model the positive component's mean is pinned at the
model-owned saturation location +15.0 and never re-estimated; the negative
component and the mixing weight are fitted freely. The comparators are two
placebo anchors at +10.0 and +20.0, a fully free two-component mixture, the
incumbent KDE-valley threshold, and a labeled oracle.

Everything here is CPU only and deterministic given the frozen seed 20260808.
No model call is made. Labels are used for scoring, for the oracle, and to
build the prevalence resamples; no fitted component reads a label.

The KDE-valley recipe and the dataset label loading are imported from the code
path that produced docs/duplex/reports/test_c2_*.json, so the valley numbers in
this report are the committed ones by construction. The script aborts if they
do not reproduce.

Output: results/anchored_operating_point/results.json. Statistics only: no
video id and no transcript text reach the output.
"""

import json
import math
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(ROOT, "src", "our_method"))

# Same default as scripts/duplex/generic_harm_2d_analyze.py: this machine keeps
# the benchmarks outside the cluster's shared path. An existing environment
# variable always wins.
os.environ.setdefault("HVD_DATA_ROOT", "/home/jehc223/data")

# kde_valley is the frozen incumbent recipe; LABEL_MAP is the binary collapse
# in which MHClip's Offensive maps to 1. Both are imported unmodified from the
# analysis module that produced the committed test-split reports.
from crossbench_analyze import LABEL_MAP, kde_valley, load_z  # noqa: E402
from data_utils import load_annotations, load_clean_split_ids  # noqa: E402

# ---------------------------------------------------------------- constants --
# Every constant below is frozen by the preregistration.
SEED = 20260808
ANCHOR_PRIMARY = 15.0
ANCHOR_PLACEBO = (10.0, 20.0)
N_RESTARTS = 50
RESAMPLE_RATES = (0.5, 0.25)
N_RESAMPLES = 200
SIGMA_FLOOR = 0.1

# EM stopping rule (implementation detail, not a preregistered constant).
MAX_ITER = 5000
LL_TOL = 1e-10

CORPORA = [
    ("ImpliHateVid", "implihatevid"),
    ("HateMM", "hatemm"),
    ("MHClip_EN", "mhclip_en"),
    ("MHClip_ZH", "mhclip_zh"),
]

# Committed macro-F1 of the incumbent valley threshold, from
# docs/duplex/reports/test_c2_<slug>_8b.json. The run aborts on mismatch.
COMMITTED_VALLEY_MACRO_F1 = {
    "ImpliHateVid": 0.8823345329369425,
    "HateMM": 0.6561808582882429,
    "MHClip_EN": 0.6962264150943396,
    "MHClip_ZH": 0.7255985267034991,
}
SELFCHECK_TOL = 0.002

LOG_2PI = math.log(2.0 * math.pi)


# ------------------------------------------------------------------ scoring --
def macro_f1(y, pred):
    """Macro-F1 over the two classes, matching operating_point() in
    crossbench_analyze.py."""
    y = np.asarray(y, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    tp = int(np.sum(y & pred))
    fp = int(np.sum(~y & pred))
    fn = int(np.sum(y & ~pred))
    tn = int(np.sum(~y & ~pred))

    def f1(a, b, c):
        prec = a / (a + b) if a + b else 0.0
        rec = a / (a + c) if a + c else 0.0
        return 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    f1p = f1(tp, fp, fn)
    f1n = f1(tn, fn, fp)
    n = tp + fp + fn + tn
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "f1_hateful": f1p, "f1_normal": f1n,
        "accuracy": (tp + tn) / n if n else 0.0,
        "macro_f1": (f1p + f1n) / 2,
    }


def oracle_macro_f1(z, y):
    """Macro-F1-maximizing threshold against gold labels. Upper reference."""
    cands = np.unique(z)
    best, best_thr = None, None
    for t in cands:
        m = macro_f1(y, z >= t)
        if best is None or m["macro_f1"] > best["macro_f1"]:
            best, best_thr = m, float(t)
    return best_thr, best


def oracle_f1_hateful(z, y):
    """Hateful-F1-maximizing threshold: the oracle convention used in the
    committed test_c2 reports, carried here for continuity."""
    cands = np.unique(z)
    best, best_thr = None, None
    for t in cands:
        m = macro_f1(y, z >= t)
        if best is None or m["f1_hateful"] > best["f1_hateful"]:
            best, best_thr = m, float(t)
    return best_thr, best


# ---------------------------------------------------------------------- EM --
def _fit_gmm(z, seed, anchor=None, n_restarts=N_RESTARTS):
    """Two-component 1-D Gaussian mixture by EM, vectorized over restarts.

    If `anchor` is not None the positive component's mean is held at that value
    and only its variance and the mixing weight are updated. Initialization
    draws component means from randomly chosen data points and never reads a
    label. Standard deviations are floored at SIGMA_FLOOR to prevent collapse
    onto a single point.

    Returns the best-log-likelihood restart as a dict of scalars, with the
    positive component being the anchored one (anchored fit) or the
    higher-mean one (free fit).
    """
    z = np.asarray(z, dtype=float)
    n = z.size
    rng = np.random.default_rng(seed)
    R = n_restarts

    s = float(np.std(z)) if np.std(z) > 0 else 1.0
    scale = rng.uniform(0.5, 1.5, size=(R, 2))
    sig_p = np.maximum(s * scale[:, 0], SIGMA_FLOOR)
    sig_n = np.maximum(s * scale[:, 1], SIGMA_FLOOR)
    pi = rng.uniform(0.2, 0.8, size=R)

    idx = rng.integers(0, n, size=(R, 2))
    if anchor is None:
        mu_p = z[idx[:, 0]].copy()
        mu_n = z[idx[:, 1]].copy()
        # Break exact ties so the two components do not start identical.
        same = mu_p == mu_n
        mu_p[same] = mu_p[same] + s * 0.5
        mu_n[same] = mu_n[same] - s * 0.5
    else:
        mu_p = np.full(R, float(anchor))
        mu_n = z[idx[:, 1]].copy()

    zz = z[None, :]
    ll_prev = np.full(R, -np.inf)
    n_iter = 0
    for n_iter in range(1, MAX_ITER + 1):
        lp = (-0.5 * ((zz - mu_p[:, None]) / sig_p[:, None]) ** 2
              - np.log(sig_p)[:, None] - 0.5 * LOG_2PI + np.log(pi)[:, None])
        ln = (-0.5 * ((zz - mu_n[:, None]) / sig_n[:, None]) ** 2
              - np.log(sig_n)[:, None] - 0.5 * LOG_2PI + np.log1p(-pi)[:, None])
        m = np.maximum(lp, ln)
        lse = m + np.log(np.exp(lp - m) + np.exp(ln - m))
        ll = lse.sum(axis=1)

        rp = np.exp(lp - lse)
        rn = 1.0 - rp
        np_p = np.maximum(rp.sum(axis=1), 1e-8)
        np_n = np.maximum(rn.sum(axis=1), 1e-8)

        pi = np.clip(np_p / n, 1e-6, 1 - 1e-6)
        if anchor is None:
            mu_p = (rp * zz).sum(axis=1) / np_p
        mu_n = (rn * zz).sum(axis=1) / np_n
        sig_p = np.maximum(
            np.sqrt((rp * (zz - mu_p[:, None]) ** 2).sum(axis=1) / np_p),
            SIGMA_FLOOR)
        sig_n = np.maximum(
            np.sqrt((rn * (zz - mu_n[:, None]) ** 2).sum(axis=1) / np_n),
            SIGMA_FLOOR)

        if np.all(np.abs(ll - ll_prev) < LL_TOL):
            ll_prev = ll
            break
        ll_prev = ll

    ll = ll_prev
    ll = np.where(np.isfinite(ll), ll, -np.inf)
    b = int(np.argmax(ll))
    out = {"mu_pos": float(mu_p[b]), "sigma_pos": float(sig_p[b]),
           "mu_neg": float(mu_n[b]), "sigma_neg": float(sig_n[b]),
           "pi_pos": float(pi[b]), "loglik": float(ll[b]),
           "n_iter": int(n_iter), "anchored": anchor is not None}
    if anchor is None and out["mu_neg"] > out["mu_pos"]:
        out = {"mu_pos": out["mu_neg"], "sigma_pos": out["sigma_neg"],
               "mu_neg": out["mu_pos"], "sigma_neg": out["sigma_pos"],
               "pi_pos": 1.0 - out["pi_pos"], "loglik": out["loglik"],
               "n_iter": out["n_iter"], "anchored": False}
    return out


def posterior_pos(fit, z):
    """Posterior probability of the positive component at each z."""
    z = np.asarray(z, dtype=float)
    lp = (-0.5 * ((z - fit["mu_pos"]) / fit["sigma_pos"]) ** 2
          - math.log(fit["sigma_pos"]) - 0.5 * LOG_2PI + math.log(fit["pi_pos"]))
    ln = (-0.5 * ((z - fit["mu_neg"]) / fit["sigma_neg"]) ** 2
          - math.log(fit["sigma_neg"]) - 0.5 * LOG_2PI
          + math.log1p(-fit["pi_pos"]))
    m = np.maximum(lp, ln)
    return np.exp(lp - (m + np.log(np.exp(lp - m) + np.exp(ln - m))))


def decision_boundaries(fit):
    """Every z at which the positive posterior crosses 0.5.

    With unequal variances the log-density difference is a quadratic in z, so
    the decision region can have two boundaries rather than one. The full real
    root set is returned; an empty set means one component dominates everywhere.
    """
    sp, sn = fit["sigma_pos"], fit["sigma_neg"]
    mp, mn = fit["mu_pos"], fit["mu_neg"]
    pp = fit["pi_pos"]
    a = 1.0 / (2 * sn * sn) - 1.0 / (2 * sp * sp)
    b = mp / (sp * sp) - mn / (sn * sn)
    c = (mn * mn / (2 * sn * sn) - mp * mp / (2 * sp * sp)
         + math.log(pp / sp) - math.log((1.0 - pp) / sn))
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return []
        return [-c / b]
    disc = b * b - 4 * a * c
    if disc < 0:
        return []
    r = math.sqrt(disc)
    return sorted([(-b - r) / (2 * a), (-b + r) / (2 * a)])


def primary_threshold(fit, z_median):
    """The crossing nearest the corpus median, used as the drift statistic."""
    bs = decision_boundaries(fit)
    if not bs:
        return None, bs
    return float(min(bs, key=lambda t: abs(t - z_median))), bs


def gmm_report(fit, z, y, z_median):
    thr, bs = primary_threshold(fit, z_median)
    pred = posterior_pos(fit, z) >= 0.5
    rep = macro_f1(y, pred)
    rep.update({
        "threshold": thr,
        "boundary_set": [float(x) for x in bs],
        "n_boundaries": len(bs),
        "decision_region_is_single_threshold": len(bs) == 1,
        "fit": fit,
        "predicted_positive_rate": float(np.mean(pred)),
    })
    return rep


# ------------------------------------------------------------------- valley --
def valley_report(z, y):
    v = kde_valley(z)
    if v.get("value") is None:
        return {"threshold": None, "macro_f1": None, "kde": v}
    rep = macro_f1(y, np.asarray(z) >= v["value"])
    rep.update({"threshold": float(v["value"]),
                "boundary_set": [float(v["value"])],
                "n_boundaries": 1,
                "decision_region_is_single_threshold": True,
                "mode_locations": v["mode_locations"],
                "relative_trough_depth": v["relative_trough_depth"]})
    return rep


# --------------------------------------------------------------------- data --
def load_corpus(dataset, slug):
    ann = load_annotations(dataset)
    lmap = LABEL_MAP[dataset]
    ids, seen = [], set()
    for v in load_clean_split_ids(dataset, "test"):
        if v not in seen:
            seen.add(v)
            ids.append(v)
    z = load_z(os.path.join(ROOT, "results", "testruns", slug, "judge_8b",
                            "scores.jsonl"))
    scored = [v for v in ids if v in z]
    zs = np.array([z[v] for v in scored], dtype=float)
    ys = np.array([lmap[ann[v]["label"]] for v in scored], dtype=int)
    return zs, ys, len(ids)


# ------------------------------------------------------------- stress test --
def prevalence_stress(zs, ys, corpus_index, full_thr):
    """Prevalence resampling: retain the positive class at the frozen rates,
    refit all three label-free thresholds per resample, and record the median
    absolute drift from the full-corpus threshold and the median macro-F1."""
    pos_idx = np.flatnonzero(ys == 1)
    neg_idx = np.flatnonzero(ys == 0)
    out = {}
    for ri, rate in enumerate(RESAMPLE_RATES):
        k = int(round(rate * pos_idx.size))
        rows = {m: {"thr": [], "f1": [], "n_two_boundaries": 0,
                    "n_no_boundary": 0, "n_no_valley": 0}
                for m in ("valley", "free_gmm", "anchored")}
        for b in range(N_RESAMPLES):
            rng = np.random.default_rng([SEED, corpus_index, ri, b])
            keep = np.concatenate([rng.choice(pos_idx, size=k, replace=False),
                                   neg_idx])
            zb, yb = zs[keep], ys[keep]
            med = float(np.median(zb))

            v = kde_valley(zb)
            if v.get("value") is None:
                rows["valley"]["n_no_valley"] += 1
            else:
                rows["valley"]["thr"].append(float(v["value"]))
                rows["valley"]["f1"].append(
                    macro_f1(yb, zb >= v["value"])["macro_f1"])

            for name, anchor in (("free_gmm", None),
                                 ("anchored", ANCHOR_PRIMARY)):
                seed = [SEED, corpus_index, ri, b, 1 if anchor else 0]
                fit = _fit_gmm(zb, seed, anchor=anchor)
                thr, bs = primary_threshold(fit, med)
                if len(bs) == 2:
                    rows[name]["n_two_boundaries"] += 1
                if thr is None:
                    rows[name]["n_no_boundary"] += 1
                else:
                    rows[name]["thr"].append(thr)
                rows[name]["f1"].append(
                    macro_f1(yb, posterior_pos(fit, zb) >= 0.5)["macro_f1"])

        res = {}
        for m, r in rows.items():
            base = full_thr.get(m)
            drift = ([abs(t - base) for t in r["thr"]]
                     if base is not None else [])
            res[m] = {
                "n_resamples": N_RESAMPLES,
                "full_corpus_threshold": base,
                "median_abs_threshold_drift": (float(np.median(drift))
                                               if drift else None),
                "iqr_abs_threshold_drift": (
                    [float(np.quantile(drift, 0.25)),
                     float(np.quantile(drift, 0.75))] if drift else None),
                "median_macro_f1": (float(np.median(r["f1"]))
                                    if r["f1"] else None),
                "n_thresholds_recovered": len(r["thr"]),
                "n_with_two_boundaries": r["n_two_boundaries"],
                "n_with_no_boundary": r["n_no_boundary"],
                "n_without_valley": r["n_no_valley"],
            }
        out[f"rate_{rate}"] = {
            "retain_rate": rate,
            "n_positive_kept": k,
            "n_negative": int(neg_idx.size),
            "resampled_prevalence": round(k / (k + neg_idx.size), 4),
            "methods": res,
        }
    return out


# -------------------------------------------------------------------- main --
def main():
    per_corpus = {}
    selfcheck = {}
    for ci, (dataset, slug) in enumerate(CORPORA):
        zs, ys, n_split = load_corpus(dataset, slug)
        med = float(np.median(zs))
        methods = {}

        methods["valley"] = valley_report(zs, ys)
        got = methods["valley"]["macro_f1"]
        want = COMMITTED_VALLEY_MACRO_F1[dataset]
        selfcheck[dataset] = {"recomputed": got, "committed": want,
                              "abs_diff": abs(got - want),
                              "matches": abs(got - want) <= SELFCHECK_TOL}
        if not selfcheck[dataset]["matches"]:
            raise SystemExit(
                f"ABORT: {dataset} valley macro-F1 {got} does not reproduce the "
                f"committed {want}; labels or scores were loaded differently")

        fit = _fit_gmm(zs, [SEED, ci, 100], anchor=ANCHOR_PRIMARY)
        methods["anchored_15.0"] = gmm_report(fit, zs, ys, med)
        for a in ANCHOR_PLACEBO:
            f = _fit_gmm(zs, [SEED, ci, int(a)], anchor=a)
            methods[f"placebo_{a}"] = gmm_report(f, zs, ys, med)
        ffit = _fit_gmm(zs, [SEED, ci, 200], anchor=None)
        methods["free_gmm"] = gmm_report(ffit, zs, ys, med)

        t_or, m_or = oracle_macro_f1(zs, ys)
        methods["oracle_macro_f1_max"] = dict(m_or, threshold=t_or,
                                              label_free=False)
        t_or2, m_or2 = oracle_f1_hateful(zs, ys)
        methods["oracle_f1_hateful_max"] = dict(m_or2, threshold=t_or2,
                                                label_free=False)

        full_thr = {"valley": methods["valley"]["threshold"],
                    "free_gmm": methods["free_gmm"]["threshold"],
                    "anchored": methods["anchored_15.0"]["threshold"]}
        stress = prevalence_stress(zs, ys, ci, full_thr)

        per_corpus[dataset] = {
            "n_test_clean": n_split,
            "n_scored": int(zs.size),
            "n_hateful": int(ys.sum()),
            "n_normal": int((1 - ys).sum()),
            "prevalence_hateful": round(float(ys.mean()), 4),
            "z_min": float(zs.min()), "z_median": med, "z_max": float(zs.max()),
            "methods": methods,
            "prevalence_stress": stress,
        }

    names = [d for d, _ in CORPORA]

    def mean_f1(key):
        return float(np.mean([per_corpus[d]["methods"][key]["macro_f1"]
                              for d in names]))

    m_anchor = mean_f1("anchored_15.0")
    m_free = mean_f1("free_gmm")
    m_valley = mean_f1("valley")
    m_p10 = mean_f1("placebo_10.0")
    m_p20 = mean_f1("placebo_20.0")

    # Clause 1 -- the anchor must be load-bearing.
    c1_margins = {"vs_placebo_10.0": m_anchor - m_p10,
                  "vs_placebo_20.0": m_anchor - m_p20}
    c1 = all(v >= 0.01 for v in c1_margins.values())

    # Clause 2 -- not merely an ordinary mixture.
    c2a = m_anchor >= m_free - 0.005
    c2b_detail = {}
    c2b = True
    for rate in RESAMPLE_RATES:
        key = f"rate_{rate}"
        wins, rows = 0, {}
        for d in names:
            s = per_corpus[d]["prevalence_stress"][key]["methods"]
            da = s["anchored"]["median_abs_threshold_drift"]
            df = s["free_gmm"]["median_abs_threshold_drift"]
            ok = (da is not None and df is not None and da <= 0.5 * df)
            wins += int(ok)
            rows[d] = {"anchored_drift": da, "free_gmm_drift": df,
                       "ratio": (da / df if da is not None and df else None),
                       "at_or_below_half": ok}
        c2b_detail[key] = {"n_corpora_at_or_below_half": wins, "per_corpus": rows,
                           "clause_holds": wins >= 3}
        c2b = c2b and wins >= 3
    c2 = c2a and c2b

    # Clause 3 -- the operating point must pay.
    hm_a = per_corpus["HateMM"]["methods"]["anchored_15.0"]["macro_f1"]
    hm_v = per_corpus["HateMM"]["methods"]["valley"]["macro_f1"]
    c3a = hm_a >= hm_v + 0.08
    c3b = m_anchor >= m_valley + 0.03
    per_corpus_floor = {
        d: {"anchored": per_corpus[d]["methods"]["anchored_15.0"]["macro_f1"],
            "valley": per_corpus[d]["methods"]["valley"]["macro_f1"],
            "delta": (per_corpus[d]["methods"]["anchored_15.0"]["macro_f1"]
                      - per_corpus[d]["methods"]["valley"]["macro_f1"]),
            "above_floor": (per_corpus[d]["methods"]["anchored_15.0"]["macro_f1"]
                            >= per_corpus[d]["methods"]["valley"]["macro_f1"]
                            - 0.03)}
        for d in names}
    c3c = all(v["above_floor"] for v in per_corpus_floor.values())
    c3 = c3a and c3b and c3c

    verdict = {
        "clause_1_anchor_is_load_bearing": {
            "rule": "mean macro-F1 of the anchored model exceeds each placebo "
                    "anchor by at least 0.01",
            "mean_macro_f1_anchored": m_anchor,
            "mean_macro_f1_placebo_10": m_p10,
            "mean_macro_f1_placebo_20": m_p20,
            "margins": c1_margins,
            "result": "PASS" if c1 else "FAIL"},
        "clause_2_not_an_ordinary_mixture": {
            "rule": "(a) anchored mean macro-F1 at least free-2-GMM mean minus "
                    "0.005, and (b) at every resampling rate the anchored "
                    "median threshold drift is at most half the free-2-GMM "
                    "drift on at least 3 of 4 corpora",
            "mean_macro_f1_anchored": m_anchor,
            "mean_macro_f1_free_gmm": m_free,
            "clause_2a": "PASS" if c2a else "FAIL",
            "clause_2b": "PASS" if c2b else "FAIL",
            "clause_2b_detail": c2b_detail,
            "result": "PASS" if c2 else "FAIL"},
        "clause_3_performance": {
            "rule": "HateMM macro-F1 at least valley + 0.08; mean macro-F1 at "
                    "least valley mean + 0.03; no corpus below valley - 0.03",
            "hatemm_anchored": hm_a, "hatemm_valley": hm_v,
            "hatemm_delta": hm_a - hm_v,
            "mean_macro_f1_anchored": m_anchor,
            "mean_macro_f1_valley": m_valley,
            "mean_delta": m_anchor - m_valley,
            "per_corpus_floor": per_corpus_floor,
            "clause_3a_hatemm": "PASS" if c3a else "FAIL",
            "clause_3b_mean": "PASS" if c3b else "FAIL",
            "clause_3c_floor": "PASS" if c3c else "FAIL",
            "result": "PASS" if c3 else "FAIL"},
        "overall": "PASS" if (c1 and c2 and c3) else "FAIL",
    }

    out = {
        "title": "Anchored operating point on the four held-out test corpora, "
                 "Qwen3-VL-8B judge",
        "preregistration": "docs/duplex/PREREG_anchored_operating_point.md",
        "status": "preregistered threshold-method experiment, run once; CPU "
                  "only; no new model calls",
        "frozen_constants": {
            "anchor_primary": ANCHOR_PRIMARY,
            "anchor_placebo": list(ANCHOR_PLACEBO),
            "seed": SEED, "em_restarts": N_RESTARTS,
            "resample_rates": list(RESAMPLE_RATES),
            "n_resamples_per_rate": N_RESAMPLES,
            "sigma_floor": SIGMA_FLOOR,
            "decision": "predict hateful iff posterior of the positive "
                        "component is at least 0.5"},
        "inputs": "results/testruns/{implihatevid,hatemm,mhclip_en,mhclip_zh}"
                  "/judge_8b/scores.jsonl (8B only; 2B out of scope)",
        "label_mapping": {d: LABEL_MAP[d] for d, _ in CORPORA},
        "valley_selfcheck": selfcheck,
        "summary_macro_f1": {
            d: {k: per_corpus[d]["methods"][k]["macro_f1"]
                for k in ("anchored_15.0", "placebo_10.0", "placebo_20.0",
                          "free_gmm", "valley", "oracle_macro_f1_max",
                          "oracle_f1_hateful_max")} for d in names},
        "summary_threshold": {
            d: {k: per_corpus[d]["methods"][k]["threshold"]
                for k in ("anchored_15.0", "placebo_10.0", "placebo_20.0",
                          "free_gmm", "valley", "oracle_macro_f1_max")}
            for d in names},
        "mean_macro_f1": {"anchored_15.0": m_anchor, "placebo_10.0": m_p10,
                          "placebo_20.0": m_p20, "free_gmm": m_free,
                          "valley": m_valley,
                          "oracle_macro_f1_max": mean_f1("oracle_macro_f1_max")},
        "decision_rule_verdict": verdict,
        "per_corpus": per_corpus,
    }

    d = os.path.join(ROOT, "results", "anchored_operating_point")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "results.json"), "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(json.dumps({"verdict": verdict["overall"],
                      "clauses": {k: v["result"] for k, v in verdict.items()
                                  if isinstance(v, dict)},
                      "mean_macro_f1": out["mean_macro_f1"],
                      "summary_macro_f1": out["summary_macro_f1"]}, indent=1))


if __name__ == "__main__":
    main()
