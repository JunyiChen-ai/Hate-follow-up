"""Self-trained readout kill test: commitment-extreme pseudo-labels, hidden head.

Pre-registration: docs/duplex/PREREG_selftrained_readout_killtest.md, frozen
2026-08-09 before any head was fitted on any corpus.

CPU only. No model call and no rescoring: the layer-27 final-token hidden states
written by the frozen single-call Qwen3-VL-8B judge runs are read from disk,
together with the raw z scores those same runs produced.

Frozen protocol, per corpus, fully label-free at fit time:

  1. Pseudo-labels from the frozen saturation bands: z >= +13 is 1, z <= -13 is
     0. The corpus abstains if either band holds fewer than 10 videos.
  2. Head: L2 logistic regression, lambda fixed at 1.0, on layer-27 final-token
     hidden states standardized per dimension with the mean and standard
     deviation of that corpus's own hidden states over every scored video, no
     label read. The head is fitted on the pseudo-labeled extremes only.
  3. Decision: predict hateful iff the head posterior is at least 0.5, on every
     video including the extremes. No threshold search of any kind.
  4. Controls on the identical pipeline: a z-only logistic with the scalar z as
     the sole feature, and a shuffled-pseudo-label placebo with 20 permutations
     inside the extreme set at seed 20260808.

Frozen decision rule on HateMM, all four required for SURVIVES:
  1. head macro-F1 >= 0.75
  2. AUC of the head posterior >= 0.9132
  3. head macro-F1 >= z-only macro-F1 + 0.03
  4. placebo mean macro-F1 <= head macro-F1 - 0.10
Regression guard: ImpliHateVid head macro-F1 >= 0.8523.

Label loading is imported from scripts/duplex/anchored_operating_point.py and
scripts/duplex/gated_anchor_hateclipseg.py, the code paths that produced the
committed valley numbers, and the run aborts if any valley macro-F1 fails to
reproduce its committed value.

Output: results/selftrained_readout/results.json. Statistics only: no video id
and no transcript text reaches the output.
"""

import json
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(ROOT, "src", "our_method"))
os.environ.setdefault("HVD_DATA_ROOT", "/home/jehc223/data")

from anchored_operating_point import macro_f1, valley_report  # noqa: E402
from crossbench_analyze import LABEL_MAP, load_z  # noqa: E402
from data_utils import load_annotations, load_clean_split_ids  # noqa: E402
from readout_bottleneck_killtest import (  # noqa: E402
    auc, load_hidden, logistic_fit, standardize,
)

# ---------------------------------------------------------------- constants --
# Every constant below is frozen by the preregistration.
SEED = 20260808
BAND_POS = 13.0
BAND_NEG = -13.0
MIN_BAND = 10
LAYER = 27
LAMBDA = 1.0
N_PLACEBO = 20
DECISION = 0.5

CLAUSE1_FLOOR = 0.75          # HateMM head macro-F1
CLAUSE2_FLOOR = 0.9132        # HateMM head posterior AUC
CLAUSE3_MARGIN = 0.03         # head over z-only control
CLAUSE4_MARGIN = 0.10         # head over shuffled placebo mean
GUARD_FLOOR = 0.8523          # ImpliHateVid head macro-F1

# Committed valley macro-F1 of the incumbent threshold. The run aborts on
# mismatch, which is how this script proves it loaded the same labels.
COMMITTED_VALLEY = {
    "HateMM": 0.6561808582882429,
    "ImpliHateVid": 0.8823345329369425,
    "HateClipSeg_strict": 0.4478965766634523,
    "MHClip_EN": 0.6962264150943396,
    "MHClip_ZH": 0.7255985267034991,
}
SELFCHECK_TOL = 0.002

OUT = os.path.join(ROOT, "results", "selftrained_readout", "results.json")


# --------------------------------------------------------------------- data --
def _ordered_split_ids(dataset):
    ids, seen = [], set()
    for v in load_clean_split_ids(dataset, "test"):
        if v not in seen:
            seen.add(v)
            ids.append(v)
    return ids


def load_corpus(name):
    """(ids, z array, gold label array, hidden directory) in split order."""
    if name == "HateClipSeg":
        from hateclipseg_prep import video_labels
        scores = os.path.join(ROOT, "results", "hateclipseg", "judge_8b",
                              "scores.jsonl")
        hidden = os.path.join(ROOT, "results", "hateclipseg", "judge_8b",
                              "hidden")
        z = load_z(scores)
        labels = video_labels()
        ids = [v for v in _ordered_split_ids("HateClipSeg") if v in z]
        # Strict collapse: only the hateful category maps to 1, the secondary
        # collapse of the gated-anchor experiment.
        y = np.array([labels[v][1] for v in ids], dtype=int)
    else:
        slug = {"HateMM": "hatemm", "ImpliHateVid": "implihatevid",
                "MHClip_EN": "mhclip_en", "MHClip_ZH": "mhclip_zh"}[name]
        scores = os.path.join(ROOT, "results", "testruns", slug, "judge_8b",
                              "scores.jsonl")
        hidden = os.path.join(ROOT, "results", "testruns", slug, "judge_8b",
                              "hidden")
        z = load_z(scores)
        ann = load_annotations(name)
        lmap = LABEL_MAP[name]
        ids = [v for v in _ordered_split_ids(name) if v in z]
        y = np.array([lmap[ann[v]["label"]] for v in ids], dtype=int)
    zs = np.array([z[v] for v in ids], dtype=float)
    return ids, zs, y, hidden


# -------------------------------------------------------------------- head ---
def fit_and_score(X, pseudo_idx, pseudo_y, Xall):
    """Fit the L2 logistic on the pseudo-labeled rows, score every row."""
    w, b, ok, nit = logistic_fit(X[pseudo_idx], pseudo_y, LAMBDA)
    margin = Xall @ w + b
    post = 1.0 / (1.0 + np.exp(-margin))
    return post, margin, bool(ok), int(nit)


def report(y, post, margin):
    pred = post >= DECISION
    rep = macro_f1(y, pred)
    rep["auc"] = auc(margin, y)
    rep["predicted_positive_rate"] = float(np.mean(pred))
    return rep, pred


def run_corpus(name):
    ids, zs, y, hidden_dir = load_corpus(name)
    n = len(ids)

    # Incumbent valley on the same z and the same labels: the reproduction
    # check that proves the label collapse matches the committed reports.
    valley = valley_report(zs, y)
    key = "HateClipSeg_strict" if name == "HateClipSeg" else name
    got, want = valley["macro_f1"], COMMITTED_VALLEY[key]
    if abs(got - want) > SELFCHECK_TOL:
        raise SystemExit(f"ABORT: {name} valley macro-F1 {got} does not "
                         f"reproduce the committed {want}")

    band_pos = np.flatnonzero(zs >= BAND_POS)
    band_neg = np.flatnonzero(zs <= BAND_NEG)
    out = {
        "n_scored": n,
        "n_hateful": int(y.sum()),
        "n_normal": int((1 - y).sum()),
        "prevalence_hateful": round(float(y.mean()), 4),
        "z_min": float(zs.min()), "z_median": float(np.median(zs)),
        "z_max": float(zs.max()),
        "n_band_positive": int(band_pos.size),
        "n_band_negative": int(band_neg.size),
        "band_positive_fraction": round(float(band_pos.size) / n, 4),
        "band_negative_fraction": round(float(band_neg.size) / n, 4),
        "valley_baseline": {k: valley[k] for k in
                            ("tp", "fp", "fn", "tn", "macro_f1", "threshold")},
        "valley_selfcheck": {"recomputed": got, "committed": want,
                             "abs_diff": abs(got - want)},
        "judge_z_auc": auc(zs, y),
    }

    if band_pos.size < MIN_BAND or band_neg.size < MIN_BAND:
        out["applicable"] = False
        out["abstain_reason"] = (
            f"a saturation band holds fewer than {MIN_BAND} videos "
            f"({band_pos.size} positive, {band_neg.size} negative); the frozen "
            "protocol abstains with no fallback tuning")
        return out
    out["applicable"] = True

    # Descriptive only: how clean the pseudo-labels are against gold. No fit
    # reads this.
    out["pseudo_label_purity"] = {
        "positive_band_gold_hateful_rate":
            float(y[band_pos].mean()),
        "negative_band_gold_normal_rate":
            float(1.0 - y[band_neg].mean()),
        "pseudo_label_accuracy":
            float((np.concatenate([y[band_pos], 1 - y[band_neg]])).mean()),
    }

    H = load_hidden(hidden_dir, ids)
    Xs, dead = standardize(H[:, LAYER, :].astype(np.float32))
    Xs = Xs.astype(np.float64)
    out["n_zero_variance_dims"] = dead

    pseudo_idx = np.concatenate([band_pos, band_neg])
    pseudo_y = np.concatenate([np.ones(band_pos.size, dtype=int),
                               np.zeros(band_neg.size, dtype=int)])

    post, margin, ok, nit = fit_and_score(Xs, pseudo_idx, pseudo_y, Xs)
    head_rep, head_pred = report(y, post, margin)
    head_rep["converged"] = ok
    head_rep["lbfgs_iterations"] = nit
    out["head"] = head_rep

    # z-only control: the same fit on the standardized scalar z alone.
    zmu, zsd = float(zs.mean()), float(zs.std())
    Z = ((zs - zmu) / (zsd if zsd > 0 else 1.0)).reshape(-1, 1)
    zpost, zmargin, zok, znit = fit_and_score(Z, pseudo_idx, pseudo_y, Z)
    z_rep, z_pred = report(y, zpost, zmargin)
    z_rep["converged"] = zok
    z_rep["lbfgs_iterations"] = znit
    out["z_only_control"] = z_rep

    # Shuffled-pseudo-label placebo on the hidden-state head.
    rng = np.random.default_rng(SEED)
    pl_f1, pl_auc = [], []
    for _ in range(N_PLACEBO):
        perm = rng.permutation(pseudo_y)
        p, m, _, _ = fit_and_score(Xs, pseudo_idx, perm, Xs)
        r, _ = report(y, p, m)
        pl_f1.append(r["macro_f1"])
        pl_auc.append(r["auc"])
    out["placebo"] = {
        "n_permutations": N_PLACEBO,
        "seed": SEED,
        "macro_f1_mean": float(np.mean(pl_f1)),
        "macro_f1_sd": float(np.std(pl_f1, ddof=1)),
        "macro_f1_min": float(np.min(pl_f1)),
        "macro_f1_max": float(np.max(pl_f1)),
        "auc_mean": float(np.mean(pl_auc)),
        "auc_sd": float(np.std(pl_auc, ddof=1)),
    }

    # Movement of the head against the incumbent valley decisions.
    vthr = valley["threshold"]
    vpred = zs >= vthr
    gold = y.astype(bool)
    v_fp = np.flatnonzero(~gold & vpred)
    v_fn = np.flatnonzero(gold & ~vpred)
    out["valley_movement"] = {
        "n_valley_false_positives": int(v_fp.size),
        "n_valley_false_positives_head_calls_normal":
            int(np.sum(~head_pred[v_fp])) if v_fp.size else 0,
        "n_valley_false_negatives": int(v_fn.size),
        "n_valley_false_negatives_head_recovers":
            int(np.sum(head_pred[v_fn])) if v_fn.size else 0,
        "agreement_head_vs_valley": float(np.mean(head_pred == vpred)),
        "n_head_positive": int(head_pred.sum()),
        "n_valley_positive": int(vpred.sum()),
        "n_new_false_positives_head_only":
            int(np.sum(~gold & head_pred & ~vpred)),
        "n_new_false_negatives_head_only":
            int(np.sum(gold & ~head_pred & vpred)),
    }
    return out


# -------------------------------------------------------------------- main ---
def main():
    corpora = ["HateMM", "ImpliHateVid", "HateClipSeg", "MHClip_EN",
               "MHClip_ZH"]
    per_corpus = {}
    for name in corpora:
        print(f"[{name}] running", flush=True)
        per_corpus[name] = run_corpus(name)
        r = per_corpus[name]
        if r["applicable"]:
            print(f"[{name}] head {r['head']['macro_f1']:.4f} "
                  f"z-only {r['z_only_control']['macro_f1']:.4f} "
                  f"placebo {r['placebo']['macro_f1_mean']:.4f} "
                  f"valley {r['valley_baseline']['macro_f1']:.4f}", flush=True)
        else:
            print(f"[{name}] abstains", flush=True)

    hm = per_corpus["HateMM"]
    if not hm["applicable"]:
        raise SystemExit("ABORT: HateMM abstained; the decision corpus must be "
                         "applicable for the rule to be evaluable")
    h_f1 = hm["head"]["macro_f1"]
    h_auc = hm["head"]["auc"]
    z_f1 = hm["z_only_control"]["macro_f1"]
    p_f1 = hm["placebo"]["macro_f1_mean"]

    c1 = bool(h_f1 >= CLAUSE1_FLOOR)
    c2 = bool(h_auc >= CLAUSE2_FLOOR)
    c3 = bool(h_f1 >= z_f1 + CLAUSE3_MARGIN)
    c4 = bool(p_f1 <= h_f1 - CLAUSE4_MARGIN)

    ihv = per_corpus["ImpliHateVid"]
    g_f1 = ihv["head"]["macro_f1"] if ihv["applicable"] else None
    guard = bool(g_f1 is not None and g_f1 >= GUARD_FLOOR)

    verdict = "SURVIVES" if (c1 and c2 and c3 and c4 and guard) else "DEAD"

    res = {
        "title": "Self-trained readout kill test: commitment-extreme "
                 "pseudo-labels supervising a layer-27 hidden-state head",
        "preregistration":
            "docs/duplex/PREREG_selftrained_readout_killtest.md",
        "status": "preregistered kill test, run once; CPU only; no model call",
        "frozen_constants": {
            "band_positive": BAND_POS, "band_negative": BAND_NEG,
            "min_band_size": MIN_BAND, "layer": LAYER, "l2_lambda": LAMBDA,
            "decision_rule": "predict hateful iff head posterior >= 0.5",
            "n_placebo_permutations": N_PLACEBO, "seed": SEED,
            "standardisation": ("per-dimension mean and standard deviation of "
                                "that corpus's own layer-27 hidden states over "
                                "every scored video, no label read"),
            "clause_floors": {
                "clause1_head_macro_f1": CLAUSE1_FLOOR,
                "clause2_head_auc": CLAUSE2_FLOOR,
                "clause3_margin_over_z_only": CLAUSE3_MARGIN,
                "clause4_margin_over_placebo": CLAUSE4_MARGIN,
                "guard_implihatevid_macro_f1": GUARD_FLOOR},
        },
        "inputs": ("results/testruns/{hatemm,implihatevid,mhclip_en,mhclip_zh}"
                   "/judge_8b/{hidden,scores.jsonl} and "
                   "results/hateclipseg/judge_8b/{hidden,scores.jsonl}; 8B only"),
        "label_collapse": {
            "MHClip_EN": "Offensive maps to 1 (union collapse)",
            "MHClip_ZH": "Offensive maps to 1 (union collapse)",
            "HateClipSeg": "strict: only the hateful category maps to 1",
        },
        "decision_rule_verdict": {
            "clause_1_performance": {
                "rule": f"HateMM head macro-F1 >= {CLAUSE1_FLOOR}",
                "head_macro_f1": h_f1, "result": "PASS" if c1 else "FAIL"},
            "clause_2_ranking_preserved": {
                "rule": f"HateMM head posterior AUC >= {CLAUSE2_FLOOR}",
                "head_auc": h_auc, "judge_z_auc": hm["judge_z_auc"],
                "result": "PASS" if c2 else "FAIL"},
            "clause_3_hidden_space_load_bearing": {
                "rule": f"head macro-F1 >= z-only + {CLAUSE3_MARGIN}",
                "head_macro_f1": h_f1, "z_only_macro_f1": z_f1,
                "delta": h_f1 - z_f1, "result": "PASS" if c3 else "FAIL"},
            "clause_4_signal_not_band_arithmetic": {
                "rule": f"placebo mean macro-F1 <= head - {CLAUSE4_MARGIN}",
                "head_macro_f1": h_f1, "placebo_mean_macro_f1": p_f1,
                "placebo_sd": hm["placebo"]["macro_f1_sd"],
                "delta": h_f1 - p_f1, "result": "PASS" if c4 else "FAIL"},
            "regression_guard_implihatevid": {
                "rule": f"ImpliHateVid head macro-F1 >= {GUARD_FLOOR}",
                "head_macro_f1": g_f1, "result": "PASS" if guard else "FAIL"},
            "overall": verdict,
        },
        "summary": {
            name: ({"applicable": False,
                    "valley_macro_f1": per_corpus[name]["valley_baseline"]["macro_f1"],
                    "n_band_positive": per_corpus[name]["n_band_positive"],
                    "n_band_negative": per_corpus[name]["n_band_negative"]}
                   if not per_corpus[name]["applicable"] else
                   {"applicable": True,
                    "n_band_positive": per_corpus[name]["n_band_positive"],
                    "n_band_negative": per_corpus[name]["n_band_negative"],
                    "head_macro_f1": per_corpus[name]["head"]["macro_f1"],
                    "head_auc": per_corpus[name]["head"]["auc"],
                    "z_only_macro_f1":
                        per_corpus[name]["z_only_control"]["macro_f1"],
                    "z_only_auc": per_corpus[name]["z_only_control"]["auc"],
                    "placebo_mean_macro_f1":
                        per_corpus[name]["placebo"]["macro_f1_mean"],
                    "placebo_sd_macro_f1":
                        per_corpus[name]["placebo"]["macro_f1_sd"],
                    "valley_macro_f1":
                        per_corpus[name]["valley_baseline"]["macro_f1"],
                    "judge_z_auc": per_corpus[name]["judge_z_auc"]})
            for name in corpora},
        "verdict": verdict,
        "per_corpus": per_corpus,
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    print(json.dumps({"verdict": verdict,
                      "clauses": res["decision_rule_verdict"],
                      "summary": res["summary"]}, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
