"""Commitment-state precision across annotation boundaries.

Preregistration: docs/duplex/PREREG_commitment_precision.md, frozen 2026-08-08
before any label statistic conditioned on the saturation band was computed on
any corpus.

The question is what the model's positive saturation band contains. The band is
z >= +13, frozen by the saturation-anchor pilot. For each of the five scored
corpora this script measures the strict-hate precision of the band, the
strict-hate precision of the interior comparison band +5 <= z < +13, and the
rate at which each annotation stratum enters the band. Nothing is fitted here:
every label use is evaluative.

CPU only, no model call. Score loading, label loading and the mixture machinery
are imported unmodified from scripts/duplex/anchored_operating_point.py and
scripts/duplex/hateclipseg_prep.py, so the corpora are exactly the ones the
committed operating-point reports used.

Output: results/commitment_precision/results.json. Statistics only: no video id
and no transcript text reaches the output.
"""

import json
import os
import sys

import numpy as np
from scipy.stats import beta

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(ROOT, "src", "our_method"))
os.environ.setdefault("HVD_DATA_ROOT", "/home/jehc223/data")

from anchored_operating_point import (  # noqa: E402
    macro_f1, oracle_f1_hateful, oracle_macro_f1, posterior_pos,
)
from crossbench_analyze import LABEL_MAP, load_z  # noqa: E402
from data_utils import load_annotations, load_clean_split_ids  # noqa: E402
from hateclipseg_prep import video_labels  # noqa: E402

# ---------------------------------------------------------------- constants --
# Every constant below is frozen by the preregistration.
SATURATION_EDGE = 13.0      # band is z >= +13
INTERIOR_LO = 5.0           # interior comparison band is +5 <= z < +13
POWERED_MIN_N = 30          # a cell is powered iff it has >= 30 band videos
CLAUSE_1_MIN_PRECISION = 0.80
CLAUSE_2_MIN_RATIO = 2.0
CLAUSE_3_MIN_GAP = 0.10
CI_ALPHA = 0.05

# Strict-hate collapse per corpus. HateMM and ImpliHateVid ship a binary hate
# label, so strict and union coincide there. MHClip's 3-class annotation is
# collapsed with Offensive on the negative side, the opposite of LABEL_MAP.
STRICT_MAP = {
    "HateMM": {"Hate": 1, "Non Hate": 0},
    "ImpliHateVid": {"Hateful": 1, "Normal": 0},
    "MHClip_EN": {"Hateful": 1, "Offensive": 0, "Normal": 0},
    "MHClip_ZH": {"Hateful": 1, "Offensive": 0, "Normal": 0},
}

TESTRUN_CORPORA = [
    ("ImpliHateVid", "implihatevid"),
    ("HateMM", "hatemm"),
    ("MHClip_EN", "mhclip_en"),
    ("MHClip_ZH", "mhclip_zh"),
]

# Corpora whose annotation carries both an offensive-inclusive and a
# strict-hate boundary, and therefore an offensive-only stratum.
BOTH_BOUNDARIES = ("MHClip_EN", "MHClip_ZH", "HateClipSeg")

ANCHORED_RESULTS = os.path.join(
    ROOT, "results", "anchored_operating_point", "results.json")
OUT_DIR = os.path.join(ROOT, "results", "commitment_precision")


# ------------------------------------------------------------- proportions --
def clopper_pearson(k, n):
    """Exact binomial 95 percent confidence interval for k successes in n."""
    if n == 0:
        return [None, None]
    lo = 0.0 if k == 0 else float(beta.ppf(CI_ALPHA / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - CI_ALPHA / 2, k + 1, n - k))
    return [lo, hi]


def proportion(k, n):
    k, n = int(k), int(n)
    ci = clopper_pearson(k, n)
    return {"k": k, "n": n,
            "estimate": (k / n) if n else None,
            "ci95_lower": ci[0], "ci95_upper": ci[1]}


# --------------------------------------------------------------------- data --
def load_testrun(dataset, slug):
    """(z array, union label array, strict label array, 3-class label list)."""
    ann = load_annotations(dataset)
    ids, seen = [], set()
    for v in load_clean_split_ids(dataset, "test"):
        if v not in seen:
            seen.add(v)
            ids.append(v)
    z = load_z(os.path.join(ROOT, "results", "testruns", slug, "judge_8b",
                            "scores.jsonl"))
    scored = [v for v in ids if v in z]
    zs = np.array([z[v] for v in scored], dtype=float)
    raw = [ann[v]["label"] for v in scored]
    union = np.array([LABEL_MAP[dataset][r] for r in raw], dtype=int)
    strict = np.array([STRICT_MAP[dataset][r] for r in raw], dtype=int)
    return zs, union, strict, raw, len(ids)


def load_hateclipseg():
    ids, seen = [], set()
    for v in load_clean_split_ids("HateClipSeg", "test"):
        if v not in seen:
            seen.add(v)
            ids.append(v)
    z = load_z(os.path.join(ROOT, "results", "hateclipseg", "judge_8b",
                            "scores.jsonl"))
    labels = video_labels()
    scored = [v for v in ids if v in z]
    zs = np.array([z[v] for v in scored], dtype=float)
    union = np.array([labels[v][0] for v in scored], dtype=int)
    strict = np.array([labels[v][1] for v in scored], dtype=int)
    # The shipped annotation is a category set, not a 3-class field; the
    # equivalent of the 3-class label is the (union, strict) pair.
    raw = ["hateful" if s else ("offensive_only" if u else "normal")
           for u, s in zip(union, strict)]
    return zs, union, strict, raw, len(ids)


# ---------------------------------------------------------------- measurement --
def band_masks(zs):
    return {"saturation": zs >= SATURATION_EDGE,
            "interior": (zs >= INTERIOR_LO) & (zs < SATURATION_EDGE)}


def corpus_measurement(dataset, zs, union, strict, raw):
    masks = band_masks(zs)
    bands = {name: proportion(int(strict[m].sum()), int(m.sum()))
             for name, m in masks.items()}

    strata = {
        "strict_hateful": strict == 1,
        "offensive_only": (union == 1) & (strict == 0),
        "normal": union == 0,
    }
    sat = masks["saturation"]
    rates = {name: proportion(int(sat[m].sum()), int(m.sum()))
             for name, m in strata.items()}

    n_sat = int(sat.sum())
    out = {
        "n_scored": int(zs.size),
        "n_strict_hateful": int(strict.sum()),
        "n_offensive_only": int(strata["offensive_only"].sum()),
        "n_normal": int(strata["normal"].sum()),
        "has_both_boundaries": dataset in BOTH_BOUNDARIES,
        "n_saturation_band": n_sat,
        "saturation_band_occupancy": float(np.mean(sat)),
        "powered": n_sat >= POWERED_MIN_N,
        "strict_precision": bands,
        "precision_gap_saturation_minus_interior": (
            bands["saturation"]["estimate"] - bands["interior"]["estimate"]
            if bands["saturation"]["estimate"] is not None
            and bands["interior"]["estimate"] is not None else None),
        "saturation_rate_by_stratum": rates,
        "saturation_band_composition": {
            "strict_hateful": int((strict[sat] == 1).sum()),
            "offensive_only": int(strata["offensive_only"][sat].sum()),
            "normal": int(strata["normal"][sat].sum()),
        },
    }
    if dataset.startswith("MHClip"):
        raw = np.asarray(raw, dtype=object)
        out["saturation_band_three_class_counts"] = {
            c: int(np.sum(raw[sat] == c)) for c in ("Hateful", "Offensive",
                                                    "Normal")}
    return out


# ------------------------------------------------- MHClip strict recollapse --
def strict_recollapse(dataset, zs, union, strict, stored):
    """Rescore every existing comparator under the strict-hate collapse.

    Label-free predictions are reproduced point by point from the parameters
    stored in results/anchored_operating_point/results.json: the valley by its
    threshold, each mixture by the posterior-0.5 rule applied to its stored fit.
    Nothing is refitted. Reproducing the union-collapse macro-F1 from the same
    parameters is the self-check that the reproduction is faithful.
    """
    m = stored["per_corpus"][dataset]["methods"]
    rows, check = {}, {}

    def record(name, pred):
        rows[name] = {
            "strict": macro_f1(strict, pred)["macro_f1"],
            "union_reproduced": macro_f1(union, pred)["macro_f1"],
            "union_committed": m[name]["macro_f1"],
        }
        rows[name]["union_abs_diff"] = abs(rows[name]["union_reproduced"]
                                           - rows[name]["union_committed"])
        check[name] = rows[name]["union_abs_diff"] <= 1e-9

    record("valley", zs >= m["valley"]["threshold"])
    for name in ("free_gmm", "anchored_15.0"):
        record(name, posterior_pos(m[name]["fit"], zs) >= 0.5)

    t_or, o_or = oracle_macro_f1(zs, strict)
    t_or2, o_or2 = oracle_f1_hateful(zs, strict)
    rows["oracle_macro_f1_max"] = {
        "strict": o_or["macro_f1"], "threshold": t_or,
        "union_committed": m["oracle_macro_f1_max"]["macro_f1"],
        "note": "refitted against the strict labels; an oracle is label-fitted "
                "by construction"}
    rows["oracle_f1_hateful_max"] = {
        "strict": o_or2["macro_f1"], "strict_f1_hateful": o_or2["f1_hateful"],
        "threshold": t_or2,
        "union_committed": m["oracle_f1_hateful_max"]["macro_f1"],
        "note": "refitted against the strict labels; the `strict` column is "
                "macro-F1 for comparability, the selection criterion is "
                "hateful-F1"}
    return rows, check


# -------------------------------------------------------------------- main --
def main():
    corpora = {}
    raw_by_corpus = {}
    for dataset, slug in TESTRUN_CORPORA:
        zs, union, strict, raw, n_split = load_testrun(dataset, slug)
        if int(zs.size) == 0:
            raise SystemExit(f"ABORT: no scores loaded for {dataset}")
        corpora[dataset] = (zs, union, strict, raw, n_split)
        raw_by_corpus[dataset] = raw
    corpora["HateClipSeg"] = load_hateclipseg()

    stored = json.load(open(ANCHORED_RESULTS))

    per_corpus, powered = {}, []
    for dataset, (zs, union, strict, raw, n_split) in corpora.items():
        rec = corpus_measurement(dataset, zs, union, strict, raw)
        rec["n_test_clean"] = n_split
        per_corpus[dataset] = rec
        if rec["powered"]:
            powered.append(dataset)

    # ------------------------------------------------------------ clause 1 --
    c1_rows = {d: {"n_saturation_band": per_corpus[d]["n_saturation_band"],
                   "strict_precision": per_corpus[d]["strict_precision"]
                                                    ["saturation"]["estimate"],
                   "ci95": [per_corpus[d]["strict_precision"]["saturation"]
                            ["ci95_lower"],
                            per_corpus[d]["strict_precision"]["saturation"]
                            ["ci95_upper"]],
                   "holds": (per_corpus[d]["strict_precision"]["saturation"]
                             ["estimate"] >= CLAUSE_1_MIN_PRECISION)}
              for d in powered}
    c1 = all(r["holds"] for r in c1_rows.values())

    # ------------------------------------------------------------ clause 2 --
    hcs = per_corpus["HateClipSeg"]["saturation_rate_by_stratum"]
    r_hate = hcs["strict_hateful"]["estimate"]
    r_off = hcs["offensive_only"]["estimate"]
    ratio = (r_hate / r_off) if r_off else None
    c2 = ratio is not None and ratio >= CLAUSE_2_MIN_RATIO

    # ------------------------------------------------------------ clause 3 --
    c3_rows = {d: {"saturation_precision": per_corpus[d]["strict_precision"]
                                                        ["saturation"]["estimate"],
                   "interior_precision": per_corpus[d]["strict_precision"]
                                                     ["interior"]["estimate"],
                   "n_interior": per_corpus[d]["strict_precision"]["interior"]["n"],
                   "gap": per_corpus[d]["precision_gap_saturation_minus_interior"],
                   "holds": (per_corpus[d]
                             ["precision_gap_saturation_minus_interior"]
                             >= CLAUSE_3_MIN_GAP)}
              for d in powered}
    c3 = all(r["holds"] for r in c3_rows.values())

    verdict = {
        "clause_1_band_is_strict_hate": {
            "rule": "in every powered cell, P(strict hate | z >= +13) >= 0.80",
            "powered_cells": powered,
            "per_corpus": c1_rows,
            "result": "PASS" if c1 else "FAIL"},
        "clause_2_strict_over_offensive_asymmetry": {
            "rule": "on HateClipSeg, the saturation rate among strict-hateful "
                    "videos is at least twice the rate among offensive-only "
                    "videos",
            "rate_strict_hateful": r_hate,
            "rate_offensive_only": r_off,
            "ratio": ratio,
            "result": "PASS" if c2 else "FAIL"},
        "clause_3_band_exceeds_interior": {
            "rule": "in every powered cell, saturation-band strict precision "
                    "exceeds interior-band strict precision by at least 0.10",
            "per_corpus": c3_rows,
            "result": "PASS" if c3 else "FAIL"},
        "overall": "PASS" if (c1 and c2 and c3) else "FAIL",
    }

    # ------------------------------------------------- descriptive analyses --
    recollapse, selfcheck = {}, {}
    for dataset in ("MHClip_EN", "MHClip_ZH"):
        zs, union, strict, _, _ = corpora[dataset]
        rows, chk = strict_recollapse(dataset, zs, union, strict, stored)
        recollapse[dataset] = rows
        selfcheck[dataset] = chk
    if not all(all(v.values()) for v in selfcheck.values()):
        raise SystemExit(
            "ABORT: the stored fit parameters do not reproduce the committed "
            f"union-collapse macro-F1: {json.dumps(selfcheck)}")

    out = {
        "title": "Commitment-state precision of the positive saturation band "
                 "across five corpora, Qwen3-VL-8B judge",
        "preregistration": "docs/duplex/PREREG_commitment_precision.md",
        "status": "preregistered mechanism analysis, run once; CPU only; no "
                  "new model calls; every label use is evaluative",
        "frozen_constants": {
            "saturation_band_lower_edge": SATURATION_EDGE,
            "interior_band": [INTERIOR_LO, SATURATION_EDGE],
            "powered_minimum_band_n": POWERED_MIN_N,
            "clause_1_min_precision": CLAUSE_1_MIN_PRECISION,
            "clause_2_min_ratio": CLAUSE_2_MIN_RATIO,
            "clause_3_min_precision_gap": CLAUSE_3_MIN_GAP,
            "confidence_interval": "Clopper-Pearson exact binomial, 95 percent"},
        "inputs": "results/testruns/{implihatevid,hatemm,mhclip_en,mhclip_zh}"
                  "/judge_8b/scores.jsonl and results/hateclipseg/judge_8b/"
                  "scores.jsonl (8B only)",
        "strict_label_definition": {
            "HateMM": "shipped binary hate label",
            "ImpliHateVid": "shipped binary hate label",
            "MHClip_EN": "3-class Label field, Hateful -> 1, Offensive and "
                         "Normal -> 0",
            "MHClip_ZH": "3-class Label field, Hateful -> 1, Offensive and "
                         "Normal -> 0",
            "HateClipSeg": "hateful-strict collapse of the video-level "
                           "category set"},
        "powered_cells": powered,
        "decision_rule_verdict": verdict,
        "per_corpus": per_corpus,
        "descriptive_mhclip_strict_recollapse": recollapse,
        "descriptive_recollapse_selfcheck": selfcheck,
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(json.dumps({"overall": verdict["overall"],
                      "clauses": {k: v["result"] for k, v in verdict.items()
                                  if isinstance(v, dict)},
                      "powered": powered}, indent=1))


if __name__ == "__main__":
    main()
