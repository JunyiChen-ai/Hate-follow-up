"""Ranking-error autopsy, stage 3: taxonomy counts, counterfactual AUC, covariate screens.

Reads the manual codings under results/ranking_autopsy/coding_*.tsv, joins them to
the item tables, and produces the aggregate statistics that reach the committed
note: code counts with Wilson intervals, the arithmetic of what each attributable
code is worth in AUC, and the label-free covariate screen per code.

No video id and no transcript text reaches the output json.
"""

import csv
import json
import math
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _THIS_DIR)
OUT = os.path.join(ROOT, "results", "ranking_autopsy")

from ranking_autopsy_decompose import auc, auc_ci, quart  # noqa: E402


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def read_coding(name):
    d = {}
    with open(os.path.join(OUT, name)) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            d[row["alias"]] = row["code"]
    return d


def counts(codes, order=None):
    from collections import Counter
    c = Counter(codes.values())
    n = len(codes)
    keys = order or [k for k, _ in c.most_common()]
    return {k: {"n": c[k], "rate": c[k] / n, "wilson95": list(wilson(c[k], n))}
            for k in keys if c[k]}, n


COV = ["fresh_chars", "n_transcript_chars", "vad_speech_frac", "wav_duration",
       "gzip_ratio_raw", "segfrac_union", "segfrac_hateful", "n_segments"]


def covariate_screen(rows, codes, min_n=5):
    """Per-code medians of the label-free covariates, plus the cohort median."""
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows:
        by[codes[r["alias"]]].append(r)
    allmed = {c: float(np.median([r[c] for r in rows if r.get(c) is not None]))
              for c in COV if any(r.get(c) is not None for r in rows)}
    out = {"_cohort_median": allmed, "_cohort_n": len(rows)}
    for code, rs in by.items():
        if len(rs) < min_n:
            continue
        e = {"n": len(rs), "median_z": float(np.median([r["z"] for r in rs]))}
        for c in COV:
            v = [r[c] for r in rs if r.get(c) is not None]
            if v:
                e[c] = float(np.median(v))
        # gate outcome: judge saw no transcript at all
        e["frac_judge_saw_no_transcript"] = float(
            np.mean([1.0 if (r.get("n_transcript_chars") or 0) == 0 else 0.0 for r in rs]))
        e["frac_judge_text_truncated"] = float(np.mean(
            [1.0 if (r.get("fresh_chars") or 0) > 0 and
             (r.get("n_transcript_chars") or 0) < 0.9 * (r.get("fresh_chars") or 0)
             else 0.0 for r in rs]))
        out[code] = e
    return out


def main():
    res = {}

    # ----------------------------------------------------------------- EN --
    en_items = json.load(open(os.path.join(OUT, "en", "items.json")))
    en_pk = json.load(open(os.path.join(OUT, "en", "packet.json")))
    en_codes = read_coding("coding_en.tsv")
    fn = [r for r in en_pk if r["side"] == "FN"]
    fp = [r for r in en_pk if r["side"] == "FP"]
    fn_c, n1 = counts({r["alias"]: en_codes[r["alias"]] for r in fn},
                      ["T", "G", "I", "R", "S", "E"])
    fp_c, n2 = counts({r["alias"]: en_codes[r["alias"]] for r in fp},
                      ["C", "A", "P", "V", "N", "D"])
    res["en"] = {
        "fn_taxonomy": {"n": n1, "codes": fn_c},
        "fp_taxonomy": {"n": n2, "codes": fp_c},
        "fn_covariates": covariate_screen(fn, en_codes),
        "fp_covariates": covariate_screen(fp, en_codes),
    }

    # Counterfactual: how much of the primary AUC does each positive class own?
    byc = {c: [i["z"] for i in en_items if i["fine"] == c]
           for c in ("Hateful", "Offensive", "Normal")}
    res["en"]["auc_arithmetic"] = {
        "primary_union": auc(byc["Hateful"] + byc["Offensive"], byc["Normal"]),
        "hateful_only": auc(byc["Hateful"], byc["Normal"]),
        "offensive_only": auc(byc["Offensive"], byc["Normal"]),
        "weighted_reconstruction": (13 * auc(byc["Hateful"], byc["Normal"])
                                    + 36 * auc(byc["Offensive"], byc["Normal"])) / 49,
    }

    # ---------------------------------------------------------------- HCS --
    hcs_items = json.load(open(os.path.join(OUT, "hcs", "items.json")))
    hcs_fn = [r for r in json.load(open(os.path.join(OUT, "hcs", "packet.json")))
              if r["side"] == "FN"]
    hcs_neg = json.load(open(os.path.join(OUT, "hcs", "packet_allneg.json")))
    fn_codes = read_coding("coding_hcs_fn.tsv")
    neg_codes = read_coding("coding_hcs_neg.tsv")

    fnc, n3 = counts(fn_codes, ["M", "E", "P", "I", "N", "B", "V", "S"])
    ngc, n4 = counts(neg_codes, ["O", "H", "X", "R"])
    res["hcs"] = {
        "fn_taxonomy": {"n": n3, "codes": fnc},
        "negative_class_taxonomy": {"n": n4, "codes": ngc,
                                    "note": "all 50 negatives coded, not a top-k slice"},
        "fn_covariates": covariate_screen(hcs_fn, fn_codes),
        "negative_covariates": covariate_screen(hcs_neg, neg_codes),
    }

    # How many of the coded FN videos carry the hateful label at all?
    res["hcs"]["fn_label_composition"] = {
        "n": len(hcs_fn),
        "carrying_hateful_label": sum(1 for r in hcs_fn if r["strict"] == 1),
        "median_segfrac_hateful": float(np.median([r.get("segfrac_hateful", 0.0)
                                                   for r in hcs_fn])),
    }

    # Counterfactual AUC: drop the negatives coded H or X (hate or extremist
    # glorification annotated normal) and refit the corpus AUC.
    bad = {r["alias"] for r in hcs_neg if neg_codes[r["alias"]] in ("H", "X")}
    bad_ids = {r["video_id"] for r in hcs_neg if r["alias"] in bad}
    keep = [i for i in hcs_items if i["video_id"] not in bad_ids]

    def two(items, key):
        return ([i["z"] for i in items if i[key] == 1],
                [i["z"] for i in items if i[key] == 0])

    ar = {}
    for nm, key in (("offensive_union", "union"), ("hateful_strict", "strict")):
        p0, n0 = two(hcs_items, key)
        p1, n1_ = two(keep, key)
        a0, lo0, hi0 = auc_ci(p0, n0)
        a1, lo1, hi1 = auc_ci(p1, n1_)
        ar[nm] = {"as_shipped": {"auc": a0, "ci95": [lo0, hi0], "n_pos": len(p0), "n_neg": len(n0)},
                  "hate_negatives_removed": {"auc": a1, "ci95": [lo1, hi1],
                                             "n_pos": len(p1), "n_neg": len(n1_)},
                  "delta": a1 - a0}
    # Second counterfactual for the strict collapse: also move the removed
    # negatives to the positive side, which is what their content implies.
    pos2 = [i["z"] for i in hcs_items if i["strict"] == 1 or i["video_id"] in bad_ids]
    neg2 = [i["z"] for i in hcs_items if i["strict"] == 0 and i["video_id"] not in bad_ids]
    a2, lo2, hi2 = auc_ci(pos2, neg2)
    ar["hateful_strict"]["hate_negatives_relabelled_positive"] = {
        "auc": a2, "ci95": [lo2, hi2], "n_pos": len(pos2), "n_neg": len(neg2),
        "delta": a2 - ar["hateful_strict"]["as_shipped"]["auc"]}
    res["hcs"]["auc_arithmetic"] = ar

    # Where do the H/X negatives sit in the score distribution?
    hx = [i["z"] for i in hcs_items if i["video_id"] in bad_ids]
    oth = [i["z"] for i in hcs_items if i["union"] == 0 and i["video_id"] not in bad_ids]
    res["hcs"]["negative_class_split"] = {
        "hate_or_extremist_negatives": quart(hx),
        "remaining_negatives": quart(oth),
        "hateful_labelled_positives": quart([i["z"] for i in hcs_items if i["strict"] == 1]),
    }

    # Degeneracy-gate exposure across the whole corpus, by collapse.
    for nm, key in (("union", "union"), ("strict", "strict")):
        res["hcs"].setdefault("gate_exposure", {})[nm] = {
            "positives_with_no_transcript": float(np.mean(
                [1.0 if (i.get("n_transcript_chars") or 0) == 0 else 0.0
                 for i in hcs_items if i[key] == 1])),
            "negatives_with_no_transcript": float(np.mean(
                [1.0 if (i.get("n_transcript_chars") or 0) == 0 else 0.0
                 for i in hcs_items if i[key] == 0])),
        }
    # AUC restricted to videos the judge actually received a transcript for.
    got = [i for i in hcs_items if (i.get("n_transcript_chars") or 0) > 0]
    for nm, key in (("offensive_union", "union"), ("hateful_strict", "strict")):
        p, n = two(got, key)
        a, lo, hi = auc_ci(p, n)
        res["hcs"]["auc_arithmetic"][nm]["transcript_present_only"] = {
            "auc": a, "ci95": [lo, hi], "n_pos": len(p), "n_neg": len(n)}

    with open(os.path.join(OUT, "autopsy_summary.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
