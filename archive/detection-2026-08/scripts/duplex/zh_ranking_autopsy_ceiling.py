"""Ranking-error autopsy for MHClip-ZH, stage 1b: the oracle-ceiling arithmetic.

Answers two questions with no new model call.

1. Is the 0.7932 labelled-oracle macro-F1 low relative to what a ranking of
   AUC 0.855 at prevalence 0.302 can buy? Compared against a binormal reference
   curve and against the three sister corpora measured under the same judge.
2. Which side of the confusion matrix owns the residual, and how much would a
   correction on the negative class be worth? Sweeps the removal of the k
   highest-scoring negatives, mirroring the HateClipSeg correction table.

Also compares the three ZH judge arms that already exist on disk (fresh Whisper,
dataset transcript, forced-Chinese ASR) so that the text channel's contribution
to the ranking is measured rather than assumed.
"""

import json
import math
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(ROOT, "results", "ranking_autopsy", "zh")
REPORTS = os.path.join(ROOT, "docs", "duplex", "reports")


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="stable")
    r = np.empty(len(allv), float)
    r[order] = np.arange(len(allv), dtype=float)
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = np.mean(r[order[i:j + 1]])
        i = j + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def oracle_macro_f1(pos, neg):
    pos, neg = list(pos), list(neg)
    best = None
    for t in sorted(set(pos + neg) | {min(pos + neg) - 1}):
        tp = sum(1 for z in pos if z >= t)
        fp = sum(1 for z in neg if z >= t)
        fn = len(pos) - tp
        tn = len(neg) - fp
        f1p = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        f1n = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) else 0.0
        m = (f1p + f1n) / 2
        if best is None or m > best[0]:
            best = (m, t, tp, fp, fn, tn)
    return {"macro_f1": round(best[0], 4), "threshold": best[1], "tp": best[2],
            "fp": best[3], "fn": best[4], "tn": best[5]}


# --------------------------------------------------------- binormal reference --
def _ndtr(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _ndtri(p):
    # Acklam inverse normal CDF, adequate to 1e-9
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def binormal_max_macro_f1(a, prev, n=200000, seed=7):
    """Max macro-F1 of an equal-variance binormal ranking with the given AUC."""
    d = math.sqrt(2.0) * _ndtri(a)
    npos = int(round(n * prev))
    nneg = n - npos
    rng = np.random.default_rng(seed)
    pos = rng.normal(d, 1.0, npos)
    neg = rng.normal(0.0, 1.0, nneg)
    grid = np.quantile(np.concatenate([pos, neg]), np.linspace(0.001, 0.999, 999))
    best = 0.0
    for t in grid:
        tp = float((pos >= t).sum())
        fp = float((neg >= t).sum())
        fn = npos - tp
        tn = nneg - fp
        f1p = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        f1n = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) else 0.0
        best = max(best, (f1p + f1n) / 2)
    return round(best, 4)


def read_jsonl(path):
    d = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                o = json.loads(line)
                d[o["video_id"]] = o
    return d


def main():
    items = json.load(open(os.path.join(OUT, "items.json")))
    lab = {i["video_id"]: i["binary"] for i in items}
    fine = {i["video_id"]: i["fine"] for i in items}
    pos = [i["z"] for i in items if i["binary"] == 1]
    neg = [i["z"] for i in items if i["binary"] == 0]
    res = {}

    # ---------------------------------------------- 1. cross-corpus ceiling --
    corpora = [("HateMM", "test_c2_hatemm_8b.json"),
               ("ImpliHateVid", "test_c2_implihatevid_8b.json"),
               ("MHClip-EN", "test_c2_mhclip_en_8b.json"),
               ("MHClip-ZH", "test_c2_mhclip_zh_8b.json")]
    rows = []
    for name, fn in corpora:
        d = json.load(open(os.path.join(REPORTS, fn)))
        a = d["auc"]["hateful_vs_normal"]
        o = d["operating_points"]["diagnostic_oracle_f1max_NOT_label_free"]
        prev = d["coverage"]["prevalence_hateful"]
        rows.append({"corpus": name, "auc": round(a, 4), "prevalence": prev,
                     "oracle_macro_f1": round(o["macro_f1"], 4),
                     "binormal_reference": binormal_max_macro_f1(a, prev),
                     "tp": o["tp"], "fp": o["fp"], "fn": o["fn"], "tn": o["tn"],
                     "fp_share_of_errors": round(o["fp"] / (o["fp"] + o["fn"]), 4)})
    for r in rows:
        r["observed_minus_reference"] = round(r["oracle_macro_f1"] - r["binormal_reference"], 4)
    res["ceiling_vs_binormal"] = rows

    # ------------------------------------- 2. negative-class correction sweep --
    order = sorted(items, key=lambda i: -i["z"])
    negs_desc = [i for i in order if i["binary"] == 0]
    sweep = []
    for k in (0, 3, 5, 7, 10, 14, 17, 21):
        drop = {i["video_id"] for i in negs_desc[:k]}
        p = [i["z"] for i in items if i["binary"] == 1]
        n = [i["z"] for i in items if i["binary"] == 0 and i["video_id"] not in drop]
        o = oracle_macro_f1(p, n)
        sweep.append({"k_top_negatives_removed": k, "n_neg_left": len(n),
                      "auc": round(auc(p, n), 4), "oracle_macro_f1": o["macro_f1"],
                      "min_z_removed": round(negs_desc[k - 1]["z"], 2) if k else None})
    res["negative_correction_sweep"] = sweep

    # the mirror sweep on the positive side, for symmetry
    pos_asc = [i for i in sorted(items, key=lambda i: i["z"]) if i["binary"] == 1]
    psweep = []
    for k in (0, 3, 5, 7, 10):
        drop = {i["video_id"] for i in pos_asc[:k]}
        p = [i["z"] for i in items if i["binary"] == 1 and i["video_id"] not in drop]
        n = [i["z"] for i in items if i["binary"] == 0]
        o = oracle_macro_f1(p, n)
        psweep.append({"k_bottom_positives_removed": k, "n_pos_left": len(p),
                       "auc": round(auc(p, n), 4), "oracle_macro_f1": o["macro_f1"],
                       "max_z_removed": round(pos_asc[k - 1]["z"], 2) if k else None})
    res["positive_correction_sweep"] = psweep

    # ------------------------------------------------ 3. text-channel arms ----
    arms = {"fresh_whisper_auto": "judge_8b", "dataset_transcript": "judge_8b_ctrl",
            "forced_chinese_asr": "judge_8b_forcedzh"}
    arm_rows = []
    zs = {}
    for nm, sub in arms.items():
        p = os.path.join(ROOT, "results", "testruns", "mhclip_zh", sub, "scores.jsonl")
        if not os.path.isfile(p):
            continue
        s = read_jsonl(p)
        s = {v: o for v, o in s.items() if v in lab}
        zs[nm] = s
        pv = [o["z"] for v, o in s.items() if lab[v] == 1]
        nv = [o["z"] for v, o in s.items() if lab[v] == 0]
        o2 = oracle_macro_f1(pv, nv)
        arm_rows.append({"arm": nm, "n": len(s), "auc": round(auc(pv, nv), 4),
                         "oracle_macro_f1": o2["macro_f1"], "fp": o2["fp"], "fn": o2["fn"],
                         "median_transcript_chars": float(np.median(
                             [o.get("n_transcript_chars", 0) for o in s.values()]))})
    res["text_channel_arms"] = arm_rows
    if "fresh_whisper_auto" in zs and "dataset_transcript" in zs:
        common = sorted(set(zs["fresh_whisper_auto"]) & set(zs["dataset_transcript"]))
        d = [zs["fresh_whisper_auto"][v]["z"] - zs["dataset_transcript"][v]["z"] for v in common]
        res["arm_delta_fresh_minus_dataset"] = {
            "n": len(common), "median_abs_delta": float(np.median(np.abs(d))),
            "mean_delta": round(float(np.mean(d)), 4),
            "n_sign_flip_about_oracle_threshold": int(sum(
                1 for v in common
                if (zs["fresh_whisper_auto"][v]["z"] >= 5.0) !=
                   (zs["dataset_transcript"][v]["z"] >= 5.0)))}

    # ------------------------------- 4. how much of the ranking is text-borne --
    # correlation between the two arms and the score spread they produce
    res["fine_class_at_oracle"] = {
        "threshold": 5.0,
        "above": {c: sum(1 for i in items if i["z"] >= 5.0 and i["fine"] == c)
                  for c in ("Hateful", "Offensive", "Normal")},
        "below": {c: sum(1 for i in items if i["z"] < 5.0 and i["fine"] == c)
                  for c in ("Hateful", "Offensive", "Normal")}}

    # positive-class compression: how narrow is the positive band
    res["distribution_shape"] = {
        "pos_iqr": round(float(np.percentile(pos, 75) - np.percentile(pos, 25)), 3),
        "neg_iqr": round(float(np.percentile(neg, 75) - np.percentile(neg, 25)), 3),
        "pos_sd": round(float(np.std(pos, ddof=1)), 3),
        "neg_sd": round(float(np.std(neg, ddof=1)), 3),
        "sd_ratio_neg_over_pos": round(float(np.std(neg, ddof=1) / np.std(pos, ddof=1)), 3),
        "frac_neg_above_pos_q1": round(float(np.mean(np.asarray(neg) >= np.percentile(pos, 25))), 4),
        "frac_pos_below_neg_q3": round(float(np.mean(np.asarray(pos) <= np.percentile(neg, 75))), 4),
        "note": "unequal-variance ranking: the negative class is the wide one",
    }

    with open(os.path.join(OUT, "ceiling.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
