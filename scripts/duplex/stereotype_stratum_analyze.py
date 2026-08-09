#!/usr/bin/env python3
"""Stereotype-stratum check on HateClipSeg, stage 2: analysis.

Question. On the pooled MHClip implicit stratum the frozen single-call
Qwen3-VL-8B judge ranks gender and sexuality stereotype content at ROC-AUC 0.71,
flat across evidence-length bins, while ranking ImpliHateVid's race and
nationality dog-whistles at 0.9255. That contrast rests on eighteen videos.
This script asks whether the same pattern appears inside a third corpus, using
a single corpus, a single negative class and codes assigned from content.

Inputs: results/stereotype_stratum/coding.tsv (149 manually coded videos) and
results/stereotype_stratum/packet.json (scores and covariates).

Outputs results/stereotype_stratum/results.json. No video identifier, no
transcript text and no recognised string reaches that file.

No model call, no GPU, CPU only.
"""

import csv
import json
import math
import os
import statistics
import sys

import numpy as np
from scipy.stats import rankdata, mannwhitneyu

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _THIS_DIR)

OUT_DIR = os.path.join(ROOT, "results", "stereotype_stratum")
SEED = 20260808
N_BOOT = 4000
CODES = ["GS", "OS", "EX", "NO"]


# ---------------------------------------------------------------- statistics
def auc(pos, neg):
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0)
                 / (len(pos) * len(neg)))


def auc_ci(pos, neg, n_boot=N_BOOT, seed=SEED):
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    vals = [auc(rng.choice(pos, size=len(pos), replace=True),
                rng.choice(neg, size=len(neg), replace=True))
            for _ in range(n_boot)]
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def med(xs):
    xs = [x for x in xs if x is not None
          and not (isinstance(x, float) and math.isnan(x))]
    return float(statistics.median(xs)) if xs else float("nan")


def mw_p(a, b):
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)


def pct_rank(value, pool):
    """Percentile of `value` inside `pool`, midrank convention, 0..100."""
    pool = [p for p in pool if p is not None]
    if not pool:
        return float("nan")
    below = sum(1 for p in pool if p < value)
    equal = sum(1 for p in pool if p == value)
    return 100.0 * (below + 0.5 * equal) / len(pool)


# ------------------------------------------------------------------- loading
def load_packet():
    with open(os.path.join(OUT_DIR, "packet.json")) as f:
        return {r["alias"]: r for r in json.load(f)}


def load_coding():
    out = {}
    with open(os.path.join(OUT_DIR, "coding.tsv")) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            out[row["alias"]] = row
    return out


def cohort(packet, coding, prefix, code=None, explicit=None, conf=None):
    sel = []
    for a, r in packet.items():
        if r["cohort"] != prefix:
            continue
        c = coding[a]
        if code is not None and c["code"] != code:
            continue
        if explicit is not None and int(c["explicit"]) != explicit:
            continue
        if conf is not None and c["conf"] != conf:
            continue
        sel.append(a)
    return sorted(sel)


def zs(packet, aliases):
    return [packet[a]["z"] for a in aliases]


def covs(packet, aliases, key):
    return [packet[a].get(key) for a in aliases]


# ------------------------------------------------------ MHClip pooled cohort
def mhclip_gender_cohort():
    """Per-video z for the MHClip implicit videos the implicitness note counted
    as gender or sexuality stereotype content, plus each corpus's own score
    pool for percentile placement. Reuses the predecessor's loaders and the
    same keyword rule, so the count reproduces the note's fifteen."""
    import implicitness_evidence_analyze as iea

    GENDER = ("gender", "women", "woman", "men ", "misogyn", "sexual-orientation",
              "same-sex", "misgender", "effeminate", "gender-expression",
              "gender-identity", "sexual shaming")

    en = iea.load_en()
    zh = iea.load_zh()

    en_notes = {}
    with open(os.path.join(ROOT,
                           "results/ranking_autopsy/coding_en_protected.tsv")) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            en_notes[row["alias"]] = row["note"]
    en_alias = {}
    for fn in ("packet.json", "packet_rest.json"):
        with open(os.path.join(ROOT, "results/ranking_autopsy/en", fn)) as f:
            for r in json.load(f):
                en_alias[r["video_id"]] = r["alias"]
    zh_note = {zh["alias2id"][a]: row["note"] for a, row in zh["codes"].items()
               if a in zh["alias2id"]}

    def is_gender(text):
        t = (text or "").lower()
        return any(k in t for k in GENDER)

    en_gs = [v for v in en["implicit"] if is_gender(en_notes.get(en_alias[v]))]
    zh_gs = [v for v in zh["implicit"] if is_gender(zh_note.get(v))]

    en_all = [r["z"] for r in en["by_id"].values()]
    zh_all = [r["z"] for r in zh["by_id"].values()]

    return {
        "EN": {"z": [en["by_id"][v]["z"] for v in en_gs], "pool": en_all,
               "normals": [en["by_id"][v]["z"] for v in en["normals"]],
               "n_implicit": len(en["implicit"])},
        "ZH": {"z": [zh["by_id"][v]["z"] for v in zh_gs], "pool": zh_all,
               "normals": [zh["by_id"][v]["z"] for v in zh["normals"]],
               "n_implicit": len(zh["implicit"])},
    }


# ------------------------------------------------------------------- main
def main():
    packet = load_packet()
    coding = load_coding()
    res = {"seed": SEED, "n_boot": N_BOOT}

    nrm_all = cohort(packet, coding, "NRM")
    ins_all = cohort(packet, coding, "INS")
    hat_all = cohort(packet, coding, "HAT")

    # The negative class. `shipped` is the fifty videos carrying no offensive
    # category, the same negative class the ranking autopsy used for 0.637.
    # `audited` additionally drops the negatives this coding calls group-hostile
    # or group-stereotyping, which is the annotation-error cohort the autopsy
    # documented from the other direction.
    neg_shipped = zs(packet, nrm_all)
    nrm_clean = [a for a in nrm_all if coding[a]["code"] == "NO"]
    neg_audited = zs(packet, nrm_clean)

    res["cohorts"] = {"insulting_only": len(ins_all), "clean_normal": len(nrm_all),
                      "hateful_sample": len(hat_all),
                      "clean_normal_after_audit": len(nrm_clean)}

    # ---- 1. code tables ---------------------------------------------------
    tables = {}
    for name, pool in (("insulting_only", ins_all), ("hateful_sample", hat_all),
                       ("clean_normal", nrm_all),
                       ("insulting_plus_hateful", ins_all + hat_all)):
        rows = {}
        for code in CODES:
            sel = [a for a in pool if coding[a]["code"] == code]
            z = zs(packet, sel)
            row = {"n": len(sel), "median_z": med(z),
                   "median_fresh_chars": med(covs(packet, sel, "fresh_chars")),
                   "median_vad": med(covs(packet, sel, "vad_speech_frac")),
                   "median_duration": med(covs(packet, sel, "wav_duration")),
                   "n_low_confidence": sum(1 for a in sel if coding[a]["conf"] == "l"),
                   "n_explicit_register": sum(1 for a in sel
                                              if int(coding[a]["explicit"]) == 1)}
            if name != "clean_normal" and sel:
                a1 = auc(z, neg_shipped)
                lo, hi = auc_ci(z, neg_shipped)
                row["auc_vs_clean_normal"] = a1
                row["auc_ci95"] = [lo, hi]
                row["auc_vs_audited_normal"] = auc(z, neg_audited)
            rows[code] = row
        tables[name] = rows
    res["code_tables"] = tables

    # ---- 2. the replication question --------------------------------------
    rep = {}
    for name, pool in (("insulting_only", ins_all),
                       ("insulting_plus_hateful", ins_all + hat_all)):
        gs = [a for a in pool if coding[a]["code"] == "GS"]
        ex = [a for a in pool if coding[a]["code"] == "EX"]
        os_ = [a for a in pool if coding[a]["code"] == "OS"]
        gs_z, ex_z, os_z = zs(packet, gs), zs(packet, ex), zs(packet, os_)
        entry = {
            "n_GS": len(gs), "n_EX": len(ex), "n_OS": len(os_),
            "auc_GS": auc(gs_z, neg_shipped), "auc_EX": auc(ex_z, neg_shipped),
            "auc_OS": auc(os_z, neg_shipped),
            "ci_GS": list(auc_ci(gs_z, neg_shipped)),
            "ci_EX": list(auc_ci(ex_z, neg_shipped)),
            "ci_OS": list(auc_ci(os_z, neg_shipped)),
            "gap_EX_minus_GS": auc(ex_z, neg_shipped) - auc(gs_z, neg_shipped),
            "auc_GS_audited_negatives": auc(gs_z, neg_audited),
            "auc_EX_audited_negatives": auc(ex_z, neg_audited),
            "mannwhitney_p_GS_vs_EX_scores": mw_p(gs_z, ex_z),
        }
        # GS split by register: does the deficit sit in the mockery register?
        gs_imp = [a for a in gs if int(coding[a]["explicit"]) == 0]
        gs_exp = [a for a in gs if int(coding[a]["explicit"]) == 1]
        entry["GS_mockery_register"] = {
            "n": len(gs_imp), "median_z": med(zs(packet, gs_imp)),
            "auc": auc(zs(packet, gs_imp), neg_shipped)}
        entry["GS_overt_register"] = {
            "n": len(gs_exp), "median_z": med(zs(packet, gs_exp)),
            "auc": auc(zs(packet, gs_exp), neg_shipped)}
        # sensitivity: drop the low-confidence codes
        gs_h = [a for a in gs if coding[a]["conf"] == "h"]
        ex_h = [a for a in ex if coding[a]["conf"] == "h"]
        entry["high_confidence_only"] = {
            "n_GS": len(gs_h), "n_EX": len(ex_h),
            "auc_GS": auc(zs(packet, gs_h), neg_shipped),
            "auc_EX": auc(zs(packet, ex_h), neg_shipped)}
        rep[name] = entry
    res["replication"] = rep

    # ---- 3. how much of the 0.637 is GS-attributable ----------------------
    ins_z = zs(packet, ins_all)
    share = {"stratum_auc_all": auc(ins_z, neg_shipped),
             "stratum_n": len(ins_all)}
    for code in CODES:
        drop = [a for a in ins_all if coding[a]["code"] != code]
        share[f"stratum_auc_excluding_{code}"] = auc(zs(packet, drop), neg_shipped)
        share[f"n_after_excluding_{code}"] = len(drop)
    share["points_attributable_to_GS"] = (share["stratum_auc_excluding_GS"]
                                          - share["stratum_auc_all"])
    # the same arithmetic against the audited negative class
    share["stratum_auc_all_audited"] = auc(ins_z, neg_audited)
    share["stratum_auc_excluding_GS_audited"] = auc(
        zs(packet, [a for a in ins_all if coding[a]["code"] != "GS"]), neg_audited)
    res["gs_attributable_share"] = share

    # ---- 4. pooled cross-corpus description --------------------------------
    mh = mhclip_gender_cohort()
    pooled = {"MHClip_EN": {}, "MHClip_ZH": {}, "HateClipSeg": {}}
    for key in ("EN", "ZH"):
        d = mh[key]
        pooled["MHClip_" + key] = {
            "n_gender_stereotype": len(d["z"]),
            "n_implicit_stratum": d["n_implicit"],
            "median_z": med(d["z"]),
            "median_percentile_in_corpus": med(
                [pct_rank(v, d["pool"]) for v in d["z"]]),
            "auc_vs_corpus_normals": auc(d["z"], d["normals"]),
        }
    hcs_gs = [a for a in packet if coding[a]["code"] == "GS"]
    with open(os.path.join(ROOT,
                           "results/ranking_autopsy/hcs/items.json")) as f:
        hcs_pool = [r["z"] for r in json.load(f)]
    pooled["HateClipSeg"] = {
        "n_gender_stereotype": len(hcs_gs),
        "n_coded": len(packet),
        "median_z": med(zs(packet, hcs_gs)),
        "median_percentile_in_corpus": med(
            [pct_rank(packet[a]["z"], hcs_pool) for a in hcs_gs]),
        "auc_vs_clean_normal": auc(zs(packet, hcs_gs), neg_shipped),
        "note": ("percentile is against all 394 scored HateClipSeg videos; the "
                 "coded set of 149 is not a random sample of that corpus"),
    }
    pooled["combined"] = {
        "n": (pooled["MHClip_EN"]["n_gender_stereotype"]
              + pooled["MHClip_ZH"]["n_gender_stereotype"]
              + pooled["HateClipSeg"]["n_gender_stereotype"]),
        "median_percentile_pooled": med(
            [pct_rank(v, mh["EN"]["pool"]) for v in mh["EN"]["z"]]
            + [pct_rank(v, mh["ZH"]["pool"]) for v in mh["ZH"]["z"]]
            + [pct_rank(packet[a]["z"], hcs_pool) for a in hcs_gs]),
        "median_percentile_mhclip_only": med(
            [pct_rank(v, mh["EN"]["pool"]) for v in mh["EN"]["z"]]
            + [pct_rank(v, mh["ZH"]["pool"]) for v in mh["ZH"]["z"]]),
    }

    # ---- 4b. matched-annotation-stratum contrasts ---------------------------
    # The pooled GS-versus-EX contrast mixes two annotation strata. These two
    # rows hold the annotation stratum fixed and vary only the content code.
    matched = {}
    for name, pool in (("insulting_only", ins_all), ("hateful_sample", hat_all)):
        gs = [a for a in pool if coding[a]["code"] == "GS"]
        rest = [a for a in pool if coding[a]["code"] in ("EX", "OS")]
        matched[name] = {
            "n_GS": len(gs), "n_EX_or_OS": len(rest),
            "auc_GS": auc(zs(packet, gs), neg_shipped),
            "auc_EX_or_OS": auc(zs(packet, rest), neg_shipped),
            "gap": auc(zs(packet, rest), neg_shipped) - auc(zs(packet, gs), neg_shipped),
            "mannwhitney_p_scores": mw_p(zs(packet, gs), zs(packet, rest)),
        }
    res["matched_annotation_stratum"] = matched
    res["cross_corpus_pooled"] = pooled

    # ---- 5. confound checks ------------------------------------------------
    conf = {}
    gs_all = [a for a in ins_all + hat_all if coding[a]["code"] == "GS"]
    ex_all = [a for a in ins_all + hat_all if coding[a]["code"] == "EX"]
    cov_rows = {}
    for key in ("fresh_chars", "vad_speech_frac", "wav_duration", "judge_chars"):
        cov_rows[key] = {
            "median_GS": med(covs(packet, gs_all, key)),
            "median_EX": med(covs(packet, ex_all, key)),
            "mannwhitney_p": mw_p(covs(packet, gs_all, key),
                                  covs(packet, ex_all, key)),
        }
    conf["evidence_covariates_GS_vs_EX"] = cov_rows
    conf["judge_saw_transcript"] = {
        "GS": sum(1 for a in gs_all if packet[a]["judge_saw_transcript"]),
        "GS_n": len(gs_all),
        "EX": sum(1 for a in ex_all if packet[a]["judge_saw_transcript"]),
        "EX_n": len(ex_all),
    }

    # annotation side: where does the corpus itself put the GS videos?
    side = {}
    for code in CODES:
        sel = [a for a in packet if coding[a]["code"] == code]
        side[code] = {
            "n": len(sel),
            "in_hateful_sample": sum(1 for a in sel if packet[a]["cohort"] == "HAT"),
            "in_insulting_only": sum(1 for a in sel if packet[a]["cohort"] == "INS"),
            "in_clean_normal": sum(1 for a in sel if packet[a]["cohort"] == "NRM"),
        }
    conf["annotation_side"] = side

    # within-corpus severity control: are GS videos simply the milder ones?
    conf["hateful_sample_only"] = {
        "auc_GS": auc(zs(packet, [a for a in hat_all if coding[a]["code"] == "GS"]),
                      neg_shipped),
        "auc_EX": auc(zs(packet, [a for a in hat_all if coding[a]["code"] == "EX"]),
                      neg_shipped),
        "n_GS": sum(1 for a in hat_all if coding[a]["code"] == "GS"),
        "n_EX": sum(1 for a in hat_all if coding[a]["code"] == "EX"),
    }
    res["confounds"] = conf

    # ---- 6. reference numbers this note is checked against -----------------
    res["reference"] = {
        "ranking_autopsy_insulting_only_auc": 0.637,
        "mhclip_pooled_implicit_auc": 0.71,
        "implihatevid_implicit_auc": 0.9255,
    }

    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
