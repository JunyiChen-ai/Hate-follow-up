#!/usr/bin/env python3
"""Implicitness-evidence analysis.

Question. The frozen single-call Qwen3-VL-8B judge shows one twice-replicated
model-side weakness: implicit protected-group hostility, at ROC-AUC 0.725 on
MHClip-EN and 0.731 on MHClip-ZH against explicit hostility at 0.983 and 0.917.
ImpliHateVid, whose positive class is built around implicit hate, ranks at 0.947.
This script tests whether the deficit is bound to evidence volume (transcript
length, speech fraction, duration), to the negative class, to implicitness type,
or to small-sample noise.

No model call. No GPU. Label use is evaluative only.

Outputs results/implicitness_analysis/results.json.
"""
import csv
import json
import math
import os
import random
import statistics
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr, mannwhitneyu

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "results", "implicitness_analysis")
SEED = 20260808
N_BOOT = 4000

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
    vals = []
    for _ in range(n_boot):
        p = rng.choice(pos, size=len(pos), replace=True)
        n = rng.choice(neg, size=len(neg), replace=True)
        vals.append(auc(p, n))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def med(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(statistics.median(xs)) if xs else float("nan")


def spear(xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    if ok.sum() < 4:
        return {"rho": float("nan"), "p": float("nan"), "n": int(ok.sum())}
    rho, p = spearmanr(xs[ok], ys[ok])
    return {"rho": float(rho), "p": float(p), "n": int(ok.sum())}


# ---------------------------------------------------------------- loading

def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_jsonl(path, key="video_id"):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r[key]] = r
    return out


def load_en():
    """MHClip-EN test_clean, 161 videos, with the score-aware protected coding."""
    items = load_json(os.path.join(ROOT, "results/ranking_autopsy/en/items.json"))
    by_id = {r["video_id"]: r for r in items}
    alias2id = {}
    for f in ("packet.json", "packet_rest.json"):
        for r in load_json(os.path.join(ROOT, "results/ranking_autopsy/en", f)):
            alias2id[r["alias"]] = r["video_id"]
    flags = load_json(os.path.join(
        ROOT, "results/ranking_autopsy/en/protected_flags.json"))["protected_flag"]
    protected = [a for a, v in flags.items() if v == 1]
    # The autopsy note reports an explicit / implicit split of the 17 protected
    # positives as 7 / 10. No file on disk records that split, so it is
    # reconstructed here as the seven highest-scoring protected positives, which
    # reproduces the note's 0.983 and 0.725 exactly. That reconstruction is
    # itself a finding and is reported as such.
    protected.sort(key=lambda a: -by_id[alias2id[a]]["z"])
    explicit = [alias2id[a] for a in protected[:7]]
    implicit = [alias2id[a] for a in protected[7:]]
    nonprot = [alias2id[a] for a, v in flags.items() if v == 0]
    normals = [r["video_id"] for r in items if r["binary"] == 0]
    return {"by_id": by_id, "explicit": explicit, "implicit": implicit,
            "nonprotected": nonprot, "normals": normals,
            "protected_sorted_aliases": protected}


def load_en_blind_offensive_normals():
    """The MHClip-EN shipped Normals that blind coders called Hateful/Offensive."""
    manifest = load_json(os.path.join(ROOT, "results/annotation_validity/manifest.json"))
    item2vid = {r["item_id"]: r["video_id"] for r in manifest
                if r["corpus"] == "EN" and r["stratum"] == "filler_shipped_normal"}
    flagged = []
    cdir = os.path.join(ROOT, "results/annotation_validity/coding")
    for fn in sorted(os.listdir(cdir)):
        if not fn.endswith(".jsonl"):
            continue
        with open(os.path.join(cdir, fn)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r["item_id"] in item2vid and r.get("label") in ("Hateful", "Offensive"):
                    flagged.append(item2vid[r["item_id"]])
    return sorted(set(flagged))


def load_zh():
    items = load_json(os.path.join(ROOT, "results/ranking_autopsy/zh/items.json"))
    by_id = {r["video_id"]: r for r in items}
    alias2id = {r["alias"]: r["video_id"]
                for r in load_json(os.path.join(ROOT, "results/ranking_autopsy/zh/packet.json"))}
    codes = {}
    with open(os.path.join(ROOT, "results/ranking_autopsy/zh/coding_zh.tsv")) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            codes[row["alias"]] = row
    explicit, implicit, nonprot = [], [], []
    for a, row in codes.items():
        if not a.startswith("ZH-POS"):
            continue
        vid = alias2id[a]
        if row["protected"] == "explicit":
            explicit.append(vid)
        elif row["protected"] == "implicit":
            implicit.append(vid)
        else:
            nonprot.append(vid)
    normals = [r["video_id"] for r in items if r["binary"] == 0]
    harvest = [r["video_id"] for r in items
               if r["binary"] == 0 and 'class="keyword"' in (r.get("title") or "")]
    return {"by_id": by_id, "explicit": explicit, "implicit": implicit,
            "nonprotected": nonprot, "normals": normals,
            "harvest_normals": harvest, "codes": codes, "alias2id": alias2id}


def load_ihv():
    base = os.path.join(ROOT, "results/testruns/implihatevid")
    sc = load_jsonl(os.path.join(base, "judge_8b/scores.jsonl"))
    ft = load_jsonl(os.path.join(base, "fresh_transcripts.jsonl"))
    am = load_jsonl(os.path.join(base, "audio_meta.jsonl"))
    by_id = {}
    for vid, r in sc.items():
        rec = {
            "video_id": vid,
            "z": r["z"],
            "n_transcript_chars": r.get("n_transcript_chars"),
            "fresh_chars": (ft.get(vid) or {}).get("fresh_chars"),
            "vad_speech_frac": (am.get(vid) or {}).get("vad_speech_frac"),
            "wav_duration": (am.get(vid) or {}).get("wav_duration"),
        }
        by_id[vid] = rec
    groups = {"EX": [], "IM": [], "NH": []}
    for vid in by_id:
        groups[vid.split("_")[0]].append(vid)
    return {"by_id": by_id, "explicit": groups["EX"], "implicit": groups["IM"],
            "normals": groups["NH"]}


# ---------------------------------------------------------------- helpers

COVARS = ["fresh_chars", "n_transcript_chars", "vad_speech_frac", "wav_duration"]


def zs(by_id, ids):
    return [by_id[v]["z"] for v in ids]


def cov(by_id, ids, name):
    return [by_id[v].get(name) for v in ids]


def stratum_profile(by_id, ids):
    out = {"n": len(ids), "median_z": med(zs(by_id, ids))}
    for c in COVARS:
        out["median_" + c] = med(cov(by_id, ids, c))
    return out


def covariate_spearman(by_id, ids):
    out = {}
    z = zs(by_id, ids)
    for c in COVARS:
        out[c] = spear(cov(by_id, ids, c), z)
    return out


def bin_of(chars):
    if chars is None or (isinstance(chars, float) and math.isnan(chars)):
        return None
    if chars < 150:
        return "<150"
    if chars < 600:
        return "150-600"
    return ">600"


BINS = ["<150", "150-600", ">600"]


# ---------------------------------------------------------------- main

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    res = {"meta": {
        "date": "2026-08-09",
        "seed": SEED,
        "n_bootstrap": N_BOOT,
        "model_calls": 0,
        "provenance_warning": (
            "MHClip-EN and MHClip-ZH implicit/explicit strata come from "
            "score-aware coding done by a reader who could see the judge "
            "scores. The MHClip-EN split is additionally reconstructed here "
            "from the score order because no file records it. ImpliHateVid "
            "strata are the corpus's own gold id prefixes and are the only "
            "score-blind implicitness labels in this analysis."),
    }}

    en = load_en()
    zh = load_zh()
    ihv = load_ihv()

    # ---- 0. provenance of the MHClip-EN split -------------------------------
    prot_z = sorted((en["by_id"][v]["z"] for v in en["explicit"] + en["implicit"]),
                    reverse=True)
    res["split_provenance"] = {
        "en_protected_n": len(en["explicit"]) + len(en["implicit"]),
        "en_explicit_n": len(en["explicit"]),
        "en_explicit_is_top_k_by_score": True,
        "en_explicit_min_z": min(zs(en["by_id"], en["explicit"])),
        "en_implicit_max_z": max(zs(en["by_id"], en["implicit"])),
        "prob_exact_top7_if_split_independent_of_score": 1.0 / math.comb(17, 7),
        "en_protected_z_sorted": prot_z,
        "zh_explicit_is_top_k_by_score": False,
        "zh_explicit_min_z": min(zs(zh["by_id"], zh["explicit"])),
        "zh_implicit_max_z": max(zs(zh["by_id"], zh["implicit"])),
        "zh_n_implicit_above_explicit_min": sum(
            1 for v in zh["implicit"]
            if zh["by_id"][v]["z"] > min(zs(zh["by_id"], zh["explicit"]))),
    }

    # ---- 1. baseline strata --------------------------------------------------
    corpora = {
        "MHClip_EN": (en["by_id"], en, en["normals"]),
        "MHClip_ZH": (zh["by_id"], zh, zh["normals"]),
        "ImpliHateVid": (ihv["by_id"], ihv, ihv["normals"]),
    }
    baseline = {}
    for name, (by_id, d, negs) in corpora.items():
        entry = {}
        neg_z = zs(by_id, negs)
        for stratum in ("explicit", "implicit", "nonprotected"):
            ids = d.get(stratum)
            if not ids:
                continue
            p = zs(by_id, ids)
            a = auc(p, neg_z)
            lo, hi = auc_ci(p, neg_z)
            entry[stratum] = dict(stratum_profile(by_id, ids), auc=a, ci95=[lo, hi])
        entry["normals"] = stratum_profile(by_id, negs)
        if "explicit" in entry and "implicit" in entry:
            entry["explicit_minus_implicit_auc"] = entry["explicit"]["auc"] - entry["implicit"]["auc"]
        baseline[name] = entry
    res["baseline_strata"] = baseline

    # ---- 1b. covariate Spearman within strata --------------------------------
    corr = {}
    for name, (by_id, d, negs) in corpora.items():
        corr[name] = {}
        for stratum in ("explicit", "implicit", "nonprotected", "normals"):
            ids = d.get(stratum) if stratum != "normals" else negs
            if not ids:
                continue
            corr[name][stratum] = covariate_spearman(by_id, ids)
        allpos = (d.get("explicit") or []) + (d.get("implicit") or []) + (d.get("nonprotected") or [])
        corr[name]["all_positives"] = covariate_spearman(by_id, allpos)
        corr[name]["whole_corpus"] = covariate_spearman(by_id, allpos + negs)
    # pooled MHClip implicit set
    pooled = [("MHClip_EN", v) for v in en["implicit"]] + [("MHClip_ZH", v) for v in zh["implicit"]]
    pooled_by = {"MHClip_EN": en["by_id"], "MHClip_ZH": zh["by_id"]}
    pooled_z = [pooled_by[c][v]["z"] for c, v in pooled]
    pooled_corr = {}
    for c in COVARS:
        pooled_corr[c] = spear([pooled_by[cc][v].get(c) for cc, v in pooled], pooled_z)
    corr["MHClip_pooled_implicit"] = pooled_corr
    pooled_exp = [("MHClip_EN", v) for v in en["explicit"]] + [("MHClip_ZH", v) for v in zh["explicit"]]
    pooled_exp_z = [pooled_by[c][v]["z"] for c, v in pooled_exp]
    corr["MHClip_pooled_explicit"] = {
        c: spear([pooled_by[cc][v].get(c) for cc, v in pooled_exp], pooled_exp_z)
        for c in COVARS}
    res["covariate_spearman"] = corr
    res["pooled_implicit_profile"] = {
        "n": len(pooled),
        "median_z": med(pooled_z),
        "median_fresh_chars": med([pooled_by[c][v].get("fresh_chars") for c, v in pooled]),
    }

    # ---- 2. evidence-matched comparison --------------------------------------
    rng = random.Random(SEED)
    ihv_im_sample = sorted(ihv["implicit"])
    rng.shuffle(ihv_im_sample)
    ihv_im_sample = ihv_im_sample[:40]

    matched = {}
    matched_defs = {
        "MHClip_EN": (en["by_id"], en["implicit"], en["normals"]),
        "MHClip_ZH": (zh["by_id"], zh["implicit"], zh["normals"]),
        "ImpliHateVid_sample40": (ihv["by_id"], ihv_im_sample, ihv["normals"]),
        "ImpliHateVid_full": (ihv["by_id"], ihv["implicit"], ihv["normals"]),
    }
    for name, (by_id, pos, negs) in matched_defs.items():
        cells = {}
        for b in BINS:
            p = [v for v in pos if bin_of(by_id[v].get("fresh_chars")) == b]
            n = [v for v in negs if bin_of(by_id[v].get("fresh_chars")) == b]
            cell = {
                "n_pos": len(p), "n_neg": len(n),
                "median_z_pos": med(zs(by_id, p)),
                "median_z_neg": med(zs(by_id, n)),
                "unpowered": len(p) < 5 or len(n) < 5,
            }
            if p and n:
                cell["auc_vs_same_bin_normals"] = auc(zs(by_id, p), zs(by_id, n))
                lo, hi = auc_ci(zs(by_id, p), zs(by_id, n))
                cell["ci95"] = [lo, hi]
                cell["auc_vs_all_normals"] = auc(zs(by_id, p), zs(by_id, negs))
            cells[b] = cell
        matched[name] = cells
    # explicit contrast, same binning
    matched_explicit = {}
    for name, (by_id, d) in {"MHClip_EN": (en["by_id"], en),
                             "MHClip_ZH": (zh["by_id"], zh),
                             "ImpliHateVid": (ihv["by_id"], ihv)}.items():
        cells = {}
        negs = d["normals"]
        for b in BINS:
            p = [v for v in d["explicit"] if bin_of(by_id[v].get("fresh_chars")) == b]
            n = [v for v in negs if bin_of(by_id[v].get("fresh_chars")) == b]
            cell = {"n_pos": len(p), "n_neg": len(n),
                    "median_z_pos": med(zs(by_id, p)),
                    "unpowered": len(p) < 5 or len(n) < 5}
            if p and n:
                cell["auc_vs_same_bin_normals"] = auc(zs(by_id, p), zs(by_id, n))
            cells[b] = cell
        matched_explicit[name] = cells
    res["evidence_matched"] = {"implicit": matched, "explicit_contrast": matched_explicit,
                               "bins": BINS,
                               "ihv_sample_seed": SEED,
                               "ihv_sample_n": len(ihv_im_sample)}

    # pooled MHClip implicit against pooled MHClip normals, per bin, corpus-internal
    pooled_cells = {}
    for b in BINS:
        n_pos = sum(1 for c, v in pooled if bin_of(pooled_by[c][v].get("fresh_chars")) == b)
        # per-corpus AUC then pool by pair weight
        num = 0.0
        den = 0.0
        for cname, d, by_id in (("MHClip_EN", en, en["by_id"]), ("MHClip_ZH", zh, zh["by_id"])):
            p = [v for v in d["implicit"] if bin_of(by_id[v].get("fresh_chars")) == b]
            n = [v for v in d["normals"] if bin_of(by_id[v].get("fresh_chars")) == b]
            if p and n:
                w = len(p) * len(n)
                num += w * auc(zs(by_id, p), zs(by_id, n))
                den += w
        pooled_cells[b] = {"n_pos": n_pos,
                           "pair_weighted_auc": (num / den) if den else float("nan"),
                           "n_pairs": den,
                           "unpowered": n_pos < 5}
    res["evidence_matched"]["mhclip_pooled_implicit_by_bin"] = pooled_cells

    # ---- 3. negative-class control -------------------------------------------
    en_bad = load_en_blind_offensive_normals()
    en_clean = [v for v in en["normals"] if v not in set(en_bad)]
    zh_harvest = set(zh["harvest_normals"])
    zh_plain = [v for v in zh["normals"] if v not in zh_harvest]

    negctl = {"MHClip_EN": {}, "MHClip_ZH": {}}
    for stratum in ("explicit", "implicit", "nonprotected"):
        p = zs(en["by_id"], en[stratum])
        a_all = auc(p, zs(en["by_id"], en["normals"]))
        a_clean = auc(p, zs(en["by_id"], en_clean))
        lo, hi = auc_ci(p, zs(en["by_id"], en_clean))
        negctl["MHClip_EN"][stratum] = {
            "auc_all_normals": a_all, "auc_clean_normals": a_clean,
            "ci95_clean": [lo, hi], "delta": a_clean - a_all}
    negctl["MHClip_EN"]["n_normals_all"] = len(en["normals"])
    negctl["MHClip_EN"]["n_normals_clean"] = len(en_clean)
    negctl["MHClip_EN"]["n_removed"] = len(en_bad)
    negctl["MHClip_EN"]["median_z_removed"] = med(zs(en["by_id"], en_bad))
    negctl["MHClip_EN"]["median_z_kept"] = med(zs(en["by_id"], en_clean))

    for stratum in ("explicit", "implicit", "nonprotected"):
        p = zs(zh["by_id"], zh[stratum])
        a_all = auc(p, zs(zh["by_id"], zh["normals"]))
        a_clean = auc(p, zs(zh["by_id"], zh_plain))
        lo, hi = auc_ci(p, zs(zh["by_id"], zh_plain))
        negctl["MHClip_ZH"][stratum] = {
            "auc_all_normals": a_all, "auc_clean_normals": a_clean,
            "ci95_clean": [lo, hi], "delta": a_clean - a_all}
    negctl["MHClip_ZH"]["n_normals_all"] = len(zh["normals"])
    negctl["MHClip_ZH"]["n_normals_clean"] = len(zh_plain)
    negctl["MHClip_ZH"]["n_removed"] = len(zh_harvest)
    negctl["MHClip_ZH"]["median_z_removed"] = med(zs(zh["by_id"], sorted(zh_harvest)))
    negctl["MHClip_ZH"]["median_z_kept"] = med(zs(zh["by_id"], zh_plain))
    for c in negctl:
        if "explicit" in negctl[c] and "implicit" in negctl[c]:
            negctl[c]["gap_all_normals"] = (negctl[c]["explicit"]["auc_all_normals"]
                                            - negctl[c]["implicit"]["auc_all_normals"])
            negctl[c]["gap_clean_normals"] = (negctl[c]["explicit"]["auc_clean_normals"]
                                              - negctl[c]["implicit"]["auc_clean_normals"])
    res["negative_class_control"] = negctl

    # combined control: clean negatives AND length-matched
    combo = {}
    for cname, by_id, imp, cleanneg in (("MHClip_EN", en["by_id"], en["implicit"], en_clean),
                                        ("MHClip_ZH", zh["by_id"], zh["implicit"], zh_plain)):
        cells = {}
        for b in BINS:
            p = [v for v in imp if bin_of(by_id[v].get("fresh_chars")) == b]
            n = [v for v in cleanneg if bin_of(by_id[v].get("fresh_chars")) == b]
            cell = {"n_pos": len(p), "n_neg": len(n), "unpowered": len(p) < 5 or len(n) < 5}
            if p and n:
                cell["auc"] = auc(zs(by_id, p), zs(by_id, n))
            cells[b] = cell
        num = sum(c["n_pos"] * c["n_neg"] * c["auc"] for c in cells.values() if "auc" in c)
        den = sum(c["n_pos"] * c["n_neg"] for c in cells.values() if "auc" in c)
        combo[cname] = {"cells": cells,
                        "pair_weighted_auc": (num / den) if den else float("nan")}
    res["clean_and_length_matched"] = combo

    # ---- 4. ImpliHateVid cross-check ----------------------------------------
    ihv_by = ihv["by_id"]
    ihv_pos = ihv["explicit"] + ihv["implicit"]
    ihv_check = {
        "spearman_z_vs_covariate": {
            "all_positives": covariate_spearman(ihv_by, ihv_pos),
            "implicit_positives": covariate_spearman(ihv_by, ihv["implicit"]),
            "explicit_positives": covariate_spearman(ihv_by, ihv["explicit"]),
            "normals": covariate_spearman(ihv_by, ihv["normals"]),
        },
        "auc_explicit": auc(zs(ihv_by, ihv["explicit"]), zs(ihv_by, ihv["normals"])),
        "auc_implicit": auc(zs(ihv_by, ihv["implicit"]), zs(ihv_by, ihv["normals"])),
    }
    lo, hi = auc_ci(zs(ihv_by, ihv["implicit"]), zs(ihv_by, ihv["normals"]))
    ihv_check["auc_implicit_ci95"] = [lo, hi]
    lo, hi = auc_ci(zs(ihv_by, ihv["explicit"]), zs(ihv_by, ihv["normals"]))
    ihv_check["auc_explicit_ci95"] = [lo, hi]
    bottom = sorted(ihv_pos, key=lambda v: ihv_by[v]["z"])[:20]
    ihv_check["bottom20"] = {
        "n": len(bottom),
        "n_implicit": sum(1 for v in bottom if v.startswith("IM")),
        "n_explicit": sum(1 for v in bottom if v.startswith("EX")),
        "median_z": med(zs(ihv_by, bottom)),
        "profile": stratum_profile(ihv_by, bottom),
        "cohort_profile": stratum_profile(ihv_by, ihv_pos),
    }
    # is the bottom-20 evidence-thin relative to the rest of the positives?
    rest = [v for v in ihv_pos if v not in set(bottom)]
    for c in COVARS:
        a = [x for x in cov(ihv_by, bottom, c) if x is not None]
        b = [x for x in cov(ihv_by, rest, c) if x is not None]
        if len(a) > 2 and len(b) > 2:
            u, p = mannwhitneyu(a, b, alternative="two-sided")
            ihv_check["bottom20"].setdefault("mannwhitney_vs_rest", {})[c] = {
                "median_bottom20": med(a), "median_rest": med(b), "p": float(p)}
    # thin-evidence subset of IHV: implicit positives with short transcripts
    thin = [v for v in ihv["implicit"] if (ihv_by[v].get("fresh_chars") or 0) < 150]
    thin_neg = [v for v in ihv["normals"] if (ihv_by[v].get("fresh_chars") or 0) < 150]
    ihv_check["thin_evidence_subset"] = {
        "n_implicit_under150": len(thin), "n_normals_under150": len(thin_neg),
        "median_z_implicit": med(zs(ihv_by, thin)),
        "auc_vs_thin_normals": auc(zs(ihv_by, thin), zs(ihv_by, thin_neg)) if thin and thin_neg else float("nan"),
        "auc_vs_all_normals": auc(zs(ihv_by, thin), zs(ihv_by, ihv["normals"])) if thin else float("nan"),
    }
    res["ihv_cross_check"] = ihv_check

    # ---- 5. hypothesis arithmetic -------------------------------------------
    arith = {}
    for cname, d, by_id, cleanneg in (("MHClip_EN", en, en["by_id"], en_clean),
                                      ("MHClip_ZH", zh, zh["by_id"], zh_plain)):
        gap = (auc(zs(by_id, d["explicit"]), zs(by_id, d["normals"]))
               - auc(zs(by_id, d["implicit"]), zs(by_id, d["normals"])))
        gap_clean = (auc(zs(by_id, d["explicit"]), zs(by_id, cleanneg))
                     - auc(zs(by_id, d["implicit"]), zs(by_id, cleanneg)))
        imp_all = auc(zs(by_id, d["implicit"]), zs(by_id, d["normals"]))
        imp_clean = auc(zs(by_id, d["implicit"]), zs(by_id, cleanneg))
        imp_matched = combo[cname]["pair_weighted_auc"]
        arith[cname] = {
            "explicit_minus_implicit_shipped_negatives": gap,
            "explicit_minus_implicit_clean_negatives": gap_clean,
            "negative_class_points_of_gap": gap - gap_clean,
            "implicit_auc_shipped": imp_all,
            "implicit_auc_clean_negatives": imp_clean,
            "implicit_auc_clean_and_length_matched": imp_matched,
            "length_matching_points_on_top_of_clean": imp_matched - imp_clean,
        }
    # cross-corpus gap: IHV implicit minus MHClip implicit
    arith["cross_corpus"] = {
        "ihv_implicit_auc": ihv_check["auc_implicit"],
        "en_implicit_auc": auc(zs(en["by_id"], en["implicit"]), zs(en["by_id"], en["normals"])),
        "zh_implicit_auc": auc(zs(zh["by_id"], zh["implicit"]), zs(zh["by_id"], zh["normals"])),
        "ihv_within_corpus_explicit_implicit_gap": ihv_check["auc_explicit"] - ihv_check["auc_implicit"],
        "en_within_corpus_gap": baseline["MHClip_EN"]["explicit_minus_implicit_auc"],
        "zh_within_corpus_gap": baseline["MHClip_ZH"]["explicit_minus_implicit_auc"],
    }
    res["hypothesis_arithmetic"] = arith

    # ---- 5b. implicitness-type composition ----------------------------------
    # Coarse attribute tally over the score-aware coder notes. Keyword rules
    # only; no video identifier and no note text leaves this function.
    GENDER = ("gender", "women", "woman", "men ", "misogyn", "sexual-orientation",
              "same-sex", "misgender", "effeminate", "gender-expression",
              "gender-identity", "sexual shaming")
    ETHNO = ("racial", "race", "nationality", "national-origin", "language and",
             "foreign", "ethnic", "haka", "countryball", "antisemit")
    RELIG = ("religious", "scriptural", "funeral custom")

    def tally(notes):
        out = {"n": len(notes), "gender_or_sexuality": 0, "race_nationality_language": 0,
               "religion_or_custom": 0}
        for t in notes:
            t = t.lower()
            if any(k in t for k in GENDER):
                out["gender_or_sexuality"] += 1
            if any(k in t for k in ETHNO):
                out["race_nationality_language"] += 1
            if any(k in t for k in RELIG):
                out["religion_or_custom"] += 1
        return out

    en_notes = {}
    with open(os.path.join(ROOT, "results/ranking_autopsy/coding_en_protected.tsv")) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            en_notes[row["alias"]] = row["note"]
    en_alias = {}
    for f_ in ("packet.json", "packet_rest.json"):
        for r in load_json(os.path.join(ROOT, "results/ranking_autopsy/en", f_)):
            en_alias[r["video_id"]] = r["alias"]
    zh_note_by_id = {zh["alias2id"][a]: row["note"] for a, row in zh["codes"].items()
                     if a in zh["alias2id"]}
    types = {
        "MHClip_EN_implicit": tally([en_notes[en_alias[v]] for v in en["implicit"]]),
        "MHClip_EN_explicit": tally([en_notes[en_alias[v]] for v in en["explicit"]]),
        "MHClip_ZH_implicit": tally([zh_note_by_id[v] for v in zh["implicit"]]),
        "MHClip_ZH_explicit": tally([zh_note_by_id[v] for v in zh["explicit"]]),
    }
    types["MHClip_pooled_implicit"] = {
        k: (types["MHClip_EN_implicit"][k] + types["MHClip_ZH_implicit"][k])
        for k in types["MHClip_EN_implicit"]}
    types["MHClip_pooled_explicit"] = {
        k: (types["MHClip_EN_explicit"][k] + types["MHClip_ZH_explicit"][k])
        for k in types["MHClip_EN_explicit"]}
    types["note"] = ("ImpliHateVid's implicit stratum is the corpus's own gold "
                     "IM prefix and carries no attribute coding here; the corpus "
                     "is built from United States social-media hate speech and "
                     "its implicit cases are predominantly race, ethnicity and "
                     "immigration directed rather than gender-stereotype humour.")
    res["implicitness_type_composition"] = types

    # ---- 5c. cross-corpus deficit decomposition ------------------------------
    ihv_imp = ihv_check["auc_implicit"]
    decomp = {}
    for cname, d, by_id, cleanneg in (("MHClip_EN", en, en["by_id"], en_clean),
                                      ("MHClip_ZH", zh, zh["by_id"], zh_plain)):
        shipped = auc(zs(by_id, d["implicit"]), zs(by_id, d["normals"]))
        clean = auc(zs(by_id, d["implicit"]), zs(by_id, cleanneg))
        matched_ = combo[cname]["pair_weighted_auc"]
        decomp[cname] = {
            "deficit_vs_ihv_implicit": ihv_imp - shipped,
            "explained_by_negative_class": clean - shipped,
            "explained_by_length_matching_on_top": matched_ - clean,
            "residual": ihv_imp - matched_,
        }
    res["cross_corpus_deficit_decomposition"] = decomp

    # small-n noise: overlap of implicit CI with explicit CI, and with IHV
    noise = {}
    for cname in ("MHClip_EN", "MHClip_ZH", "ImpliHateVid"):
        e = baseline[cname]["explicit"]
        i = baseline[cname]["implicit"]
        noise[cname] = {
            "explicit_n": e["n"], "explicit_auc": e["auc"], "explicit_ci95": e["ci95"],
            "implicit_n": i["n"], "implicit_auc": i["auc"], "implicit_ci95": i["ci95"],
            "ci_overlap": not (i["ci95"][1] < e["ci95"][0] or e["ci95"][1] < i["ci95"][0]),
        }
    # does the MHClip implicit AUC's CI contain the IHV implicit AUC?
    for cname in ("MHClip_EN", "MHClip_ZH"):
        ci = baseline[cname]["implicit"]["ci95"]
        noise[cname]["ci_contains_ihv_implicit_auc"] = ci[0] <= ihv_check["auc_implicit"] <= ci[1]
    res["small_n_noise"] = noise

    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(res, f, indent=2, sort_keys=False)
    print(json.dumps({k: res[k] for k in ("split_provenance", "hypothesis_arithmetic")},
                     indent=2)[:4000])
    print("wrote", os.path.join(OUT_DIR, "results.json"))


if __name__ == "__main__":
    main()
