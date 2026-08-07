"""Analysis for the blind annotation-validity audit.

Run only after every coding batch is on disk. This is the first step in the
audit that opens a judge score file, which is why it is a separate script from
the manifest builder.

Frozen decision rule (docs/duplex/PREREG_annotation_validity_audit.md):
  claim 1 CONFIRMED iff the blind-coded rate of hateful-toward-protected-group
          or extremist-glorification among the 50 HateClipSeg clean normals is
          at least 0.20.
  claim 2 CONFIRMED iff the blind-coded rate of "no protected group targeted"
          among the 49 MHClip-EN union positives is at least 0.40.
"Cannot determine" items leave the numerator and the denominator both.

Writes results/annotation_validity/results.json.
"""

import glob
import json
import math
import os

import numpy as np
from scipy.stats import beta

ROOT = "/home/jehc223/Hate-follow-up"
OUT = os.path.join(ROOT, "results", "annotation_validity")
CLAIM1_FLOOR = 0.20
CLAIM2_FLOOR = 0.40
ALPHA = 0.05

HCS_SCORES = os.path.join(ROOT, "results", "hateclipseg", "judge_8b", "scores.jsonl")
EN_SCORES = os.path.join(ROOT, "results", "testruns", "mhclip_en", "judge_8b",
                         "scores.jsonl")


def clopper_pearson(k, n):
    if n == 0:
        return [None, None]
    lo = 0.0 if k == 0 else float(beta.ppf(ALPHA / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - ALPHA / 2, k + 1, n - k))
    return [lo, hi]


def proportion(k, n):
    k, n = int(k), int(n)
    ci = clopper_pearson(k, n)
    return {"k": k, "n": n, "rate": (k / n) if n else None,
            "ci95_lower": ci[0], "ci95_upper": ci[1]}


def auc(scores, labels):
    """Mann-Whitney ROC-AUC with midranks for ties."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if npos == 0 or nneg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(s)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def load_z(path):
    d = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if isinstance(r.get("z"), (int, float)) and math.isfinite(r["z"]):
                d[r["video_id"]] = float(r["z"])
    return d


def load_coding():
    codes = {}
    for p in sorted(glob.glob(os.path.join(OUT, "coding", "*.jsonl"))):
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["item_id"] in codes:
                raise SystemExit(f"duplicate coding for {r['item_id']}")
            r["_batch"] = os.path.basename(p)
            codes[r["item_id"]] = r
    return codes


def main():
    manifest = json.load(open(os.path.join(OUT, "manifest.json")))
    codes = load_coding()
    missing = [m["item_id"] for m in manifest if m["item_id"] not in codes]
    if missing:
        raise SystemExit(f"ABORT: {len(missing)} items uncoded, e.g. {missing[:5]}")

    for m in manifest:
        m["coding"] = codes[m["item_id"]]

    hcs = [m for m in manifest if m["corpus"] == "HCS"]
    en = [m for m in manifest if m["corpus"] == "EN"]

    # ---- HateClipSeg coding -------------------------------------------------
    def hcs_flags(m):
        c = set(m["coding"].get("codes") or [])
        return {
            "cd": "cannot_determine" in c,
            "hate_or_extremist": bool(c & {"hateful_protected_group",
                                           "extremist_glorification"}),
            "hate": "hateful_protected_group" in c,
            "extremist": "extremist_glorification" in c,
            "any_offensive": bool(c & {"hateful_protected_group",
                                       "extremist_glorification",
                                       "other_offensive"}),
        }

    for m in hcs:
        m["flags"] = hcs_flags(m)

    hcs_target = [m for m in hcs if m["stratum"] == "target_clean_normal"]
    hcs_filler = [m for m in hcs if m["stratum"] == "filler_shipped_positive"]

    t_dec = [m for m in hcs_target if not m["flags"]["cd"]]
    claim1 = proportion(sum(m["flags"]["hate_or_extremist"] for m in t_dec), len(t_dec))
    claim1_verdict = ("CONFIRMED" if (claim1["rate"] or 0) >= CLAIM1_FLOOR
                      else "RETRACTED")

    f_dec = [m for m in hcs_filler if not m["flags"]["cd"]]
    filler_hcs = proportion(sum(m["flags"]["any_offensive"] for m in f_dec), len(f_dec))
    filler_hcs_hate = proportion(
        sum(m["flags"]["hate_or_extremist"] for m in f_dec), len(f_dec))

    # ---- MHClip-EN coding ---------------------------------------------------
    for m in en:
        lab = m["coding"].get("label")
        m["flags"] = {
            "cd": lab == "cannot_determine",
            "label": lab,
            "positive": lab in ("Hateful", "Offensive"),
            "hateful": lab == "Hateful",
            "targeted": m["coding"].get("protected_group_targeted"),
        }

    en_target = [m for m in en if m["stratum"] == "target_union_positive"]
    en_filler = [m for m in en if m["stratum"] == "filler_shipped_normal"]

    e_dec = [m for m in en_target if not m["flags"]["cd"]]
    claim2 = proportion(sum(1 for m in e_dec if m["flags"]["targeted"] is False),
                        len(e_dec))
    claim2_verdict = ("CONFIRMED" if (claim2["rate"] or 0) >= CLAIM2_FLOOR
                      else "RETRACTED")

    ef_dec = [m for m in en_filler if not m["flags"]["cd"]]
    filler_en = proportion(sum(m["flags"]["positive"] for m in ef_dec), len(ef_dec))
    en_target_agree = proportion(sum(m["flags"]["positive"] for m in e_dec), len(e_dec))

    # ---- secondary: judge AUC against shipped vs blind-coded labels ---------
    zh = load_z(HCS_SCORES)
    ze = load_z(EN_SCORES)

    def auc_pair(items, z, shipped_fn, blind_fn):
        keep = [m for m in items if not m["flags"]["cd"] and m["video_id"] in z]
        s = [z[m["video_id"]] for m in keep]
        return {
            "n": len(keep),
            "auc_shipped": auc(s, [shipped_fn(m) for m in keep]),
            "auc_blind": auc(s, [blind_fn(m) for m in keep]),
            "n_pos_shipped": int(sum(shipped_fn(m) for m in keep)),
            "n_pos_blind": int(sum(blind_fn(m) for m in keep)),
        }

    OFF = {"hateful", "insulting", "sexual", "violence", "harm"}
    sec = {
        "hateclipseg_union": auc_pair(
            hcs, zh,
            lambda m: int(bool(set(m["shipped_label"]) & OFF)),
            lambda m: int(m["flags"]["any_offensive"])),
        "hateclipseg_strict_hate": auc_pair(
            hcs, zh,
            lambda m: int("hateful" in m["shipped_label"]),
            lambda m: int(m["flags"]["hate_or_extremist"])),
        "mhclip_en_union": auc_pair(
            en, ze,
            lambda m: int(m["shipped_label"] in ("Hateful", "Offensive")),
            lambda m: int(m["flags"]["positive"])),
        "mhclip_en_strict_hate": auc_pair(
            en, ze,
            lambda m: int(m["shipped_label"] == "Hateful"),
            lambda m: int(m["flags"]["hateful"])),
        "mhclip_en_protected_target": {
            "note": "blind arm scores the protected-group-target boolean directly",
            **auc_pair(
                en, ze,
                lambda m: int(m["shipped_label"] in ("Hateful", "Offensive")),
                lambda m: int(bool(m["flags"]["targeted"]))),
        },
    }

    # ---- the autopsy's corrected-corpus arithmetic, redone on blind codes ---
    # HateClipSeg: full corpus under the offensive union, with the blind-flagged
    # negatives dropped from the negative class. Positives are the shipped ones.
    import ast
    import csv as _csv
    hlabs = {}
    with open(os.path.join(ROOT, "idea-stage", "pilots", "b1_coverage_audit",
                           "data", "video_level_annotation.csv")) as f:
        for row in _csv.DictReader(f):
            hlabs[row["Video Id"].strip()] = ast.literal_eval(row["Video-Level Label"])
    hids = [v for v in zh if v in hlabs]
    flag = {m["video_id"]: m["flags"] for m in hcs_target}

    def hcs_corpus_auc(drop):
        keep = [v for v in hids if v not in drop]
        return {"n": len(keep),
                "auc": auc([zh[v] for v in keep],
                           [int(bool(set(hlabs[v]) & OFF)) for v in keep])}

    drop_hx = {v for v, f in flag.items() if f["hate_or_extremist"]}
    drop_cd = {v for v, f in flag.items() if f["cd"]}
    corrected = {
        "hateclipseg_union_as_shipped": hcs_corpus_auc(set()),
        "hateclipseg_union_blind_hx_negatives_removed": hcs_corpus_auc(drop_hx),
        "hateclipseg_union_blind_hx_and_cannot_determine_removed":
            hcs_corpus_auc(drop_hx | drop_cd),
        "n_negatives_removed": len(drop_hx),
        "n_cannot_determine_negatives": len(drop_cd),
    }

    # MHClip-EN: full corpus, positive class restricted to the blind-coded
    # protected-target positives, negatives are the shipped Normals.
    enann = {m["video_id"]: m for m in en_target}
    en_all = [m["video_id"] for m in
              json.load(open(os.path.join(OUT, "manifest.json")))
              if m["corpus"] == "EN"]
    shipped_norm = [v for v in ze if v not in {m["video_id"] for m in en}] + \
                   [m["video_id"] for m in en_filler]
    # every shipped Normal in the scored split, coded or not
    en_shipped = {}
    for line in open(EN_SCORES):
        line = line.strip()
        if line:
            r = json.loads(line)
            en_shipped[r["video_id"]] = None
    ann_en = {x["Video_ID"]: x["Label"] for x in json.load(
        open("/home/jehc223/data/Multihateclip/English/annotation(new).json"))}
    norms = [v for v in ze if ann_en.get(v) == "Normal"]
    pos_all = [m["video_id"] for m in en_target]
    pos_pt = [m["video_id"] for m in en_target if m["flags"]["targeted"] is True]

    def en_corpus_auc(pos):
        vids = list(pos) + norms
        return {"n": len(vids), "n_pos": len(pos),
                "auc": auc([ze[v] for v in vids],
                           [1] * len(pos) + [0] * len(norms))}

    corrected["mhclip_en_union_as_shipped"] = en_corpus_auc(pos_all)
    corrected["mhclip_en_blind_protected_target_positives_only"] = en_corpus_auc(pos_pt)

    # ---- duplicate-transcript audit, plus its overlap with the target set ---
    dup = json.load(open(os.path.join(OUT, "duplicates.json")))
    target_ids = {m["video_id"] for m in hcs_target}
    coded = {m["video_id"]: m for m in hcs}
    dup_summary = {k: v for k, v in dup.items() if k != "groups"}
    dup_summary["groups"] = []
    n_touch, n_blind_agree, n_blind_pairs = 0, 0, 0
    for g in dup["groups"]:
        vids = g["video_ids"]
        g2 = {k: v for k, v in g.items() if k != "video_ids"}
        g2["n_in_claim1_stratum"] = sum(1 for v in vids if v in target_ids)
        if g2["n_in_claim1_stratum"]:
            n_touch += 1
        blind = [coded[v]["flags"]["any_offensive"] for v in vids
                 if v in coded and not coded[v]["flags"]["cd"]]
        g2["n_blind_coded"] = len(blind)
        g2["blind_labels"] = [int(b) for b in blind]
        if len(blind) == len(vids):
            n_blind_pairs += 1
            if len(set(blind)) == 1:
                n_blind_agree += 1
        dup_summary["groups"].append(g2)
    dup_summary["n_groups_touching_claim1_stratum"] = n_touch
    dup_summary["n_groups_fully_blind_coded"] = n_blind_pairs
    dup_summary["n_groups_fully_blind_coded_agreeing"] = n_blind_agree

    res = {
        "protocol": "docs/duplex/PREREG_annotation_validity_audit.md",
        "seed": 20260808,
        "n_items_coded": len(manifest),
        "claim1": {
            "statement": "HateClipSeg clean normals contain hateful or "
                         "extremist-glorifying content",
            "floor": CLAIM1_FLOOR,
            "n_stratum": len(hcs_target),
            "n_cannot_determine": len(hcs_target) - len(t_dec),
            **claim1,
            "verdict": claim1_verdict,
            "breakdown": {
                "hateful_protected_group": proportion(
                    sum(m["flags"]["hate"] for m in t_dec), len(t_dec)),
                "extremist_glorification": proportion(
                    sum(m["flags"]["extremist"] for m in t_dec), len(t_dec)),
                "any_offensive_incl_other": proportion(
                    sum(m["flags"]["any_offensive"] for m in t_dec), len(t_dec)),
            },
        },
        "claim2": {
            "statement": "MHClip-EN union positives carry no protected-group target",
            "floor": CLAIM2_FLOOR,
            "n_stratum": len(en_target),
            "n_cannot_determine": len(en_target) - len(e_dec),
            **claim2,
            "verdict": claim2_verdict,
            "breakdown": {
                "coded_positive_agreeing_with_shipped": en_target_agree,
                "coded_hateful": proportion(
                    sum(m["flags"]["hateful"] for m in e_dec), len(e_dec)),
                "coded_normal": proportion(
                    sum(1 for m in e_dec if m["flags"]["label"] == "Normal"),
                    len(e_dec)),
            },
        },
        "fillers": {
            "hateclipseg_shipped_positives_coded_any_offensive": filler_hcs,
            "hateclipseg_shipped_positives_coded_hate_or_extremist": filler_hcs_hate,
            "hateclipseg_filler_n_cannot_determine": len(hcs_filler) - len(f_dec),
            "mhclip_en_shipped_normals_coded_positive": filler_en,
            "mhclip_en_filler_n_cannot_determine": len(en_filler) - len(ef_dec),
        },
        "cannot_determine_total": sum(1 for m in manifest if m["flags"]["cd"]),
        "secondary_auc": sec,
        "corrected_corpus_auc": corrected,
        "duplicate_audit": dup_summary,
        "duplicate_groups_touching_claim1_stratum": sum(
            1 for g in dup["groups"] if g.get("_touches", False)),
    }
    del res["duplicate_groups_touching_claim1_stratum"]

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
        f.write("\n")

    printable = {k: v for k, v in res.items() if k != "duplicate_audit"}
    print(json.dumps(printable, indent=1))
    print("\nduplicate audit:",
          json.dumps({k: v for k, v in dup_summary.items() if k != "groups"}))


if __name__ == "__main__":
    main()
