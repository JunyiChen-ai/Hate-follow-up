"""Ranking-error autopsy, stage 1: stratified AUC decomposition.

Diagnostic only. No preregistration, no model call, no method. Reads the frozen
single-call Qwen3-VL-8B judge scores for MHClip-EN (test_clean) and HateClipSeg
and decomposes the corpus-level ROC-AUC into class-pair strata, so that a weak
overall number can be attributed to a specific pair of annotation classes.

Outputs machine-readable JSON plus a working item table under
results/ranking_autopsy/{en,hcs}/. The item tables carry video ids and stay
gitignored; only aggregate statistics reach the committed note.
"""

import ast
import csv
import json
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(ROOT, "src", "our_method"))
os.environ.setdefault("HVD_DATA_ROOT", "/home/jehc223/data")

from data_utils import load_annotations, load_clean_split_ids  # noqa: E402

OUT = os.path.join(ROOT, "results", "ranking_autopsy")
HCS_DATA = os.path.join(ROOT, "idea-stage", "pilots", "b1_coverage_audit", "data")
IDX = ["normal", "hateful", "insulting", "sexual", "violence", "harm"]
OFFENSIVE_DIMS = [1, 2, 3, 4, 5]


# ------------------------------------------------------------------ helpers --
def auc(pos, neg):
    """Mann-Whitney ROC-AUC with ties counted at one half."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(allv, kind="stable"), kind="stable").astype(float)
    # average ranks for ties
    order = np.argsort(allv, kind="stable")
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = np.mean(r[order[i:j + 1]])
        i = j + 1
    rp = r[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def auc_ci(pos, neg, n_boot=2000, seed=20260808):
    a = auc(pos, neg)
    if a is None:
        return None, None, None
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    bs = []
    for _ in range(n_boot):
        bs.append(auc(rng.choice(pos, len(pos), replace=True),
                      rng.choice(neg, len(neg), replace=True)))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return a, float(lo), float(hi)


def quart(v):
    v = np.asarray(v, float)
    if len(v) == 0:
        return {}
    return {"n": int(len(v)), "min": float(v.min()), "q1": float(np.percentile(v, 25)),
            "median": float(np.median(v)), "q3": float(np.percentile(v, 75)),
            "max": float(v.max()), "mean": float(v.mean())}


def read_jsonl(path, key="video_id"):
    d = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            d[o[key]] = o
    return d


# ---------------------------------------------------------------- MHClip-EN --
def do_en():
    scores = read_jsonl(os.path.join(ROOT, "results/testruns/mhclip_en/judge_8b/scores.jsonl"))
    trans = read_jsonl(os.path.join(ROOT, "results/testruns/mhclip_en/fresh_transcripts.jsonl"))
    ann = load_annotations("MHClip_EN")
    ids = load_clean_split_ids("MHClip_EN", "test")
    seen, use = set(), []
    for v in ids:
        if v in seen or v not in scores:
            continue
        seen.add(v)
        use.append(v)

    items = []
    for v in use:
        s = scores[v]
        t = trans.get(v, {})
        lab = ann[v]["label"]
        items.append({
            "video_id": v, "z": s["z"], "fine": lab,
            "binary": 1 if lab in ("Hateful", "Offensive") else 0,
            "n_transcript_chars": s.get("n_transcript_chars"),
            "transcript_source": s.get("transcript_source"),
            "fresh_chars": t.get("fresh_chars"), "top_language": t.get("top_language"),
            "languages": t.get("languages"), "vad_speech_frac": t.get("vad_speech_frac"),
            "wav_duration": t.get("wav_duration"),
            "gzip_ratio_raw": t.get("gzip_ratio_raw"),
        })

    byc = {}
    for c in ("Hateful", "Offensive", "Normal"):
        byc[c] = [i["z"] for i in items if i["fine"] == c]

    res = {"corpus": "MHClip_EN", "split": "test_clean", "n": len(items),
           "class_counts": {c: len(byc[c]) for c in byc},
           "score_distribution_by_class": {c: quart(byc[c]) for c in byc},
           "stratified_auc": {}}

    pairs = [("hateful_vs_normal", "Hateful", "Normal"),
             ("offensive_vs_normal", "Offensive", "Normal"),
             ("hateful_vs_offensive", "Hateful", "Offensive")]
    for name, p, n in pairs:
        a, lo, hi = auc_ci(byc[p], byc[n])
        res["stratified_auc"][name] = {"auc": a, "ci95": [lo, hi],
                                       "n_pos": len(byc[p]), "n_neg": len(byc[n])}
    a, lo, hi = auc_ci(byc["Hateful"] + byc["Offensive"], byc["Normal"])
    res["stratified_auc"]["union_vs_normal_primary"] = {
        "auc": a, "ci95": [lo, hi],
        "n_pos": len(byc["Hateful"]) + len(byc["Offensive"]), "n_neg": len(byc["Normal"])}

    # Counterfactual: what would the primary AUC be if the Offensive class were
    # dropped from the positive side, or if it were moved to the negative side?
    res["counterfactual_auc"] = {
        "drop_offensive_hateful_only": res["stratified_auc"]["hateful_vs_normal"]["auc"],
        "offensive_as_negative": auc(byc["Hateful"], byc["Offensive"] + byc["Normal"]),
    }

    os.makedirs(os.path.join(OUT, "en"), exist_ok=True)
    with open(os.path.join(OUT, "en", "decomposition.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(OUT, "en", "items.json"), "w") as f:
        json.dump(items, f, indent=2)
    return res, items


# -------------------------------------------------------------- HateClipSeg --
def do_hcs():
    scores = read_jsonl(os.path.join(ROOT, "results/hateclipseg/judge_8b/scores.jsonl"))
    trans = read_jsonl(os.path.join(ROOT, "results/hateclipseg/fresh_transcripts.jsonl"))

    vids = {}
    with open(os.path.join(HCS_DATA, "video_level_annotation.csv")) as f:
        for row in csv.DictReader(f):
            vids[row["Video Id"].strip()] = {
                "labels": ast.literal_eval(row["Video-Level Label"]),
                "victims": ast.literal_eval(row["Target Victim"])}
    segs = {}
    with open(os.path.join(HCS_DATA, "segment_level_annotation.csv")) as f:
        for row in csv.DictReader(f):
            v = row["Video Id"].strip()
            segs[v] = {"labels": ast.literal_eval(row["Segment-Level Label"]),
                       "ts": ast.literal_eval(row["Segment Timestamp"])}

    items = []
    for v, s in scores.items():
        if v not in vids:
            continue
        labs = vids[v]["labels"]
        t = trans.get(v, {})
        it = {"video_id": v, "z": s["z"], "labels": labs,
              "victims": vids[v]["victims"],
              "union": 1 if any(x in labs for x in
                                ("hateful", "insulting", "sexual", "violence", "harm")) else 0,
              "strict": 1 if "hateful" in labs else 0,
              "n_transcript_chars": s.get("n_transcript_chars"),
              "transcript_source": s.get("transcript_source"),
              "fresh_chars": t.get("fresh_chars"), "top_language": t.get("top_language"),
              "vad_speech_frac": t.get("vad_speech_frac"),
              "wav_duration": t.get("wav_duration"),
              "gzip_ratio_raw": t.get("gzip_ratio_raw")}
        # segment-level density
        if v in segs:
            sl, ts = segs[v]["labels"], segs[v]["ts"]
            dur = [float(b) - float(a) for a, b in ts]
            total = sum(dur) or 1.0
            it["n_segments"] = len(sl)
            it["seg_total_duration"] = total
            for d in OFFENSIVE_DIMS + [0]:
                nm = IDX[d]
                hit = [k for k, x in enumerate(sl) if x[d]]
                it[f"segfrac_{nm}"] = sum(dur[k] for k in hit) / total
                it[f"segcount_{nm}"] = len(hit)
            un = [k for k, x in enumerate(sl) if any(x[d] for d in OFFENSIVE_DIMS)]
            it["segfrac_union"] = sum(dur[k] for k in un) / total
            it["segcount_union"] = len(un)
        items.append(it)

    res = {"corpus": "HateClipSeg", "n": len(items)}

    # enumerate label vocabulary
    from collections import Counter
    lc = Counter()
    for i in items:
        for l in i["labels"]:
            lc[l] += 1
        if not i["labels"]:
            lc["<empty>"] += 1
    res["video_level_label_counts"] = dict(lc.most_common())
    vc = Counter()
    for i in items:
        for v_ in i["victims"]:
            vc[v_] += 1
    res["target_victim_counts"] = dict(vc.most_common())

    z = {i["video_id"]: i["z"] for i in items}
    neg_union = [i["z"] for i in items if i["union"] == 0]
    neg_strict = [i["z"] for i in items if i["strict"] == 0]

    res["collapse_auc"] = {}
    for nm, key in (("offensive_union", "union"), ("hateful_strict", "strict")):
        pos = [i["z"] for i in items if i[key] == 1]
        neg = [i["z"] for i in items if i[key] == 0]
        a, lo, hi = auc_ci(pos, neg)
        res["collapse_auc"][nm] = {"auc": a, "ci95": [lo, hi],
                                   "n_pos": len(pos), "n_neg": len(neg),
                                   "prevalence": len(pos) / len(items)}

    # per-dimension AUC against the clean-normal set (no offensive label at all)
    clean_normal = [i["z"] for i in items if i["union"] == 0]
    res["per_label_vs_clean_normal"] = {}
    res["score_distribution_by_label"] = {}
    for d in OFFENSIVE_DIMS:
        nm = IDX[d]
        pos = [i["z"] for i in items if nm in i["labels"]]
        a, lo, hi = auc_ci(pos, clean_normal)
        res["per_label_vs_clean_normal"][nm] = {"auc": a, "ci95": [lo, hi],
                                                "n_pos": len(pos), "n_neg": len(clean_normal)}
        res["score_distribution_by_label"][nm] = quart(pos)
    res["score_distribution_by_label"]["clean_normal"] = quart(clean_normal)

    # exclusive strata: exactly which single label
    excl = {}
    for d in OFFENSIVE_DIMS:
        nm = IDX[d]
        pos = [i["z"] for i in items if i["labels"] == [nm]]
        excl[nm] = {"n": len(pos), "dist": quart(pos)}
        if len(pos) >= 5:
            a, lo, hi = auc_ci(pos, clean_normal)
            excl[nm]["auc_vs_clean_normal"] = {"auc": a, "ci95": [lo, hi]}
    res["exclusive_label_strata"] = excl

    # hateful-vs-offensive-only (the class-pair analogue of MHClip's H-vs-O)
    hate = [i["z"] for i in items if i["strict"] == 1]
    offonly = [i["z"] for i in items if i["union"] == 1 and i["strict"] == 0]
    a, lo, hi = auc_ci(hate, offonly)
    res["hateful_vs_offensive_only"] = {"auc": a, "ci95": [lo, hi],
                                        "n_pos": len(hate), "n_neg": len(offonly)}
    res["score_distribution_by_label"]["offensive_only"] = quart(offonly)
    res["score_distribution_by_label"]["hateful_any"] = quart(hate)

    # segment-density angle: are low-z hateful videos the sparse-hate ones?
    hs = [i for i in items if i["strict"] == 1 and "segfrac_hateful" in i]
    if hs:
        hs_sorted = sorted(hs, key=lambda i: i["z"])
        k = max(1, len(hs_sorted) // 4)
        res["segment_density_hateful_strict"] = {
            "n_with_segments": len(hs),
            "overall_segfrac_hateful": quart([i["segfrac_hateful"] for i in hs]),
            "lowest_z_quartile_segfrac": quart([i["segfrac_hateful"] for i in hs_sorted[:k]]),
            "highest_z_quartile_segfrac": quart([i["segfrac_hateful"] for i in hs_sorted[-k:]]),
            "spearman_z_vs_segfrac": spearman([i["z"] for i in hs],
                                              [i["segfrac_hateful"] for i in hs]),
            "spearman_z_vs_duration": spearman([i["z"] for i in hs],
                                               [i["seg_total_duration"] for i in hs]),
            "spearman_z_vs_hateful_seconds": spearman(
                [i["z"] for i in hs],
                [i["segfrac_hateful"] * i["seg_total_duration"] for i in hs]),
        }
    hu = [i for i in items if i["union"] == 1 and "segfrac_union" in i]
    if hu:
        res["segment_density_union"] = {
            "n_with_segments": len(hu),
            "overall_segfrac_union": quart([i["segfrac_union"] for i in hu]),
            "spearman_z_vs_segfrac": spearman([i["z"] for i in hu],
                                              [i["segfrac_union"] for i in hu]),
        }

    os.makedirs(os.path.join(OUT, "hcs"), exist_ok=True)
    with open(os.path.join(OUT, "hcs", "decomposition.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(OUT, "hcs", "items.json"), "w") as f:
        json.dump(items, f, indent=2)
    return res, items


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


if __name__ == "__main__":
    en, _ = do_en()
    print(json.dumps(en, indent=2))
    print("=" * 70)
    hcs, _ = do_hcs()
    print(json.dumps(hcs, indent=2))
