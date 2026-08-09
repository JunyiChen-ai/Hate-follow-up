"""Ranking-error autopsy for MHClip-ZH, stage 1: decomposition and covariates.

Diagnostic only. No preregistration, no model call, no method. Reads the frozen
single-call Qwen3-VL-8B judge scores for MHClip_ZH test_clean and decomposes the
corpus ROC-AUC into class-pair strata, localises the oracle threshold's residual
errors, and screens label-free covariates.

Writes results/ranking_autopsy/zh/{decomposition.json,items.json}. Those files
carry video ids and transcript statistics and stay gitignored.
"""

import json
import os
import re
import sys
import unicodedata

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "our_method"))
os.environ.setdefault("HVD_DATA_ROOT", "/home/jehc223/data")

from data_utils import load_annotations, load_clean_split_ids  # noqa: E402

OUT = os.path.join(ROOT, "results", "ranking_autopsy", "zh")
RUN = os.path.join(ROOT, "results", "testruns", "mhclip_zh")


# ------------------------------------------------------------------ helpers --
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
    rp = r[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def auc_ci(pos, neg, n_boot=2000, seed=20260809):
    a = auc(pos, neg)
    if a is None:
        return None, None, None
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    bs = [auc(rng.choice(pos, len(pos), replace=True),
              rng.choice(neg, len(neg), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return a, float(lo), float(hi)


def quart(v):
    v = np.asarray(v, float)
    if len(v) == 0:
        return {"n": 0}
    return {"n": int(len(v)), "min": float(v.min()), "q1": float(np.percentile(v, 25)),
            "median": float(np.median(v)), "q3": float(np.percentile(v, 75)),
            "max": float(v.max()), "mean": round(float(v.mean()), 4)}


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return round(float(np.corrcoef(ra, rb)[0, 1]), 4)


def wilson(k, n):
    if n == 0:
        return [None, None]
    p, z = k / n, 1.959963985
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return [round((c - h) / d, 4), round((c + h) / d, 4)]


def read_jsonl(path, key="video_id"):
    d = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                o = json.loads(line)
                d[o[key]] = o
    return d


def confusion(items, thr):
    tp = sum(1 for i in items if i["binary"] == 1 and i["z"] >= thr)
    fp = sum(1 for i in items if i["binary"] == 0 and i["z"] >= thr)
    fn = sum(1 for i in items if i["binary"] == 1 and i["z"] < thr)
    tn = sum(1 for i in items if i["binary"] == 0 and i["z"] < thr)
    f1p = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    f1n = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) else 0.0
    return {"threshold": thr, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "f1_hateful": round(f1p, 6), "f1_normal": round(f1n, 6),
            "macro_f1": round((f1p + f1n) / 2, 6),
            "accuracy": round((tp + tn) / len(items), 6)}


# ------------------------------------------------------------- script-family --
_CJK = re.compile(r"[一-鿿㐀-䶿]")
_LATIN = re.compile(r"[A-Za-z]")
_HANGUL = re.compile(r"[가-힯]")
_KANA = re.compile(r"[぀-ヿ]")
_CYR = re.compile(r"[Ѐ-ӿ]")


def script_profile(text):
    text = text or ""
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    n = len(letters)
    if n == 0:
        return {"n_letters": 0, "frac_cjk": None, "frac_latin": None,
                "dominant_script": "none"}
    fr = {
        "cjk": len(_CJK.findall(text)) / n,
        "latin": len(_LATIN.findall(text)) / n,
        "hangul": len(_HANGUL.findall(text)) / n,
        "kana": len(_KANA.findall(text)) / n,
        "cyrillic": len(_CYR.findall(text)) / n,
    }
    dom = max(fr, key=fr.get)
    return {"n_letters": n, "frac_cjk": round(fr["cjk"], 4),
            "frac_latin": round(fr["latin"], 4),
            "frac_hangul": round(fr["hangul"], 4),
            "frac_kana": round(fr["kana"], 4),
            "dominant_script": dom if fr[dom] > 0.5 else "mixed"}


# ------------------------------------------------------------------- loading --
def build_items():
    scores = read_jsonl(os.path.join(RUN, "judge_8b", "scores.jsonl"))
    trans = read_jsonl(os.path.join(RUN, "fresh_transcripts.jsonl"))
    audio = read_jsonl(os.path.join(RUN, "audio_meta.jsonl"))
    gate = read_jsonl(os.path.join(RUN, "gate_outcomes.jsonl"))
    ann = load_annotations("MHClip_ZH")
    ids = load_clean_split_ids("MHClip_ZH", "test")

    items, seen = [], set()
    for v in ids:
        if v in seen or v not in scores:
            continue
        seen.add(v)
        s, t, a, g = scores[v], trans.get(v, {}), audio.get(v, {}), gate.get(v, {})
        lab = ann[v]["label"]
        fresh = t.get("fresh_text", "") or ""
        ds_tx = ann[v].get("transcript", "") or ""
        items.append({
            "video_id": v, "z": s["z"], "fine": lab,
            "binary": 1 if lab in ("Hateful", "Offensive") else 0,
            "title": ann[v].get("title", ""),
            "n_transcript_chars": s.get("n_transcript_chars"),
            "transcript_source": s.get("transcript_source"),
            "fresh_chars": t.get("fresh_chars"), "old_chars": t.get("old_chars"),
            "top_language": t.get("top_language"), "languages": t.get("languages"),
            "vad_speech_frac": t.get("vad_speech_frac", a.get("vad_speech_frac")),
            "wav_duration": t.get("wav_duration", a.get("wav_duration")),
            "container_duration": a.get("container_duration"),
            "has_audio": a.get("has_audio"),
            "gzip_ratio_raw": t.get("gzip_ratio_raw"),
            "edit_norm_vs_dataset": t.get("edit_norm_vs_dataset"),
            "gate_outcome": g.get("outcome"),
            "fresh_text": fresh, "dataset_transcript": ds_tx,
            "script_fresh": script_profile(fresh),
            "script_dataset": script_profile(ds_tx),
            "dataset_has_music_glyph": ("\U0001f3bc" in ds_tx) or ("♪" in ds_tx)
                                       or ("♫" in ds_tx),
        })
    return items


# --------------------------------------------------------------- main report --
def main():
    items = build_items()
    os.makedirs(OUT, exist_ok=True)
    byc = {c: [i["z"] for i in items if i["fine"] == c]
           for c in ("Hateful", "Offensive", "Normal")}
    pos = [i["z"] for i in items if i["binary"] == 1]
    neg = [i["z"] for i in items if i["binary"] == 0]

    res = {"corpus": "MHClip_ZH", "split": "test_clean", "n": len(items),
           "class_counts": {c: len(byc[c]) for c in byc},
           "score_distribution_by_class": {c: quart(byc[c]) for c in byc},
           "score_distribution_union": {"positive": quart(pos), "negative": quart(neg)},
           "stratified_auc": {}}

    for name, p, n in (("hateful_vs_normal", "Hateful", "Normal"),
                       ("offensive_vs_normal", "Offensive", "Normal"),
                       ("hateful_vs_offensive", "Hateful", "Offensive")):
        a, lo, hi = auc_ci(byc[p], byc[n])
        res["stratified_auc"][name] = {"auc": round(a, 4), "ci95": [round(lo, 4), round(hi, 4)],
                                       "n_pos": len(byc[p]), "n_neg": len(byc[n])}
    a, lo, hi = auc_ci(pos, neg)
    res["stratified_auc"]["union_vs_normal_primary"] = {
        "auc": round(a, 4), "ci95": [round(lo, 4), round(hi, 4)],
        "n_pos": len(pos), "n_neg": len(neg)}

    # rank-weighted decomposition check
    nh, no = len(byc["Hateful"]), len(byc["Offensive"])
    res["union_decomposition_check"] = {
        "weighted_average":
            round((nh * res["stratified_auc"]["hateful_vs_normal"]["auc"]
                   + no * res["stratified_auc"]["offensive_vs_normal"]["auc"]) / (nh + no), 4),
        "observed_union": res["stratified_auc"]["union_vs_normal_primary"]["auc"]}

    # ------------------------------------------------------------ threshold --
    grid = sorted({i["z"] for i in items} | {min(i["z"] for i in items) - 1})
    sweep = [confusion(items, t) for t in grid]
    best = max(sweep, key=lambda c: c["macro_f1"])
    res["oracle"] = best
    res["oracle_sweep_top5"] = sorted(sweep, key=lambda c: -c["macro_f1"])[:5]
    res["method_valley"] = confusion(items, 0.6068749999999987)
    res["diagnostic_zero"] = confusion(items, 0.0)

    # residual error localisation at the oracle threshold
    thr = best["threshold"]
    fp_items = [i for i in items if i["binary"] == 0 and i["z"] >= thr]
    fn_items = [i for i in items if i["binary"] == 1 and i["z"] < thr]
    res["oracle_residual"] = {
        "n_fp": len(fp_items), "n_fn": len(fn_items),
        "fp_z": quart([i["z"] for i in fp_items]),
        "fn_z": quart([i["z"] for i in fn_items]),
        "fn_by_class": {c: sum(1 for i in fn_items if i["fine"] == c)
                        for c in ("Hateful", "Offensive")},
        "fp_share_of_errors": round(len(fp_items) / (len(fp_items) + len(fn_items)), 4),
    }
    # counterfactual: perfect FP removal vs perfect FN removal
    tp, fp, fn, tn = best["tp"], best["fp"], best["fn"], best["tn"]

    def mf1(tp, fp, fn, tn):
        f1p = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        f1n = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) else 0.0
        return round((f1p + f1n) / 2, 4)

    res["oracle_error_budget"] = {
        "as_is": mf1(tp, fp, fn, tn),
        "if_all_fp_fixed": mf1(tp, 0, fn, tn + fp),
        "if_all_fn_fixed": mf1(tp + fn, fp, 0, tn),
        "if_half_fp_fixed": mf1(tp, fp // 2, fn, tn + fp - fp // 2),
        "note": "hypothetical relabelling of the errors, to show which side owns the gap",
    }

    # class composition inside score bands
    bands = [(-99, -5), (-5, 0), (0, 5), (5, 9), (9, 99)]
    res["band_composition"] = []
    for lo_, hi_ in bands:
        sel = [i for i in items if lo_ <= i["z"] < hi_]
        res["band_composition"].append({
            "band": f"[{lo_}, {hi_})", "n": len(sel),
            "Hateful": sum(1 for i in sel if i["fine"] == "Hateful"),
            "Offensive": sum(1 for i in sel if i["fine"] == "Offensive"),
            "Normal": sum(1 for i in sel if i["fine"] == "Normal"),
            "positive_rate": round(sum(i["binary"] for i in sel) / len(sel), 4) if sel else None})

    # --------------------------------------------------------- covariates ----
    covs = ["fresh_chars", "n_transcript_chars", "vad_speech_frac", "wav_duration",
            "gzip_ratio_raw", "edit_norm_vs_dataset", "old_chars"]
    res["covariate_by_class"] = {}
    for c in ("Hateful", "Offensive", "Normal"):
        sel = [i for i in items if i["fine"] == c]
        res["covariate_by_class"][c] = {
            k: quart([i[k] for i in sel if i.get(k) is not None]) for k in covs}
    res["covariate_spearman_vs_z"] = {
        k: spearman([i["z"] for i in items if i.get(k) is not None],
                    [i[k] for i in items if i.get(k) is not None]) for k in covs}
    # AUC of each covariate as a standalone detector
    res["covariate_standalone_auc"] = {
        k: (lambda p, n: round(auc(p, n), 4) if p and n else None)(
            [i[k] for i in items if i["binary"] == 1 and i.get(k) is not None],
            [i[k] for i in items if i["binary"] == 0 and i.get(k) is not None])
        for k in covs}

    # script / language
    from collections import Counter
    res["script_profile"] = {
        "fresh_dominant_script": dict(Counter(i["script_fresh"]["dominant_script"]
                                              for i in items).most_common()),
        "fresh_dominant_script_by_class": {
            c: dict(Counter(i["script_fresh"]["dominant_script"]
                            for i in items if i["fine"] == c).most_common())
            for c in ("Hateful", "Offensive", "Normal")},
        "n_top_language_null": sum(1 for i in items if i["top_language"] is None),
        "median_frac_cjk_fresh": round(float(np.median(
            [i["script_fresh"]["frac_cjk"] for i in items
             if i["script_fresh"]["frac_cjk"] is not None])), 4),
        "n_fresh_empty": sum(1 for i in items if not i["fresh_text"].strip()),
        "n_fresh_under_20_chars": sum(1 for i in items if (i["fresh_chars"] or 0) < 20),
        "n_dataset_music_glyph": sum(1 for i in items if i["dataset_has_music_glyph"]),
        "dataset_music_glyph_by_class": {
            c: sum(1 for i in items if i["fine"] == c and i["dataset_has_music_glyph"])
            for c in ("Hateful", "Offensive", "Normal")},
    }
    # music-marked subset AUC
    mus = [i for i in items if i["dataset_has_music_glyph"]]
    nomus = [i for i in items if not i["dataset_has_music_glyph"]]
    for nm, sub in (("music_glyph", mus), ("no_music_glyph", nomus)):
        p = [i["z"] for i in sub if i["binary"] == 1]
        n = [i["z"] for i in sub if i["binary"] == 0]
        res["script_profile"][f"auc_{nm}"] = {
            "n_pos": len(p), "n_neg": len(n),
            "auc": round(auc(p, n), 4) if p and n else None}

    # speech-poor vs speech-rich split (the restoration-class question)
    med = float(np.median([i["fresh_chars"] or 0 for i in items]))
    res["speech_split"] = {"median_fresh_chars": med}
    for nm, sel in (("speech_poor", [i for i in items if (i["fresh_chars"] or 0) <= med]),
                    ("speech_rich", [i for i in items if (i["fresh_chars"] or 0) > med])):
        p = [i["z"] for i in sel if i["binary"] == 1]
        n = [i["z"] for i in sel if i["binary"] == 0]
        a2, lo2, hi2 = auc_ci(p, n) if p and n else (None, None, None)
        res["speech_split"][nm] = {
            "n": len(sel), "n_pos": len(p), "n_neg": len(n),
            "auc": round(a2, 4) if a2 else None,
            "ci95": [round(lo2, 4), round(hi2, 4)] if a2 else None,
            "median_z_pos": round(float(np.median(p)), 3) if p else None,
            "median_z_neg": round(float(np.median(n)), 3) if n else None}

    res["gate_outcomes"] = dict(Counter(i["gate_outcome"] for i in items).most_common())
    res["transcript_source"] = dict(Counter(i["transcript_source"] for i in items).most_common())

    with open(os.path.join(OUT, "decomposition.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT, "items.json"), "w") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("oracle_sweep_top5",)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
