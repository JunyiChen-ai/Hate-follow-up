"""Ranking-error autopsy for MHClip-ZH, stage 3: aggregate the manual codings and
screen the label-free covariates that the reading suggested.

Reads results/ranking_autopsy/zh/coding_zh.tsv, which was written by hand after
reading every transcript and four frames of every video in the packet. Produces
results/ranking_autopsy/zh/autopsy_summary.json, from which the committed note
is written. Only aggregate counts and rates reach the note.

The two corpus-wide covariates measured here were both suggested by the reading:

  boilerplate  the recogniser returned a stock closing phrase, a subtitling
               credit or a channel plug instead of the video's speech. Measured
               with a fixed phrase list applied to the text the judge received.
  title cue    the shipped title carries the dataset harvester's highlighted
               query term, which is an offensive word. Available without labels.
"""

import csv
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(ROOT, "results", "ranking_autopsy", "zh")

CODE_MEANING = {
    "G": "gender-stereotype commentary or skit, no slur",
    "T": "title-borne: the offence is only in the uploader's title",
    "F": "fiction or filmed conflict; the offence is a plot or scene event",
    "S": "sexual content or covert footage as the whole basis of the label",
    "X": "the video is not in Chinese",
    "I": "implicit protected-group hostility carried by a meme juxtaposition",
    "M": "clinical sexual-health advertorial, anatomical vocabulary",
    "A": "profanity or insult with no group target, often only in the title",
    "Q": "counter-speech: reproduces prejudice in order to criticise it",
    "P": "protected group present as topic, not attacked",
    "R": "news or report about harm",
    "D": "content the judge is arguably right about, annotated Normal",
    "O": "other benign, mild suggestion or a folkloric nickname",
}

# Fixed phrase list. Every entry is a recogniser artefact, not video content:
# stock closing lines, subtitling-group credits and one channel's donation plug
# that Whisper emits verbatim on Chinese material with little or no speech.
BOILERPLATE = [
    "thanks for watching", "thank you for watching", "see you next time",
    "terima kasih telah menonton", "sub indo", "amara.org", "субтитр",
    "редактор субтитров", "字幕by", "字幕志愿者", "请不吝点赞", "打赏支持明镜",
    "订阅 转发", "dramatisk musikk", "по субтитрам",
]


def wilson(k, n):
    if n == 0:
        return [None, None]
    p, z = k / n, 1.959963985
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return [round((c - h) / d, 3), round((c + h) / d, 3)]


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
    return round(float((r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2)
                       / (len(pos) * len(neg))), 4)


def auc_ci(pos, neg, n_boot=2000, seed=20260809):
    a = auc(pos, neg)
    if a is None:
        return None, None
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    bs = [auc(rng.choice(pos, len(pos), replace=True),
              rng.choice(neg, len(neg), replace=True)) for _ in range(n_boot)]
    return a, [round(float(x), 4) for x in np.percentile(bs, [2.5, 97.5])]


def med(v):
    return round(float(np.median(v)), 3) if len(v) else None


def is_boilerplate(text):
    t = (text or "").lower()
    return any(p in t for p in BOILERPLATE)


KEYWORD = re.compile(r'<em class="keyword">(.*?)</em>')


def main():
    items = json.load(open(os.path.join(OUT, "items.json")))
    packet = json.load(open(os.path.join(OUT, "packet.json")))
    rows = list(csv.DictReader(open(os.path.join(OUT, "coding_zh.tsv")), delimiter="\t"))

    for i in items:
        judged = i["fresh_text"] if i["transcript_source"] == "override" \
            else i["dataset_transcript"]
        i["judged_text"] = judged
        i["boilerplate"] = is_boilerplate(judged)
        i["kw"] = KEYWORD.findall(i["title"])
        i["nonchinese_asr"] = i["script_fresh"]["dominant_script"] not in ("cjk", "none")

    by_vid = {i["video_id"]: i for i in items}
    by_alias = {r["alias"]: by_vid[r["video_id"]] for r in packet}

    res = {"corpus": "MHClip_ZH", "n": len(items), "code_meanings": CODE_MEANING}

    # ------------------------------------------------------- taxonomy counts --
    for side, key in (("FN", "false_negative_taxonomy"), ("FP", "false_positive_taxonomy")):
        sel = [r for r in rows if r["side"] == side]
        c = Counter(r["code"] for r in sel)
        tab = []
        for code, n in c.most_common():
            zs = [by_alias[r["alias"]]["z"] for r in sel if r["code"] == code]
            tab.append({"code": code, "meaning": CODE_MEANING[code], "n": n,
                        "rate": round(n / len(sel), 3), "wilson95": wilson(n, len(sel)),
                        "median_z": med(zs)})
        res[key] = {"cohort_size": len(sel), "codes": tab}

    # -------------------------------------------------- construct measurement --
    prot = {r["alias"]: r["protected"] for r in rows if r["alias"].startswith("ZH-POS-")}
    strata = defaultdict(list)
    for a, p in prot.items():
        strata[p].append(by_alias[a])
    negz = [i["z"] for i in items if i["binary"] == 0]
    tab = []
    for name in ("explicit", "implicit", "none"):
        zs = [r["z"] for r in strata[name]]
        a, ci = auc_ci(zs, negz)
        tab.append({"stratum": name, "n": len(zs), "median_z": med(zs),
                    "auc_vs_normal": a, "ci95": ci,
                    "fine_labels": dict(Counter(r["fine"] for r in strata[name]))})
    allpos = [i["z"] for i in items if i["binary"] == 1]
    a, ci = auc_ci(allpos, negz)
    tab.append({"stratum": "all positives", "n": len(allpos), "median_z": med(allpos),
                "auc_vs_normal": a, "ci95": ci})
    res["protected_target_strata"] = tab
    res["protected_target_note"] = (
        "coded by the same reader who saw the scores, so this is score-aware and "
        "descriptive; it is not a blind audit")

    # does the Hateful-only stratum separate, as MHClip-EN's explicit stratum did
    hz = [i["z"] for i in items if i["fine"] == "Hateful"]
    hz_expl = [by_alias[a]["z"] for a, p in prot.items()
               if p == "explicit" and by_alias[a]["fine"] == "Hateful"]
    a1, c1 = auc_ci(hz, negz)
    a2, c2 = auc_ci(hz_expl, negz) if hz_expl else (None, None)
    res["hateful_stratum"] = {
        "hateful_vs_normal": {"n": len(hz), "auc": a1, "ci95": c1},
        "hateful_and_explicit_vs_normal": {"n": len(hz_expl), "auc": a2, "ci95": c2},
        "explicit_any_label_vs_normal": tab[0]}

    # -------------------------------------------------------- covariate screen --
    bp = [i for i in items if i["boilerplate"]]
    nbp = [i for i in items if not i["boilerplate"]]
    res["covariate_boilerplate_transcript"] = {
        "definition": "the text the judge read matches a fixed list of recogniser "
                      "artefacts: stock closing lines, subtitling credits, a channel plug",
        "n": len(bp), "rate": round(len(bp) / len(items), 4),
        "by_class": {c: sum(1 for i in bp if i["fine"] == c)
                     for c in ("Hateful", "Offensive", "Normal")},
        "class_denominators": {c: sum(1 for i in items if i["fine"] == c)
                               for c in ("Hateful", "Offensive", "Normal")},
        "median_vad_speech_frac": {"boilerplate": med([i["vad_speech_frac"] for i in bp]),
                                   "clean": med([i["vad_speech_frac"] for i in nbp])},
        "median_judged_chars": {"boilerplate": med([i["n_transcript_chars"] for i in bp]),
                                "clean": med([i["n_transcript_chars"] for i in nbp])},
        "gate_outcome": dict(Counter(i["gate_outcome"] for i in bp)),
        "auc_on_boilerplate_subset": auc([i["z"] for i in bp if i["binary"] == 1],
                                         [i["z"] for i in bp if i["binary"] == 0]),
        "auc_on_clean_subset": auc([i["z"] for i in nbp if i["binary"] == 1],
                                   [i["z"] for i in nbp if i["binary"] == 0]),
        "n_pos_boilerplate": sum(i["binary"] for i in bp),
        "n_pos_clean": sum(i["binary"] for i in nbp),
        "rate_in_fn_cohort": round(sum(1 for r in rows if r["side"] == "FN"
                                       and by_alias[r["alias"]]["boilerplate"]) / 20, 3),
        "rate_in_fp_cohort": round(sum(1 for r in rows if r["side"] == "FP"
                                       and by_alias[r["alias"]]["boilerplate"]) / 20, 3),
    }

    nz = [i for i in items if i["nonchinese_asr"]]
    res["covariate_non_chinese_asr"] = {
        "definition": "the recogniser's dominant script for this video is not Han",
        "n": len(nz), "rate": round(len(nz) / len(items), 4),
        "by_class": {c: sum(1 for i in nz if i["fine"] == c)
                     for c in ("Hateful", "Offensive", "Normal")},
        "median_vad": med([i["vad_speech_frac"] for i in nz]),
        "overlap_with_boilerplate": sum(1 for i in nz if i["boilerplate"]),
    }

    kw = [i for i in items if i["kw"]]
    nokw = [i for i in items if not i["kw"]]
    res["covariate_harvest_keyword"] = {
        "definition": "the shipped title carries the collection query term marked up "
                      "by the harvester; the terms are offensive words",
        "n_with_markup": len(kw), "rate": round(len(kw) / len(items), 4),
        "by_class": {c: sum(1 for i in kw if i["fine"] == c)
                     for c in ("Hateful", "Offensive", "Normal")},
        "median_z_normal_with_keyword": med([i["z"] for i in kw if i["binary"] == 0]),
        "median_z_normal_without": med([i["z"] for i in nokw if i["binary"] == 0]),
        "median_z_positive_with_keyword": med([i["z"] for i in kw if i["binary"] == 1]),
        "median_z_positive_without": med([i["z"] for i in nokw if i["binary"] == 1]),
        "auc_within_keyword_subset": auc([i["z"] for i in kw if i["binary"] == 1],
                                         [i["z"] for i in kw if i["binary"] == 0]),
        "auc_within_no_keyword_subset": auc([i["z"] for i in nokw if i["binary"] == 1],
                                            [i["z"] for i in nokw if i["binary"] == 0]),
        "n_pos_kw": sum(i["binary"] for i in kw), "n_neg_kw": len(kw) - sum(i["binary"] for i in kw),
        "n_pos_nokw": sum(i["binary"] for i in nokw),
        "n_neg_nokw": len(nokw) - sum(i["binary"] for i in nokw),
        "keyword_presence_as_standalone_detector_auc":
            auc([1 if i["kw"] else 0 for i in items if i["binary"] == 1],
                [1 if i["kw"] else 0 for i in items if i["binary"] == 0]),
        "share_of_top_scoring_normals_with_keyword": round(
            sum(1 for i in items if i["binary"] == 0 and i["z"] >= 5.0 and i["kw"])
            / max(1, sum(1 for i in items if i["binary"] == 0 and i["z"] >= 5.0)), 3),
    }

    # per-code covariate table for codes with at least four videos
    percode = []
    for side in ("FN", "FP"):
        sel = [r for r in rows if r["side"] == side]
        for code, n in Counter(r["code"] for r in sel).items():
            if n < 4:
                continue
            g = [by_alias[r["alias"]] for r in sel if r["code"] == code]
            gg = g
            percode.append({
                "side": side, "code": code, "n": n, "median_z": med([r["z"] for r in g]),
                "median_vad": med([r["vad_speech_frac"] for r in g]),
                "median_judged_chars": med([r["n_transcript_chars"] for r in g]),
                "median_duration_s": med([r["wav_duration"] for r in g]),
                "n_boilerplate": sum(1 for i in gg if i["boilerplate"]),
                "n_harvest_keyword": sum(1 for i in gg if i["kw"]),
            })
    res["per_code_covariates"] = percode
    res["cohort_reference_medians"] = {
        "corpus_median_vad": med([i["vad_speech_frac"] for i in items]),
        "corpus_median_judged_chars": med([i["n_transcript_chars"] for i in items]),
        "corpus_median_duration_s": med([i["wav_duration"] for i in items]),
    }

    # duplicates by judged text
    dd = defaultdict(list)
    for i in items:
        t = (i["judged_text"] or "").strip()
        if len(t) > 15:
            dd[t].append(i)
    dups = [{"n": len(v), "labels": [x["fine"] for x in v], "z": [x["z"] for x in v]}
            for v in dd.values() if len(v) > 1]
    res["exact_transcript_duplicate_groups"] = {
        "n_groups": len(dups), "groups": dups,
        "n_groups_disagreeing_on_collapse": sum(
            1 for d in dups if len(set(1 if l in ("Hateful", "Offensive") else 0
                                       for l in d["labels"])) > 1)}

    with open(os.path.join(OUT, "autopsy_summary.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
