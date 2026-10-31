#!/usr/bin/env python3
"""Post-hoc diagnostics of the plan runs (CPU). This is the ONLY place in the experiment that reads labels: the
video-level label (any positive 4 fps GT frame; HateMM also carries it in the id) is used to split the plan and
read-change statistics by hateful / non-hateful video. Nothing here feeds any score.
Reads: runs/20261101_video_plan/<corpus>/records/<ds>/<vid>.json (own plan + reads), records_other (other plan),
the stored adopted reads (base). Writes runs/20261101_video_plan/diagnostics.json and prints a summary.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "runs/20261101_video_plan"
ADOPTED = ROOT / "runs/20261014_m1_audio_bound/vl_aligned/vl_text_aligned/predictions.jsonl"
DATASETS = ["HateMM", "HateClipSeg"]
TIME_RE = re.compile(r"(\b\d+(?:\.\d+)?\s?s(?:ec(?:ond)?s?)?\b|\b\d{1,2}:\d{2}\b|\bt\s?=\s?\d|\btimestamp|\bat\s\d+(?:\.\d+)?\b|\b\d+(?:\.\d+)?\s?(?:to|-|–)\s?\d+(?:\.\d+)?\s?s\b)", re.I)
# crude keyword list for "names a target group" (reported with the number; not a label of anything)
GROUP_RE = re.compile(r"\b(black|blacks|african|jew|jews|jewish|muslim|muslims|islam|arab|arabs|asian|asians|chinese|indian|indians|mexican|mexicans|"
                      r"hispanic|latino|latinos|immigrant|immigrants|migrant|migrants|refugee|refugees|white people|whites|women|woman|female|females|"
                      r"men\b|gay|gays|lesbian|lgbt|lgbtq|trans|transgender|homosexual|queer|christian|christians|catholic|hindu|hindus|sikh|"
                      r"disabled|disability|autis|jewry|zionist|n-word|nigg|fag|tranny|kike|chink|spic|gook|wetback|retard)", re.I)
NONE_RE = re.compile(r"\b(nothing in (this|the) video|does not (violate|contain)|no (content|evidence|violation|hateful)|not (hateful|violate)|"
                     r"doesn't (violate|contain)|none of the rules|no rule)", re.I)


def labels(ds):
    g = np.load(ROOT / f"data/gt_4fps/{ds}.npz", allow_pickle=True)
    return {str(v): bool(np.asarray(y).max() > 0) if len(np.asarray(y)) else None for v, y in zip(g["video_ids"], g["y4"])}


def main():
    base = {(r["dataset"], r["video_id"]): r for r in map(json.loads, open(ADOPTED)) if not r.get("error")}
    out = {}
    examples = {}
    for ds in DATASETS:
        lab = labels(ds)
        own = {p.stem: json.loads(p.read_text()) for p in sorted((R / ds / "records" / ds).glob("*.json"))}
        oth = {p.stem: json.loads(p.read_text()) for p in sorted((R / ds / "records_other" / ds).glob("*.json"))}
        if not own:
            print(ds, "no records"); continue
        toks = np.array([r["plan"]["tokens"] for r in own.values()])
        words = np.array([len(r["plan"]["text"].split()) for r in own.values()])
        hit = np.array([r["plan"]["hit_max_tokens"] for r in own.values()])
        times = np.array([bool(TIME_RE.search(r["plan"]["text"])) for r in own.values()])
        group = {v: bool(GROUP_RE.search(r["plan"]["text"])) for v, r in own.items()}
        none = {v: bool(NONE_RE.search(r["plan"]["text"])) for v, r in own.items()}
        y = {v: lab.get(v) for v in own}
        hateful = [v for v in own if y[v] is True]
        nonhate = [v for v in own if y[v] is False]
        d = dict(videos=len(own), videos_other=len(oth), plan_tokens_mean=float(toks.mean()), plan_tokens_median=float(np.median(toks)),
                 plan_words_mean=float(words.mean()), hit_max_tokens_share=float(hit.mean()), mentions_time_share=float(times.mean()),
                 names_group_share_all=float(np.mean([group[v] for v in own])),
                 names_group_share_nonhateful=float(np.mean([group[v] for v in nonhate])) if nonhate else None,
                 names_group_share_hateful=float(np.mean([group[v] for v in hateful])) if hateful else None,
                 says_nothing_share_nonhateful=float(np.mean([none[v] for v in nonhate])) if nonhate else None,
                 says_nothing_share_hateful=float(np.mean([none[v] for v in hateful])) if hateful else None,
                 stance_yes_share=float(np.mean([r["stance"] == "Yes" for r in own.values()])), n_hateful=len(hateful), n_nonhateful=len(nonhate))
        # per-video mean change of the window reads, own - base and other - base, by video label
        def deltas(recs, key):
            dv, dsp, up_v, up_s = {}, {}, {}, {}
            for v, r in recs.items():
                b = base[(ds, v)]["extra"]["windows"]
                t = r["traces"]
                assert len(b) == len(t)
                x = [tt["z_visual"] - bb["z_visual"] for bb, tt in zip(b, t)]
                s = [tt["z_speech"] - bb["z_speech"] for bb, tt in zip(b, t) if "z_speech" in bb and tt["z_speech"] is not None]
                dv[v] = float(np.mean(x)); up_v[v] = float(np.mean([e > 0 for e in x]))
                if s:
                    dsp[v] = float(np.mean(s)); up_s[v] = float(np.mean([e > 0 for e in s]))
            res = {}
            for name, ids in (("hateful", hateful), ("nonhateful", nonhate)):
                res[name] = dict(visual_mean_delta=float(np.mean([dv[v] for v in ids if v in dv])) if ids else None,
                                 visual_share_up=float(np.mean([up_v[v] for v in ids if v in up_v])) if ids else None,
                                 speech_mean_delta=float(np.mean([dsp[v] for v in ids if v in dsp])) if any(v in dsp for v in ids) else None,
                                 speech_share_up=float(np.mean([up_s[v] for v in ids if v in up_s])) if any(v in up_s for v in ids) else None)
            return res
        d["own_minus_base"] = deltas(own, "own")
        if oth:
            d["other_minus_base"] = deltas(oth, "other")
        # rank agreement of the own and base window reads, within video (Spearman, mean over videos with >= 3 windows)
        def spearman(recs):
            from scipy.stats import spearmanr
            vals = []
            for v, r in recs.items():
                b = [w["z_visual"] for w in base[(ds, v)]["extra"]["windows"]]
                t = [tt["z_visual"] for tt in r["traces"]]
                if len(b) >= 3 and np.std(b) > 0 and np.std(t) > 0:
                    vals.append(spearmanr(b, t).correlation)
            return float(np.mean(vals)) if vals else None
        d["spearman_visual_own_vs_base"] = spearman(own)
        if oth:
            d["spearman_visual_other_vs_base"] = spearman(oth)
        out[ds] = d
        ex = []
        for name, ids in (("hateful", hateful), ("nonhateful", nonhate)):
            for v in ids[:2]:
                ex.append(dict(dataset=ds, video_id=v, label=name, stance=own[v]["stance"], z_video=own[v]["global_margin"], plan=own[v]["plan"]["text"]))
        examples[ds] = ex
    out["examples"] = examples
    out["note"] = "labels used only here, post hoc; group / time / nothing detectors are keyword regexes (see diagnose.py)"
    (R / "diagnostics.json").write_text(json.dumps(out, indent=1) + "\n")
    for ds in DATASETS:
        if ds in out:
            print(ds, json.dumps({k: v for k, v in out[ds].items()}, indent=1))
    for ds, ex in examples.items():
        for e in ex:
            print(f"\n== {ds} {e['video_id']} ({e['label']}, verdict {e['stance']} z={e['z_video']:.2f})\n{e['plan'][:700]}")


if __name__ == "__main__":
    main()
