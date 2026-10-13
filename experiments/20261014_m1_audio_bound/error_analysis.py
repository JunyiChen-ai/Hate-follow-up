#!/usr/bin/env python3
"""Candidate 43 rule-10 error analysis (CPU, reads test GT; logged in README §8). Window-level reads from
runs/20261014_m1_audio_bound/main/records, decoded curves from main_decoded/<arm>, r6_bma and adjacent_local.
Window gold = majority of the 4 fps GT frames inside the window. Writes main_analysis/error_analysis.{json,txt}."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "runs/20261014_m1_audio_bound"
DATASETS = ("HateMM", "HateClipSeg")
BRANCHES = ("visual_native", "visual_local", "speech_prop", "speech_aligned", "speech_audio", "speech_swap", "speech_nbr")
ARMS = {"audio": ("visual_local", "speech_audio"), "text_aligned": ("visual_local", "speech_aligned"), "text_prop": ("visual_local", "speech_prop"),
        "audio_swap": ("visual_local", "speech_swap"), "audio_nbr": ("visual_local", "speech_nbr"), "r6_inputs": ("visual_native", "speech_prop")}
DECODED = {"r6_bma": ROOT / "runs/20260926_twolevel/r6_bma/predictions.jsonl",
           "adjacent_local": ROOT / "runs/20261007_m1_streamingtom/placement_controls_main_decoded/adjacent_local/predictions.jsonl",
           **{a: EXP / "main_decoded" / a / "predictions.jsonl" for a in ARMS}}


def gt(ds):
    g = np.load(ROOT / "data/gt_4fps" / f"{ds}.npz", allow_pickle=True)
    return {v: np.asarray(y, dtype=np.int8) for v, y in zip(g["video_ids"], g["y4"])}


def auc(y, s):
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    if len(y) < 2 or y.min() == y.max():
        return None
    return float(roc_auc_score(y, s))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return (float(np.mean(xs)), len(xs)) if xs else (None, 0)


def branch_value(t, b):
    v = t[b]
    if v is None:
        return None
    return v if isinstance(v, (int, float)) else v["z"]


def arm_z(t, arm):
    vb, sb = ARMS[arm]
    v = branch_value(t, vb)
    if v is None:  # visual_local falls back to visual_native in the records' predictions
        v = branch_value(t, "visual_native")
    s = branch_value(t, sb)
    return max(v, s) if s is not None else v


def curves(path):
    out = {}
    for r in map(json.loads, open(path)):
        if not r.get("error"):
            out[(r["dataset"], r["video_id"])] = np.asarray(r["score_curve"], dtype=float)
    return out


def main():
    dec = {k: curves(p) for k, p in DECODED.items()}
    report, lines = {}, []
    for ds in DATASETS:
        y4 = gt(ds)
        recs = {p.stem: json.loads(p.read_text()) for p in sorted((EXP / "main/records" / ds).glob("*.json"))}
        win_auc = {b: [] for b in BRANCHES}
        arm_auc = {a: [] for a in ARMS}
        silent = dict(windows=0, positive=0, audio_scores=[], visual_scores=[], gold=[], rank_audio=[], rank_text=[], rank_prop=[], videos_with_silent_positive=0)
        per_video = []
        for vid, b in recs.items():
            y = y4[vid]
            labels, any_pos, sil = [], [], []
            for t in b["traces"]:
                t1, t2 = t["bounds"]
                seg = y[int(round(t1 * 4)):int(round(t2 * 4))]
                labels.append(int(seg.mean() > .5) if len(seg) else 0)
                any_pos.append(int(seg.max()) if len(seg) else 0)
                sil.append(t["speech_prop"] is None)
            labels = np.asarray(labels)
            for br in BRANCHES:
                vals = [(branch_value(t, br), l) for t, l in zip(b["traces"], labels) if branch_value(t, br) is not None]
                if vals:
                    win_auc[br].append(auc([l for _, l in vals], [v for v, _ in vals]))
            zs = {a: np.asarray([arm_z(t, a) for t in b["traces"]]) for a in ARMS}
            for a in ARMS:
                arm_auc[a].append(auc(labels, zs[a]))
            # silent windows: what does the audio read say where ASR gave nothing?
            has_silent_pos = False
            for k, (t, l, s) in enumerate(zip(b["traces"], labels, sil)):
                if s:
                    silent["windows"] += 1; silent["positive"] += int(l); silent["gold"].append(int(l))
                    silent["audio_scores"].append(branch_value(t, "speech_audio")); silent["visual_scores"].append(zs["text_prop"][k])
                    if l:
                        has_silent_pos = True
                        for key, a in (("rank_audio", "audio"), ("rank_text", "text_aligned"), ("rank_prop", "text_prop")):
                            silent[key].append(float((zs[a] < zs[a][k]).mean()))
            silent["videos_with_silent_positive"] += has_silent_pos
            n = min(len(y), *(len(dec[k][(ds, vid)]) for k in dec if (ds, vid) in dec[k]))
            within = {k: auc(y[:n], dec[k][(ds, vid)][:n]) if (ds, vid) in dec[k] else None for k in dec}
            per_video.append(dict(video_id=vid, windows=len(labels), silent_share=float(np.mean(sil)), silent_positive=has_silent_pos,
                                  hate_share=float(y.mean()), within=within))
        # per-video decoded within deltas
        defined = [v for v in per_video if all(v["within"].get(k) is not None for k in ("audio", "r6_bma", "adjacent_local", "text_aligned", "text_prop"))]
        def delta(a, b, rows):
            d = np.asarray([v["within"][a] - v["within"][b] for v in rows])
            return dict(n=len(d), mean=float(d.mean()), median=float(np.median(d)), improved=float((d > 0).mean()), p10=float(np.percentile(d, 10)), p90=float(np.percentile(d, 90))) if len(d) else None
        groups = {"all": defined, "silent_positive": [v for v in defined if v["silent_positive"]], "no_silent_positive": [v for v in defined if not v["silent_positive"]],
                  "speech_poor(silent>=.5)": [v for v in defined if v["silent_share"] >= .5], "speech_rich(silent<.5)": [v for v in defined if v["silent_share"] < .5],
                  "edge_structure(hate>=.75)": [v for v in defined if v["hate_share"] >= .75], "other(hate<.75)": [v for v in defined if v["hate_share"] < .75]}
        report[ds] = dict(
            videos=len(recs), windows=int(sum(v["windows"] for v in per_video)),
            window_auc_within={br: dict(zip(("mean", "n_videos"), mean(win_auc[br]))) for br in BRANCHES},
            arm_window_auc_within={a: dict(zip(("mean", "n_videos"), mean(arm_auc[a]))) for a in ARMS},
            silent=dict(windows=silent["windows"], share=silent["windows"] / max(1, sum(v["windows"] for v in per_video)), positive=silent["positive"],
                        positive_share=silent["positive"] / max(1, silent["windows"]), videos_with_silent_positive=silent["videos_with_silent_positive"],
                        pooled_auc_audio_read=auc(silent["gold"], silent["audio_scores"]), pooled_auc_visual_fallback=auc(silent["gold"], silent["visual_scores"]),
                        silent_positive_rank_pct=dict(audio=mean(silent["rank_audio"])[0], text_aligned=mean(silent["rank_text"])[0], text_prop=mean(silent["rank_prop"])[0])),
            decoded_within_mean={k: dict(zip(("mean", "n_videos"), mean([v["within"].get(k) for v in defined]))) for k in dec},
            deltas={g: {"audio-r6_bma": delta("audio", "r6_bma", rows), "audio-adjacent_local": delta("audio", "adjacent_local", rows),
                        "audio-text_aligned": delta("audio", "text_aligned", rows), "text_aligned-text_prop": delta("text_aligned", "text_prop", rows),
                        "text_prop-adjacent_local": delta("text_prop", "adjacent_local", rows)} for g, rows in groups.items()},
            per_video=per_video)
        r = report[ds]
        lines.append(f"== {ds}: {r['videos']} videos, {r['windows']} windows")
        lines.append("window AUC within video (mean over videos with both classes): " + ", ".join(f"{b} {v['mean']:.3f}" for b, v in r["window_auc_within"].items() if v["mean"] is not None))
        lines.append("arm z window AUC within: " + ", ".join(f"{a} {v['mean']:.3f}" for a, v in r["arm_window_auc_within"].items()))
        s = r["silent"]
        lines.append(f"ASR-silent windows {s['windows']} ({s['share']:.1%}), gold-positive {s['positive']} ({s['positive_share']:.1%}) in {s['videos_with_silent_positive']} videos; "
                     f"pooled AUC on silent windows: audio read {s['pooled_auc_audio_read']}, visual fallback {s['pooled_auc_visual_fallback']}; "
                     f"silent-positive rank pct within video: audio {s['silent_positive_rank_pct']['audio']}, text_aligned {s['silent_positive_rank_pct']['text_aligned']}, text_prop {s['silent_positive_rank_pct']['text_prop']}")
        lines.append("decoded within mean (videos defined for all): " + ", ".join(f"{k} {v['mean']:.3f}" for k, v in r["decoded_within_mean"].items() if v["mean"] is not None))
        for g, dd in r["deltas"].items():
            d = dd["audio-r6_bma"]
            if d:
                lines.append(f"  {g:26s} n={d['n']:3d} audio-r6_bma mean {d['mean']:+.3f} median {d['median']:+.3f} improved {d['improved']:.2f} | audio-adjacent_local {dd['audio-adjacent_local']['mean']:+.3f} | "
                             f"audio-text_aligned {dd['audio-text_aligned']['mean']:+.3f} | text_aligned-text_prop {dd['text_aligned-text_prop']['mean']:+.3f} | text_prop-adjacent_local {dd['text_prop-adjacent_local']['mean']:+.3f}")
    out = EXP / "main_analysis"
    (out / "error_analysis.json").write_text(json.dumps(report, indent=1) + "\n")
    (out / "error_analysis.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
