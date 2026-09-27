#!/usr/bin/env python3
"""Cross-corpus error analysis of r3_m2 on HateMM, HateClipSeg and DeHate (README.md). Reads test labels (rule 10);
analysis only, nothing here feeds a method.

A. Which level limits pooled: r3_m2 score = video key + within-video term. Oracle-video replaces the key by the gold
   video label (x100); oracle-within replaces the within term by the gold frame label (minus .5). Pooled metrics by
   the shared evaluator's `pooled`.
B. Video level: AUC / AP of the key; verdict > 0 rate among hateful and non-hateful videos; protected-group mention
   rate of the transcript for false-positive vs true-negative videos.
C. Within level (videos with both classes): mean ROC, share inverted (< .5), by hate coverage and by GT segment
   length; visual-only / speech-only / max-of-branches read orders; DeHate by modality flags.
D. Window level: OLS of the within-video standardized read on the window label and a protected-group mention.
E. False alarms inside hateful videos: negative frames in the top 20 % of their video's scores, distance to the
   nearest gold-positive frame vs all negative frames of the same videos.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval.evaluate import pooled  # noqa: E402
from src.video_inputs import load_asr, window_text  # noqa: E402

CORPORA = {
    "HateMM": (ROOT / "runs/20260926_twolevel/r3_m2", ROOT / "runs/20260926_glr/base_gridA"),
    "HateClipSeg": (ROOT / "runs/20260926_twolevel/r3_m2", ROOT / "runs/20260926_glr/base_gridA"),
    "DeHate": (ROOT / "runs/20260927_dehate_external/r3_m2", ROOT / "runs/20260927_dehate_external/reads_gridA"),
}
GROUP_WORDS = {  # same list as experiments/20260926_glr/glr_analyze.py (group names, no slurs)
    "black", "blacks", "white", "whites", "jew", "jews", "jewish", "muslim", "muslims", "islam", "islamic", "arab",
    "arabs", "asian", "asians", "chinese", "mexican", "mexicans", "latino", "latinos", "hispanic", "hispanics",
    "immigrant", "immigrants", "migrant", "migrants", "refugee", "refugees", "gay", "gays", "lesbian", "lesbians",
    "homosexual", "homosexuals", "lgbt", "trans", "transgender", "women", "woman", "christian", "christians",
    "hindu", "hindus", "indian", "indians", "african", "africans"}
OUT = ROOT / "runs/20260927_error_analysis"


def words(text):
    return {w.strip(".,!?;:\"'()[]").lower() for w in text.split()}


def load(ds):
    pdir, rdir = CORPORA[ds]
    g = np.load(ROOT / f"data/gt_4fps/{ds}.npz", allow_pickle=True)
    Y = {str(v): np.asarray(y, int) for v, y in zip(g["video_ids"], g["y4"])}
    P, R = {}, {}
    for l in open(pdir / "predictions.jsonl"):
        r = json.loads(l)
        if r["dataset"] == ds and r["video_id"] in Y:
            P[r["video_id"]] = r
    for l in open(rdir / "predictions.jsonl"):
        r = json.loads(l)
        if r["dataset"] == ds and r["video_id"] in Y:
            R[r["video_id"]] = r
    return Y, P, R


def within(y, s):
    n = min(len(y), len(s)); y, s = y[:n], s[:n]
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else None


def window_frames(w, n):
    return np.arange(int(round(w["start"] * 4)), min(n, int(round(w["end"] * 4))))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    flags = {}
    csv.field_size_limit(1 << 30)
    for r in csv.DictReader(open(Path.home() / "data/DeHate/DeHate_labels.csv", encoding="utf-8")):
        flags[r["Video ID"].strip()] = r
    report, lines = {}, []
    for ds in CORPORA:
        Y, P, R = load(ds)
        asr = load_asr(ds)
        vids = sorted(set(Y) & set(P))
        rep = {"n_videos": len(vids)}
        lab = {v: int(Y[v].any()) for v in vids}
        # ---- A
        key = {v: float(P[v]["extra"]["p_video_logodds"]) for v in vids}
        S = {v: np.asarray(P[v]["score_curve"], float) for v in vids}
        act = pooled(Y, S)
        ov = {v: (S[v] - key[v]) + 100.0 * lab[v] for v in vids}
        ow = {}
        for v in vids:
            n = len(S[v]); y = np.zeros(n); m = min(n, len(Y[v])); y[:m] = Y[v][:m]
            ow[v] = key[v] + (y - 0.5)
        rep["A"] = {"actual": [act["frame_ROC_AUC"], act["frame_PR_AUC"]],
                    "oracle_video": [pooled(Y, ov)["frame_ROC_AUC"], pooled(Y, ov)["frame_PR_AUC"]],
                    "oracle_within": [pooled(Y, ow)["frame_ROC_AUC"], pooled(Y, ow)["frame_PR_AUC"]],
                    "base_rate": act["base_rate"]}
        # ---- B
        y_v = np.array([lab[v] for v in vids]); k_v = np.array([key[v] for v in vids])
        zv = np.array([float(P[v]["extra"]["z_video"]) for v in vids])
        mention = np.array([bool(words(" ".join(t for _, _, t in asr.get(v, []))) & GROUP_WORDS) for v in vids])
        fp = (y_v == 0) & (zv > 0); tn = (y_v == 0) & (zv <= 0)
        rep["B"] = {"video_auc": roc_auc_score(y_v, k_v), "video_ap": average_precision_score(y_v, k_v),
                    "share_hateful": y_v.mean(), "verdict_pos_hateful": (zv[y_v == 1] > 0).mean(),
                    "verdict_pos_nonhateful": (zv[y_v == 0] > 0).mean(),
                    "mention_rate_fp": mention[fp].mean() if fp.any() else None,
                    "mention_rate_tn": mention[tn].mean() if tn.any() else None,
                    "mention_rate_hateful": mention[y_v == 1].mean()}
        # ---- C
        rows = []
        for v in vids:
            y = Y[v]
            if not (0 < y.sum() < len(y)):
                continue
            n = len(y)
            wins = R[v]["extra"]["windows"]
            fv = np.full(n, np.nan); fs = np.full(n, np.nan)
            for w in wins:
                idx = window_frames(w, n)
                fv[idx] = w["z_visual"]; fs[idx] = w["z_speech"] if w.get("z_speech") is not None else -30.0
            fv = np.nan_to_num(fv, nan=-30.0); fs = np.nan_to_num(fs, nan=-30.0)
            runs = np.flatnonzero(np.diff(np.r_[0, y, 0])).reshape(-1, 2)
            rows.append({"v": v, "auc": within(y, S[v]), "cov": y.mean(), "seg_med_s": float(np.median(runs[:, 1] - runs[:, 0]) / 4),
                         "vis": within(y, fv), "sp": within(y, fs), "mx": within(y, np.maximum(fv, fs))})
        a = np.array([r["auc"] for r in rows])
        cov = np.array([r["cov"] for r in rows]); seg = np.array([r["seg_med_s"] for r in rows])
        bins = lambda x, e: {f"{lo}-{hi}": (float(a[(x >= lo) & (x < hi)].mean()), int(((x >= lo) & (x < hi)).sum()))
                             for lo, hi in zip(e[:-1], e[1:]) if ((x >= lo) & (x < hi)).any()}
        rep["C"] = {"n": len(rows), "mean": a.mean(), "inverted_share": (a < .5).mean(),
                    "by_coverage": bins(cov, [0, .25, .5, .75, 1.01]),
                    "by_median_segment_s": bins(seg, [0, 8, 16, 32, 1e9]),
                    "reads_visual": np.mean([r["vis"] for r in rows]), "reads_speech": np.mean([r["sp"] for r in rows]),
                    "reads_max": np.mean([r["mx"] for r in rows])}
        if ds == "DeHate":
            mod = {}
            for name in ("Textual Content", "Visual Content", "Audio Content"):
                sel = [r["auc"] for r in rows if flags[r["v"]][name] == "1"]
                mod[name] = (float(np.mean(sel)), len(sel)) if sel else None
            for kind in ("Explicit", "Implicit"):
                sel = [r["auc"] for r in rows if flags[r["v"]]["Explicit or Implicit"] == kind]
                mod[kind] = (float(np.mean(sel)), len(sel)) if sel else None
            vis_only = [r["auc"] for r in rows if flags[r["v"]]["Visual Content"] == "1" and flags[r["v"]]["Audio Content"] == "0"]
            aud_only = [r["auc"] for r in rows if flags[r["v"]]["Audio Content"] == "1" and flags[r["v"]]["Visual Content"] == "0"]
            mod["visual_not_audio"] = (float(np.mean(vis_only)), len(vis_only)) if vis_only else None
            mod["audio_not_visual"] = (float(np.mean(aud_only)), len(aud_only)) if aud_only else None
            rep["C"]["dehate_flags"] = mod
        # ---- D
        X, Z = [], []
        for v in vids:
            y = Y[v]
            if not (0 < y.sum() < len(y)):
                continue
            wins = R[v]["extra"]["windows"]; recs = []
            for w in wins:
                idx = window_frames(w, len(y))
                if len(idx) == 0:
                    continue
                recs.append((int(y[idx].mean() >= .5), int(bool(words(window_text(asr.get(v, []), w["start"], w["end"])) & GROUP_WORDS)), w["z"]))
            zz = np.array([r[2] for r in recs], float)
            if len(recs) < 2 or zz.std() == 0 or len({r[0] for r in recs}) < 2:
                continue
            for r, z in zip(recs, (zz - zz.mean()) / zz.std()):
                X.append([1.0, r[0], r[1]]); Z.append(z)
        X, Z = np.array(X), np.array(Z)
        beta = np.linalg.lstsq(X, Z, rcond=None)[0]
        rep["D"] = {"n_windows": len(Z), "coef_label": beta[1], "coef_mention": beta[2], "mention_rate": X[:, 2].mean()}
        # ---- E
        near, base = [], []
        for v in vids:
            y = Y[v]
            if not (0 < y.sum() < len(y)):
                continue
            s = S[v][:len(y)]; y = y[:len(s)]
            pos = np.flatnonzero(y == 1); neg = np.flatnonzero(y == 0)
            d = np.abs(neg[:, None] - pos[None, :]).min(1) / 4.0
            top = s[neg] >= np.quantile(s, .8)
            near.extend(d[top]); base.extend(d)
        near, base = np.array(near), np.array(base)
        rep["E"] = {"top_false_alarm_within_8s": (near <= 8).mean(), "all_negative_within_8s": (base <= 8).mean(),
                    "top_false_alarm_median_s": float(np.median(near)), "all_negative_median_s": float(np.median(base))}
        report[ds] = rep
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=float))
    for ds, r in report.items():
        A, B, C, D, E = r["A"], r["B"], r["C"], r["D"], r["E"]
        lines += [f"== {ds} ({r['n_videos']} videos, frame base rate {A['base_rate']:.3f})",
                  f"A pooled ROC/PR  actual {A['actual'][0]:.3f}/{A['actual'][1]:.3f}  oracle-video {A['oracle_video'][0]:.3f}/{A['oracle_video'][1]:.3f}  "
                  f"oracle-within {A['oracle_within'][0]:.3f}/{A['oracle_within'][1]:.3f}",
                  f"B video AUC {B['video_auc']:.3f} AP {B['video_ap']:.3f} (hateful share {B['share_hateful']:.3f}); verdict>0: hateful {B['verdict_pos_hateful']:.3f}, "
                  f"non-hateful {B['verdict_pos_nonhateful']:.3f}; group mention: FP {B['mention_rate_fp']}, TN {B['mention_rate_tn']}, hateful {B['mention_rate_hateful']:.3f}",
                  f"C within {C['mean']:.3f} (n {C['n']}, inverted {C['inverted_share']:.3f}); reads visual {C['reads_visual']:.3f} speech {C['reads_speech']:.3f} max {C['reads_max']:.3f}",
                  f"  by coverage {C['by_coverage']}", f"  by median segment (s) {C['by_median_segment_s']}"]
        if "dehate_flags" in C:
            lines.append(f"  DeHate flags {C['dehate_flags']}")
        lines += [f"D window read ~ label {D['coef_label']:+.3f} + group mention {D['coef_mention']:+.3f} (n {D['n_windows']}, mention rate {D['mention_rate']:.3f})",
                  f"E top false alarms within 8 s of hate {E['top_false_alarm_within_8s']:.3f} (all negatives {E['all_negative_within_8s']:.3f}); "
                  f"median distance {E['top_false_alarm_median_s']:.1f} s (all {E['all_negative_median_s']:.1f} s)", ""]
    (OUT / "table.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
