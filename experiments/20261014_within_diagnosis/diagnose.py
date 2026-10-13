#!/usr/bin/env python3
"""Within-video error diagnosis of the current best configuration (r6 reads + in-window frames) on HateMM and
HateClipSeg. Reads test labels (rule 10); analysis only, nothing here feeds a method. README.md has the questions,
the findings and the test-read log.

Configurations (raw reads, decoded curve):
  r6        runs/20260926_glr/base_gridA                              runs/20260926_twolevel/r6_bma
  adjacent  runs/20261007_m1_streamingtom/placement_controls_main/adjacent_local   ..._decoded/adjacent_local
  E         runs/20261007_m1_streamingtom/local_controls_main/local_clean          ..._decoded/local_clean
Windows are the 8 s reading windows of the raw record (`extra.windows`: z_visual, z_speech or absent, z = max).
Window gold = share of positive 4 fps frames in the window; positive window = share >= .5, clean negative = share 0.

A. Per video (videos with both classes): window-level AUC of visual-only, speech-only, max read, decoded window
   mean; frame-level within of the raw max read (piecewise constant) and of the decoded curve (the metric).
   Decoder effect = decoded within - raw within, per video.
B. Mis-ordered pairs (positive window below or equal to a clean negative window under the decoded window mean):
   which channel supplied the negative window's read, whether visual-only / speech-only would have ordered the
   pair correctly, whether the positive window has no speech, whether the negative window is adjacent to a
   positive window, whether its transcript mentions a protected group. Pooled over pairs and macro over videos.
C. No-speech windows: share, positive rate, rank, share of mis-ordered pairs involving them; window-level AUC over
   speech windows only.
D. Error concentration: per-video within distribution, deficit share of the worst videos, overlap of inverted
   videos between r6 and the frames configuration; the worst videos with their properties.
E. Frames gain by channel: visual-only / max / decoded within, r6 vs frames; speech reads asserted identical.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.video_inputs import load_asr, window_text  # noqa: E402

OUT = ROOT / "runs/20261014_within_diagnosis"
DATASETS = ("HateMM", "HateClipSeg")
CONFIGS = {
    "r6": (ROOT / "runs/20260926_glr/base_gridA", ROOT / "runs/20260926_twolevel/r6_bma"),
    "adjacent": (ROOT / "runs/20261007_m1_streamingtom/placement_controls_main/adjacent_local",
                 ROOT / "runs/20261007_m1_streamingtom/placement_controls_main_decoded/adjacent_local"),
    "E": (ROOT / "runs/20261007_m1_streamingtom/local_controls_main/local_clean",
          ROOT / "runs/20261007_m1_streamingtom/local_controls_main_decoded/local_clean"),
}
GROUP_WORDS = {  # same list as experiments/20260927_error_analysis/error_analysis.py (group names, no slurs)
    "black", "blacks", "white", "whites", "jew", "jews", "jewish", "muslim", "muslims", "islam", "islamic", "arab",
    "arabs", "asian", "asians", "chinese", "mexican", "mexicans", "latino", "latinos", "hispanic", "hispanics",
    "immigrant", "immigrants", "migrant", "migrants", "refugee", "refugees", "gay", "gays", "lesbian", "lesbians",
    "homosexual", "homosexuals", "lgbt", "trans", "transgender", "women", "woman", "christian", "christians",
    "hindu", "hindus", "indian", "indians", "african", "africans"}
NOSPEECH = -30.0


def words(text):
    return {w.strip(".,!?;:\"'()[]").lower() for w in text.split()}


def records(path, ds):
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r["dataset"] == ds and not r.get("error"):
            out[r["video_id"]] = r
    return out


def gold(ds):
    g = np.load(ROOT / f"data/gt_4fps/{ds}.npz", allow_pickle=True)
    return {str(v): np.asarray(y, int) for v, y, s in zip(g["video_ids"], g["y4"], g["split"]) if str(s) == "test"}


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None


def frames_of(w, n):
    return np.arange(int(round(w["start"] * 4)), min(n, int(round(w["end"] * 4))))


def video_table(y, raw, dec, asr):
    """One row per window of one video."""
    n = len(y); curve = np.asarray(dec["score_curve"], float)[:n]
    rows = []
    for w in raw["extra"]["windows"]:
        idx = frames_of(w, n)
        if len(idx) == 0:
            continue
        zs = w.get("z_speech")
        rows.append({"i": w["i"], "share": float(y[idx].mean()), "zv": float(w["z_visual"]),
                     "zs": None if zs is None else float(zs), "z": float(w["z"]),
                     "dec": float(curve[idx].mean()),
                     "deciding": "nospeech" if zs is None else ("speech" if zs >= w["z_visual"] else "visual"),
                     "mention": bool(words(window_text(asr, w["start"], w["end"])) & GROUP_WORDS)})
    return rows, curve


def raw_curve(rows, n):
    c = np.full(n, NOSPEECH)
    for r, w in rows:
        c[frames_of(w, n)] = r["z"]
    return c


def pair_analysis(rows):
    """Mis-ordered (positive, clean negative) pairs under the decoded window mean and under the raw max read."""
    pos = [r for r in rows if r["share"] >= .5]; neg = [r for r in rows if r["share"] == 0]
    pos_idx = {r["i"] for r in pos}
    out = {"n_pairs": len(pos) * len(neg), "n_pos": len(pos), "n_neg": len(neg)}
    if not pos or not neg:
        return None
    keys = ["mis_dec", "mis_raw", "mis_dec_ok_raw", "mis_raw_ok_dec", "neg_from_visual", "neg_from_speech",
            "neg_nospeech", "pos_nospeech", "visual_would_fix", "speech_would_fix", "both_have_speech",
            "neither_fixes", "neg_adjacent_pos", "neg_mention", "pos_mention"]
    c = {k: 0 for k in keys}
    for p in pos:
        for q in neg:
            md = q["dec"] >= p["dec"]; mr = q["z"] >= p["z"]
            c["mis_dec"] += md; c["mis_raw"] += mr; c["mis_dec_ok_raw"] += md and not mr; c["mis_raw_ok_dec"] += mr and not md
            if not md:
                continue
            c["neg_from_visual"] += q["deciding"] == "visual"; c["neg_from_speech"] += q["deciding"] == "speech"
            c["neg_nospeech"] += q["deciding"] == "nospeech"; c["pos_nospeech"] += p["deciding"] == "nospeech"
            vf = p["zv"] > q["zv"]; both = p["zs"] is not None and q["zs"] is not None
            sf = both and p["zs"] > q["zs"]
            c["visual_would_fix"] += vf; c["speech_would_fix"] += sf; c["both_have_speech"] += both
            c["neither_fixes"] += (not vf) and (not sf)
            c["neg_adjacent_pos"] += (q["i"] - 1 in pos_idx) or (q["i"] + 1 in pos_idx)
            c["neg_mention"] += q["mention"]; c["pos_mention"] += p["mention"]
    out.update({k: int(v) for k, v in c.items()})
    return out


def analyse(ds, cfg, Y, asr_all):
    raw_dir, dec_dir = CONFIGS[cfg]
    R = records(raw_dir / "predictions.jsonl", ds); D = records(dec_dir / "predictions.jsonl", ds)
    vids = sorted(v for v in Y if v in R and v in D)
    per_video = []; all_windows = []
    for v in vids:
        y = Y[v]
        rows, curve = video_table(y, R[v], D[v], asr_all.get(v, []))
        n = len(y); win_objs = R[v]["extra"]["windows"]
        rc = raw_curve(list(zip(rows, [w for w in win_objs if len(frames_of(w, n))])), n)
        lab = [int(r["share"] >= .5) for r in rows]
        vis = [r["zv"] for r in rows]; sp = [NOSPEECH if r["zs"] is None else r["zs"] for r in rows]
        mx = [r["z"] for r in rows]; dm = [r["dec"] for r in rows]
        sp_rows = [r for r in rows if r["zs"] is not None]
        rec = {"video": v, "eligible": bool(0 < y.sum() < n), "hate_share": float(y.mean()), "duration_s": n / 4,
               "n_windows": len(rows), "n_gold_runs": int((np.diff(np.r_[0, y, 0]) == 1).sum()),
               "nospeech_share": float(np.mean([r["zs"] is None for r in rows])),
               "z_video": float(R[v]["extra"]["z_video"]), "stance": R[v]["extra"].get("stance"),
               "within_frame_dec": auc(y, curve), "within_frame_raw": auc(y, rc),
               "win_auc_visual": auc(lab, vis), "win_auc_speech": auc(lab, sp), "win_auc_max": auc(lab, mx),
               "win_auc_dec": auc(lab, dm),
               "win_auc_max_speechwin": auc([int(r["share"] >= .5) for r in sp_rows], [r["z"] for r in sp_rows]) if sp_rows else None,
               "win_auc_dec_speechwin": auc([int(r["share"] >= .5) for r in sp_rows], [r["dec"] for r in sp_rows]) if sp_rows else None,
               "pairs": pair_analysis(rows)}
        per_video.append(rec)
        # rank percentile of each window inside its video under the decoded mean (1 = top)
        order = np.argsort(np.argsort(-np.asarray(dm))) / max(1, len(dm) - 1) if len(dm) > 1 else np.zeros(len(dm))
        for r, pr in zip(rows, order):
            all_windows.append({**r, "video": v, "eligible": rec["eligible"], "rank_pct": float(pr)})
    return per_video, all_windows


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def summarise(per_video, all_windows):
    E = [r for r in per_video if r["eligible"] and r["within_frame_dec"] is not None]
    s = {"n_videos": len(per_video), "n_eligible": len(E)}
    for k in ("within_frame_dec", "within_frame_raw", "win_auc_visual", "win_auc_speech", "win_auc_max", "win_auc_dec",
              "win_auc_max_speechwin", "win_auc_dec_speechwin"):
        s[k] = mean([r[k] for r in E])
    d = np.array([r["within_frame_dec"] - r["within_frame_raw"] for r in E if r["within_frame_raw"] is not None])
    s["decoder_effect"] = {"mean": float(d.mean()), "helps_gt_.02": float((d > .02).mean()), "hurts_gt_.02": float((d < -.02).mean()),
                           "median": float(np.median(d))}
    dv = [r for r in E if r["win_auc_visual"] is not None and r["win_auc_speech"] is not None]
    s["channel_per_video"] = {"visual_better_than_speech": float(np.mean([r["win_auc_visual"] > r["win_auc_speech"] for r in dv])),
                              "n": len(dv),
                              "dec_below_visual": float(np.mean([r["win_auc_dec"] < r["win_auc_visual"] for r in dv])),
                              "dec_below_speech": float(np.mean([r["win_auc_dec"] < r["win_auc_speech"] for r in dv])),
                              "dec_below_both": float(np.mean([r["win_auc_dec"] < min(r["win_auc_visual"], r["win_auc_speech"]) for r in dv])),
                              "dec_above_both": float(np.mean([r["win_auc_dec"] > max(r["win_auc_visual"], r["win_auc_speech"]) for r in dv]))}
    # B pooled and macro
    P = [r["pairs"] for r in E if r["pairs"]]
    keys = [k for k in P[0] if k.startswith(("mis", "neg", "pos", "visual", "speech", "both", "neither"))]
    tot = {k: sum(p[k] for p in P) for k in keys}; npairs = sum(p["n_pairs"] for p in P); nmis = tot["mis_dec"]
    pooled = {"n_pairs": npairs, "mis_dec_rate": tot["mis_dec"] / npairs, "mis_raw_rate": tot["mis_raw"] / npairs,
              "mis_dec_ok_raw_rate": tot["mis_dec_ok_raw"] / npairs, "mis_raw_ok_dec_rate": tot["mis_raw_ok_dec"] / npairs}
    for k in ("neg_from_visual", "neg_from_speech", "neg_nospeech", "pos_nospeech", "visual_would_fix", "speech_would_fix",
              "both_have_speech", "neither_fixes", "neg_adjacent_pos", "neg_mention", "pos_mention"):
        pooled[k + "_share_of_mis"] = tot[k] / nmis if nmis else None
    macro = {}
    for k in ("neg_from_visual", "neg_from_speech", "neg_nospeech", "visual_would_fix", "speech_would_fix", "neither_fixes", "neg_adjacent_pos"):
        macro[k + "_share_of_mis"] = mean([p[k] / p["mis_dec"] for p in P if p["mis_dec"]])
    macro["mis_dec_rate"] = mean([p["mis_dec"] / p["n_pairs"] for p in P])
    s["pairs_pooled"] = pooled; s["pairs_macro"] = macro
    # C no-speech windows
    W = [w for w in all_windows]; We = [w for w in all_windows if w["eligible"]]
    ns = [w for w in We if w["zs"] is None]; ws = [w for w in We if w["zs"] is not None]
    s["nospeech"] = {"share_all_videos": float(np.mean([w["zs"] is None for w in W])),
                     "share_eligible_videos": float(np.mean([w["zs"] is None for w in We])),
                     "pos_rate_nospeech": float(np.mean([w["share"] >= .5 for w in ns])) if ns else None,
                     "pos_rate_speech": float(np.mean([w["share"] >= .5 for w in ws])),
                     "mean_rank_pct_nospeech_pos": mean([w["rank_pct"] for w in ns if w["share"] >= .5]),
                     "mean_rank_pct_nospeech_neg": mean([w["rank_pct"] for w in ns if w["share"] == 0]),
                     "mean_rank_pct_speech_pos": mean([w["rank_pct"] for w in ws if w["share"] >= .5]),
                     "mean_rank_pct_speech_neg": mean([w["rank_pct"] for w in ws if w["share"] == 0]),
                     "mean_zv_nospeech": mean([w["zv"] for w in ns]), "mean_zv_speech": mean([w["zv"] for w in ws]),
                     "videos_with_nospeech_window": float(np.mean([r["nospeech_share"] > 0 for r in E]))}
    # D concentration
    a = np.array([r["within_frame_dec"] for r in E]); deficit = 1 - a; order = np.argsort(-deficit)
    k10 = max(1, int(round(.1 * len(a)))); k20 = max(1, int(round(.2 * len(a))))
    s["concentration"] = {"quantiles_10_25_50_75_90": [float(x) for x in np.quantile(a, [.1, .25, .5, .75, .9])],
                          "share_below_.5": float((a < .5).mean()), "share_below_.6": float((a < .6).mean()), "share_above_.9": float((a > .9).mean()),
                          "deficit_total": float(deficit.sum()),
                          "deficit_share_worst_10pct": float(deficit[order[:k10]].sum() / deficit.sum()),
                          "deficit_share_worst_20pct": float(deficit[order[:k20]].sum() / deficit.sum()),
                          "within_if_worst_10pct_at_.5": float(np.mean(np.where(np.isin(np.arange(len(a)), order[:k10]), np.maximum(a, .5), a))),
                          "within_if_worst_10pct_at_1": float(np.mean(np.where(np.isin(np.arange(len(a)), order[:k10]), 1., a)))}
    # by hate share and by number of gold runs
    def group(key, edges):
        g = {}
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = [r for r in E if lo <= r[key] < hi]
            if sel:
                g[f"{lo}-{hi}"] = {"n": len(sel), "within": mean([r["within_frame_dec"] for r in sel]),
                                   "visual": mean([r["win_auc_visual"] for r in sel]), "speech": mean([r["win_auc_speech"] for r in sel])}
        return g
    s["by_hate_share"] = group("hate_share", [0, .25, .5, .75, 1.01])
    s["by_gold_runs"] = group("n_gold_runs", [1, 2, 3, 5, 1000])
    s["by_nospeech_share"] = group("nospeech_share", [0, 1e-9, .25, .5, 1.01])
    return s


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"scope": "development-selected; reads test labels (rule 10); analysis only", "configs": {k: [str(p.relative_to(ROOT)) for p in v] for k, v in CONFIGS.items()}}
    per_video_all = {}
    for ds in DATASETS:
        Y = gold(ds); asr_all = load_asr(ds)
        report[ds] = {}
        for cfg in CONFIGS:
            pv, aw = analyse(ds, cfg, Y, asr_all)
            per_video_all[(ds, cfg)] = pv
            report[ds][cfg] = summarise(pv, aw)
        # E frames gain by channel + overlap of inverted videos (r6 vs adjacent), speech reads identical
        Rr = records(CONFIGS["r6"][0] / "predictions.jsonl", ds); Ra = records(CONFIGS["adjacent"][0] / "predictions.jsonl", ds)
        same_speech = all((w.get("z_speech") is None and x.get("z_speech") is None) or (w.get("z_speech") is not None and x.get("z_speech") is not None and abs(w["z_speech"] - x["z_speech"]) < 1e-9)
                          for v in Rr for w, x in zip(Rr[v]["extra"]["windows"], Ra[v]["extra"]["windows"]))
        A = {r["video"]: r for r in per_video_all[(ds, "r6")] if r["eligible"]}; B = {r["video"]: r for r in per_video_all[(ds, "adjacent")] if r["eligible"]}
        vids = sorted(set(A) & set(B))
        inv_a = {v for v in vids if A[v]["within_frame_dec"] < .5}; inv_b = {v for v in vids if B[v]["within_frame_dec"] < .5}
        dd = np.array([B[v]["within_frame_dec"] - A[v]["within_frame_dec"] for v in vids])
        report[ds]["frames_vs_r6"] = {
            "speech_reads_identical": bool(same_speech), "n": len(vids),
            "delta_within_frame_dec": float(dd.mean()), "delta_within_frame_raw": mean([B[v]["within_frame_raw"] - A[v]["within_frame_raw"] for v in vids]),
            "delta_win_auc_visual": mean([B[v]["win_auc_visual"] - A[v]["win_auc_visual"] for v in vids if A[v]["win_auc_visual"] is not None]),
            "delta_win_auc_max": mean([B[v]["win_auc_max"] - A[v]["win_auc_max"] for v in vids if A[v]["win_auc_max"] is not None]),
            "videos_improved_gt_.02": float((dd > .02).mean()), "videos_worsened_gt_.02": float((dd < -.02).mean()),
            "corr_per_video_within": float(np.corrcoef([A[v]["within_frame_dec"] for v in vids], [B[v]["within_frame_dec"] for v in vids])[0, 1]),
            "inverted_r6": len(inv_a), "inverted_adjacent": len(inv_b), "inverted_both": len(inv_a & inv_b),
            "fixed_by_frames": sorted(inv_a - inv_b), "broken_by_frames": sorted(inv_b - inv_a),
            "delta_by_r6_within_tercile": {}}
        q = np.quantile([A[v]["within_frame_dec"] for v in vids], [1 / 3, 2 / 3])
        for name, sel in (("low", [v for v in vids if A[v]["within_frame_dec"] < q[0]]), ("mid", [v for v in vids if q[0] <= A[v]["within_frame_dec"] < q[1]]), ("high", [v for v in vids if A[v]["within_frame_dec"] >= q[1]])):
            report[ds]["frames_vs_r6"]["delta_by_r6_within_tercile"][name] = {"n": len(sel), "delta": mean([B[v]["within_frame_dec"] - A[v]["within_frame_dec"] for v in sel])}
        # worst videos under the frames configuration
        worst = sorted((r for r in per_video_all[(ds, "adjacent")] if r["eligible"]), key=lambda r: r["within_frame_dec"])[:12]
        report[ds]["worst_adjacent"] = [{k: r[k] for k in ("video", "within_frame_dec", "within_frame_raw", "win_auc_visual", "win_auc_speech", "hate_share", "duration_s", "n_windows", "n_gold_runs", "nospeech_share", "z_video", "stance")}
                                        | {"r6_within": A[r["video"]]["within_frame_dec"]} for r in worst]
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "per_video.json").write_text(json.dumps({f"{ds}|{cfg}": pv for (ds, cfg), pv in per_video_all.items()}, indent=1) + "\n")
    # table
    L = []
    f = lambda x: "  -  " if x is None else f"{x:.3f}"
    for ds in DATASETS:
        L.append(f"== {ds}")
        L.append("A  config | within(dec) within(raw max) | win AUC visual speech max dec | speech-windows-only max dec | decoder effect mean helps hurts")
        for cfg in CONFIGS:
            s = report[ds][cfg]; de = s["decoder_effect"]
            L.append(f"   {cfg:9s}| {f(s['within_frame_dec'])} {f(s['within_frame_raw'])} | {f(s['win_auc_visual'])} {f(s['win_auc_speech'])} {f(s['win_auc_max'])} {f(s['win_auc_dec'])} | {f(s['win_auc_max_speechwin'])} {f(s['win_auc_dec_speechwin'])} | {de['mean']:+.3f} {de['helps_gt_.02']:.2f} {de['hurts_gt_.02']:.2f}")
        L.append("B  config | mis-ordered pair rate dec raw | neg read from: visual speech nospeech | visual-only fixes, speech-only fixes, neither | neg adjacent to pos | neg mentions group, pos mentions")
        for cfg in CONFIGS:
            p = report[ds][cfg]["pairs_pooled"]
            L.append(f"   {cfg:9s}| {p['mis_dec_rate']:.3f} {p['mis_raw_rate']:.3f} | {p['neg_from_visual_share_of_mis']:.2f} {p['neg_from_speech_share_of_mis']:.2f} {p['neg_nospeech_share_of_mis']:.2f} | {p['visual_would_fix_share_of_mis']:.2f} {p['speech_would_fix_share_of_mis']:.2f} {p['neither_fixes_share_of_mis']:.2f} | {p['neg_adjacent_pos_share_of_mis']:.2f} | {p['neg_mention_share_of_mis']:.2f} {p['pos_mention_share_of_mis']:.2f}")
        for cfg in ("adjacent",):
            s = report[ds][cfg]; ns = s["nospeech"]; c = s["concentration"]; ch = s["channel_per_video"]
            L.append(f"C  {cfg}: no-speech windows {ns['share_eligible_videos']:.2f} of eligible-video windows ({ns['share_all_videos']:.2f} all), positive rate nospeech {f(ns['pos_rate_nospeech'])} vs speech {ns['pos_rate_speech']:.2f}; rank pct (0 top) nospeech pos {f(ns['mean_rank_pct_nospeech_pos'])} neg {f(ns['mean_rank_pct_nospeech_neg'])}, speech pos {f(ns['mean_rank_pct_speech_pos'])} neg {f(ns['mean_rank_pct_speech_neg'])}; videos with a no-speech window {ns['videos_with_nospeech_window']:.2f}")
            L.append(f"   per-video channel: visual better than speech {ch['visual_better_than_speech']:.2f}; decoded below visual {ch['dec_below_visual']:.2f}, below speech {ch['dec_below_speech']:.2f}, below both {ch['dec_below_both']:.2f}, above both {ch['dec_above_both']:.2f} (n {ch['n']})")
            L.append(f"D  {cfg}: within quantiles 10/25/50/75/90 {' '.join(f'{x:.2f}' for x in c['quantiles_10_25_50_75_90'])}; below .5 {c['share_below_.5']:.2f}, below .6 {c['share_below_.6']:.2f}, above .9 {c['share_above_.9']:.2f}; deficit share worst 10% {c['deficit_share_worst_10pct']:.2f}, worst 20% {c['deficit_share_worst_20pct']:.2f}; within if worst 10% at .5 {c['within_if_worst_10pct_at_.5']:.3f}, at 1 {c['within_if_worst_10pct_at_1']:.3f}")
            for key in ("by_hate_share", "by_gold_runs", "by_nospeech_share"):
                L.append(f"   {key}: " + "; ".join(f"{k} n{v['n']} within {f(v['within'])} vis {f(v['visual'])} sp {f(v['speech'])}" for k, v in s[key].items()))
        fv = report[ds]["frames_vs_r6"]
        L.append(f"E  frames vs r6 (speech identical {fv['speech_reads_identical']}): within dec {fv['delta_within_frame_dec']:+.4f}, raw {fv['delta_within_frame_raw']:+.4f}, win visual {fv['delta_win_auc_visual']:+.4f}, win max {fv['delta_win_auc_max']:+.4f}; improved>.02 {fv['videos_improved_gt_.02']:.2f} worsened>.02 {fv['videos_worsened_gt_.02']:.2f}; corr {fv['corr_per_video_within']:.2f}; inverted r6 {fv['inverted_r6']} adjacent {fv['inverted_adjacent']} both {fv['inverted_both']}; by r6 tercile " + ", ".join(f"{k} n{v['n']} {v['delta']:+.3f}" for k, v in fv['delta_by_r6_within_tercile'].items()))
        L.append("   worst (adjacent): video within(dec) raw vis sp | r6 | hate share, dur s, windows, runs, nospeech, z_video, stance")
        for r in report[ds]["worst_adjacent"]:
            L.append(f"   {r['video']:22s} {f(r['within_frame_dec'])} {f(r['within_frame_raw'])} {f(r['win_auc_visual'])} {f(r['win_auc_speech'])} | {f(r['r6_within'])} | {r['hate_share']:.2f} {r['duration_s']:.0f} {r['n_windows']} {r['n_gold_runs']} {r['nospeech_share']:.2f} {r['z_video']:+.1f} {r['stance']}")
    (OUT / "table.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
