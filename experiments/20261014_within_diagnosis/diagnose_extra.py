#!/usr/bin/env python3
"""Second stage of the within-video diagnosis (README.md §3): the follow-up checks run after `diagnose.py`.
Reads test labels (rule 10); analysis only. Output `runs/20261014_within_diagnosis/extra.json` and `extra.txt`.

F. Base rates of the deciding channel (which read supplies the window's max) among all / positive / clean negative
   windows, so that the shares among mis-ordered pairs in `summary.json` can be read against them; read scales.
G. Decoder effect by the video verdict (stance, z_video).
H. Boundary text spill: clean negative windows next to a positive window, split by whether a Whisper segment spans
   both windows; their mis-order rate against the video's positive windows; Whisper segment lengths.
I. Position-only curves (middle-high, later-high) by hate share; where the negative frames of mostly-hateful videos
   sit (first / last 10 % of the video).
J. The frames effect by the quality of the speech ordering (terciles of the r6 speech window AUC).
K. Visual window ordering of the frames-only-prefix reads (`runs/20260928_infer/main/reads_noctx`, no transcript
   in the prefix) against r6 and the frames configuration.
L. Fixed-decoder variants (speech chain only, visual chain only, independent cells) of r6 and the frames reads:
   metrics read from `runs/20261014_within_diagnosis/decoded/<tag>/metrics.json` (produced by twolevel_r2.py with
   the r6_bma flags plus `--chains` / `--nocoupling`; commands in README §2).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.video_inputs import load_asr  # noqa: E402

OUT = ROOT / "runs/20261014_within_diagnosis"
DATASETS = ("HateMM", "HateClipSeg")
RAW = {"r6": ROOT / "runs/20260926_glr/base_gridA",
       "adjacent": ROOT / "runs/20261007_m1_streamingtom/placement_controls_main/adjacent_local",
       "E": ROOT / "runs/20261007_m1_streamingtom/local_controls_main/local_clean",
       "noctx": ROOT / "runs/20260928_infer/main/reads_noctx"}
DEC = {"r6": ROOT / "runs/20260926_twolevel/r6_bma",
       "adjacent": ROOT / "runs/20261007_m1_streamingtom/placement_controls_main_decoded/adjacent_local"}
DECODED_VARIANTS = ("r6_z_speech", "r6_z_visual", "r6_nocoupling", "adjacent_z_speech", "adjacent_z_visual", "adjacent_nocoupling")


def records(path, ds):
    return {r["video_id"]: r for r in map(json.loads, open(path / "predictions.jsonl")) if r["dataset"] == ds and not r.get("error")}


def gold(ds):
    g = np.load(ROOT / f"data/gt_4fps/{ds}.npz", allow_pickle=True)
    return {str(v): np.asarray(y, int) for v, y, s in zip(g["video_ids"], g["y4"], g["split"]) if str(s) == "test"}


def frames_of(w, n):
    return np.arange(int(round(w["start"] * 4)), min(n, int(round(w["end"] * 4))))


def auc(y, s):
    return float(roc_auc_score(y, s)) if 0 < sum(y) < len(y) else None


def mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else None


def windows(R, D, Y, v):
    y = Y[v]; n = len(y); curve = np.asarray(D[v]["score_curve"], float)[:n] if D is not None else None
    rows = []
    for w in R[v]["extra"]["windows"]:
        idx = frames_of(w, n)
        if len(idx) == 0 or "z_visual" not in w:
            continue
        zs = w.get("z_speech")
        rows.append({"i": w["i"], "start": w["start"], "end": w["end"], "share": float(y[idx].mean()), "zv": float(w["z_visual"]),
                     "zs": None if zs is None else float(zs), "z": float(w["z"]),
                     "dec": None if curve is None else float(curve[idx].mean()),
                     "deciding": "nospeech" if zs is None else ("speech" if zs >= w["z_visual"] else "visual")})
    return rows


def main():
    pv = json.load(open(OUT / "per_video.json"))
    report = {"scope": "development-selected; reads test labels (rule 10); analysis only"}
    L = []
    for ds in DATASETS:
        Y = gold(ds); asr = load_asr(ds); rep = {}; L.append(f"== {ds}")
        elig = [r["video"] for r in pv[f"{ds}|r6"] if r["eligible"]]
        PV = {cfg: {r["video"]: r for r in pv[f"{ds}|{cfg}"]} for cfg in ("r6", "adjacent", "E")}
        # ---- F, G
        for cfg in ("r6", "adjacent"):
            R = records(RAW[cfg], ds); D = records(DEC[cfg], ds)
            dec = {"all": [], "pos": [], "neg": []}; zv = []; zs = []; rho_v = []; rho_s = []; rho_vs = []; lab = []; pzv = []; pzs = []
            for v in elig:
                rows = windows(R, D, Y, v)
                for r in rows:
                    dec["all"].append(r["deciding"])
                    if r["share"] >= .5:
                        dec["pos"].append(r["deciding"])
                    if r["share"] == 0:
                        dec["neg"].append(r["deciding"])
                    zv.append(r["zv"]); lab.append(int(r["share"] >= .5)); pzv.append(r["zv"]); pzs.append(-30. if r["zs"] is None else r["zs"])
                    if r["zs"] is not None:
                        zs.append(r["zs"])
                sp = [r for r in rows if r["zs"] is not None]
                if len(rows) > 2:
                    rho_v.append(spearmanr([r["dec"] for r in rows], [r["zv"] for r in rows]).correlation)
                    rho_s.append(spearmanr([r["dec"] for r in rows], [-30. if r["zs"] is None else r["zs"] for r in rows]).correlation)
                if len(sp) > 2:
                    rho_vs.append(spearmanr([r["zv"] for r in sp], [r["zs"] for r in sp]).correlation)
            share = lambda k, c: float(np.mean([x == c for x in dec[k]]))
            F = {"deciding_speech": {k: share(k, "speech") for k in dec}, "deciding_visual": {k: share(k, "visual") for k in dec},
                 "deciding_nospeech": {k: share(k, "nospeech") for k in dec},
                 "z_visual_mean": float(np.mean(zv)), "z_visual_sd": float(np.std(zv)), "z_visual_share_pos": float(np.mean(np.array(zv) > 0)),
                 "z_speech_mean": float(np.mean(zs)), "z_speech_sd": float(np.std(zs)), "z_speech_share_pos": float(np.mean(np.array(zs) > 0)),
                 "pooled_window_auc_visual": auc(lab, pzv), "pooled_window_auc_speech": auc(lab, pzs),
                 "spearman_decoded_visual": mean(rho_v), "spearman_decoded_speech": mean(rho_s), "spearman_visual_speech_per_video": mean(rho_vs)}
            E_ = [PV[cfg][v] for v in elig if PV[cfg][v]["within_frame_raw"] is not None]
            G = {}
            for name, sel in (("stance_yes", [r for r in E_ if r["stance"] == "Yes"]), ("stance_no", [r for r in E_ if r["stance"] != "Yes"]),
                              ("z_video_0_10", [r for r in E_ if 0 <= r["z_video"] < 10]), ("z_video_ge_10", [r for r in E_ if r["z_video"] >= 10])):
                if sel:
                    G[name] = {"n": len(sel), "raw": mean([r["within_frame_raw"] for r in sel]), "decoded": mean([r["within_frame_dec"] for r in sel]),
                               "hurt_gt_.02": float(np.mean([r["within_frame_dec"] - r["within_frame_raw"] < -.02 for r in sel]))}
            for lo, hi in ((0, .25), (.25, .5), (.5, .75), (.75, 1.01)):
                sel = [r for r in E_ if lo <= r["hate_share"] < hi]
                G[f"hate_share_{lo}_{hi}"] = {"n": len(sel), "raw": mean([r["within_frame_raw"] for r in sel]), "decoded": mean([r["within_frame_dec"] for r in sel])}
            rep[cfg] = {"F_channels": F, "G_decoder_by_verdict": G}
            L.append(f"F  {cfg:9s} deciding=speech all {F['deciding_speech']['all']:.2f} pos {F['deciding_speech']['pos']:.2f} neg {F['deciding_speech']['neg']:.2f}; z_visual mean {F['z_visual_mean']:+.2f} sd {F['z_visual_sd']:.2f} >0 {F['z_visual_share_pos']:.2f}; z_speech mean {F['z_speech_mean']:+.2f} sd {F['z_speech_sd']:.2f} >0 {F['z_speech_share_pos']:.2f}; pooled window AUC visual {F['pooled_window_auc_visual']:.3f} speech {F['pooled_window_auc_speech']:.3f}; spearman(decoded, visual) {F['spearman_decoded_visual']:.2f} (decoded, speech) {F['spearman_decoded_speech']:.2f}; (visual, speech) {F['spearman_visual_speech_per_video']:.2f}")
            L.append("G  " + cfg + ": " + "; ".join(f"{k} n{v['n']} raw {v['raw']:.3f} dec {v['decoded']:.3f}" for k, v in G.items()))
        # ---- H boundary spill (frames configuration)
        R = records(RAW["adjacent"], ds); D = records(DEC["adjacent"], ds)
        seglen = [e - s for v in Y for s, e, t in asr.get(v, []) if t]
        H = {"whisper_segments": {"n": len(seglen), "mean_s": float(np.mean(seglen)), "median_s": float(np.median(seglen)), "p90_s": float(np.quantile(seglen, .9)), "share_gt_8s": float(np.mean(np.array(seglen) > 8))}}
        cnt = {k: {"windows": 0, "pairs": 0, "mis": 0} for k in ("adjacent_crossing_segment", "adjacent_no_crossing", "not_adjacent")}
        for v in elig:
            rows = windows(R, D, Y, v); pos = [r for r in rows if r["share"] >= .5]; pos_i = {r["i"] for r in pos}; segs = asr.get(v, [])
            for q in rows:
                if q["share"] != 0:
                    continue
                adj = [r for r in rows if r["i"] in (q["i"] - 1, q["i"] + 1) and r["i"] in pos_i]
                if adj:
                    cross = any(min(e, q["end"]) > max(s, q["start"]) and any(min(e, p["end"]) > max(s, p["start"]) for p in adj) for s, e, t in segs if t)
                    k = "adjacent_crossing_segment" if cross else "adjacent_no_crossing"
                else:
                    k = "not_adjacent"
                cnt[k]["windows"] += 1
                for p in pos:
                    cnt[k]["pairs"] += 1; cnt[k]["mis"] += q["dec"] >= p["dec"]
        tot_mis = sum(c["mis"] for c in cnt.values())
        for k, c in cnt.items():
            c["mis_rate"] = c["mis"] / c["pairs"] if c["pairs"] else None; c["share_of_mis_pairs"] = c["mis"] / tot_mis
        H["clean_negatives"] = cnt; rep["H_boundary_spill"] = H
        L.append(f"H  whisper segments mean {H['whisper_segments']['mean_s']:.1f} s median {H['whisper_segments']['median_s']:.1f} p90 {H['whisper_segments']['p90_s']:.1f}, >8 s {H['whisper_segments']['share_gt_8s']:.2f}; " + "; ".join(f"{k} n{c['windows']} mis-rate {c['mis_rate']:.3f} share {c['share_of_mis_pairs']:.2f}" for k, c in cnt.items()))
        # ---- I position-only curves
        I = {}
        for lo, hi in ((0, .25), (.25, .5), (.5, .75), (.75, 1.01), (0, 1.01)):
            sel = [v for v in elig if lo <= PV["adjacent"][v]["hate_share"] < hi]; mid = []; late = []; edge = []
            for v in sel:
                y = Y[v]; n = len(y); t = np.arange(n) / n; k = max(1, int(round(.1 * n))); neg = np.flatnonzero(y == 0)
                mid.append(roc_auc_score(y, -np.abs(t - .5))); late.append(roc_auc_score(y, t)); edge.append(float(np.mean((neg < k) | (neg >= n - k))))
            I[f"hate_share_{lo}_{hi}"] = {"n": len(sel), "decoded_within_adjacent": mean([PV["adjacent"][v]["within_frame_dec"] for v in sel]),
                                          "position_middle_high": mean(mid), "position_later_high": mean(late), "negative_frames_in_first_last_10pct": mean(edge)}
        rep["I_position"] = I
        L.append("I  " + "; ".join(f"{k} n{v['n']} within {v['decoded_within_adjacent']:.3f} middle-high {v['position_middle_high']:.3f} later-high {v['position_later_high']:.3f} edge-negatives {v['negative_frames_in_first_last_10pct']:.2f}" for k, v in I.items()))
        # ---- J frames effect by speech ordering tercile; K noctx visual ordering
        A = {v: PV["r6"][v] for v in elig if PV["r6"][v]["win_auc_speech"] is not None and PV["r6"][v]["win_auc_visual"] is not None}
        q = np.quantile([A[v]["win_auc_speech"] for v in A], [1 / 3, 2 / 3])
        terc = lambda v: "low" if A[v]["win_auc_speech"] < q[0] else "mid" if A[v]["win_auc_speech"] < q[1] else "high"
        J = {}
        for name in ("low", "mid", "high"):
            sel = [v for v in A if terc(v) == name]; B = PV["adjacent"]
            J[name] = {"n": len(sel), "speech_auc": mean([A[v]["win_auc_speech"] for v in sel]), "r6_visual_auc": mean([A[v]["win_auc_visual"] for v in sel]),
                       "r6_within": mean([A[v]["within_frame_dec"] for v in sel]),
                       "delta_visual_auc": mean([B[v]["win_auc_visual"] - A[v]["win_auc_visual"] for v in sel]),
                       "delta_within_raw": mean([B[v]["within_frame_raw"] - A[v]["within_frame_raw"] for v in sel]),
                       "delta_within_decoded": mean([B[v]["within_frame_dec"] - A[v]["within_frame_dec"] for v in sel])}
        rep["J_frames_by_speech_tercile"] = J
        L.append("J  " + "; ".join(f"{k} n{v['n']} speech {v['speech_auc']:.2f} r6 within {v['r6_within']:.3f} d-visual {v['delta_visual_auc']:+.3f} d-raw {v['delta_within_raw']:+.3f} d-decoded {v['delta_within_decoded']:+.3f}" for k, v in J.items()))
        K = {}
        for cfg in ("r6", "adjacent", "noctx"):
            Rc = records(RAW[cfg], ds); vals = {"all": [], "low": [], "mid": [], "high": []}
            for v in A:
                rows = windows(Rc, None, Y, v); a = auc([int(r["share"] >= .5) for r in rows], [r["zv"] for r in rows]) if rows else None
                if a is not None:
                    vals["all"].append(a); vals[terc(v)].append(a)
            K[cfg] = {k: mean(x) for k, x in vals.items()} | {"n": len(vals["all"])}
        rep["K_noctx_visual_ordering"] = K
        L.append("K  visual window AUC macro: " + "; ".join(f"{cfg} all {v['all']:.3f} low {v['low']:.3f} mid {v['mid']:.3f} high {v['high']:.3f}" for cfg, v in K.items()))
        # ---- L fixed-decoder variants
        Lv = {}
        for tag in DECODED_VARIANTS:
            m = json.load(open(OUT / "decoded" / tag / "metrics.json"))["per_dataset"]
            d = next(x for x in m if x["dataset"] == ds)
            Lv[tag] = {k: d[k] for k in ("frame_ROC_AUC", "frame_PR_AUC", "within_video_macro_ROC_AUC", "n_videos_defined")}
        rep["L_decoder_variants"] = Lv
        L.append("L  " + "; ".join(f"{t} within {v['within_video_macro_ROC_AUC']:.4f}" for t, v in Lv.items()))
        report[ds] = rep
    (OUT / "extra.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "extra.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
