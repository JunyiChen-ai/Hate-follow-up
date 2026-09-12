#!/usr/bin/env python3
"""Does the fixed 8-second grid ask the model about windows it has no evidence for?

SPVL-r2 puts 20 uniformly sampled frames in the shared prefix and then asks, per 8-second window,
"look only at the frames whose timestamps fall inside this window". For a 240-second video that is one
frame per 12 seconds against an 8-second grid, so a large share of windows contain no frame at all and
the visual branch has nothing local to look at.

This script measures, on the frozen SPVL-r2 run and reading GT only to score (rule 10):
  1. how many windows contain at least one prefix frame;
  2. within-video window AUC of the composed read, of the visual branch and of the speech branch,
     computed separately on framed and frameless windows;
  3. whether the visual branch still orders frameless windows above chance -- if it does, it is answering
     from the global context, not from the window;
  4. the within-video spread of each branch on framed vs frameless windows.

The point of (2) is that the .758/.621 ceiling was measured over ALL windows. If it splits, the ceiling
is a property of this decision grid rather than of the model.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

TS = re.compile(r"_t(\d+(?:\.\d+)?)\.jpg$")


def frame_times(ds, vid):
    d = ROOT / "data/frames_k20" / ds / vid
    if not d.is_dir():
        return None
    out = []
    for p in d.iterdir():
        m = TS.search(p.name)
        if m:
            out.append(float(m.group(1)))
    return sorted(out)


def mean_auc(pairs):
    """pairs: list of (labels, scores) per video; keep videos where the subset has both classes."""
    vals = [roc_auc_score(y, s) for y, s in pairs
            if len(y) > 1 and len(set(y)) == 2 and np.ptp(s) > 0]
    return (float(np.mean(vals)), len(vals)) if vals else (float("nan"), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "runs/20260910_spvl/mllm/q3vl-8b/full/predictions.jsonl"))
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"])
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.run) if l.strip()]
    for ds in a.datasets:
        g = np.load(ROOT / f"data/gt_4fps/{ds}.npz", allow_pickle=True)
        gt = {str(v): np.asarray(y, float) for v, y in zip(g["video_ids"], g["y4"])}

        n_win = n_framed = 0
        all_p, fr_p, nf_p = [], [], []                      # composed read z
        all_v, fr_v, nf_v = [], [], []                      # visual branch
        fr_s, nf_s = [], []                                 # speech branch
        std_fr, std_nf, vis_std_fr, vis_std_nf = [], [], [], []
        vis_wins_framed = vis_wins_frameless = spk_n_fr = spk_n_nf = 0
        pair_fr, pair_nf, sp_vs = [], [], []   # paired: same video, both subsets have both classes

        for r in rows:
            if r["dataset"] != ds or r.get("error") or not r.get("extra"):
                continue
            W = r["extra"].get("windows") or []
            y = gt.get(r["video_id"])
            ft = frame_times(ds, r["video_id"])
            if y is None or ft is None or len(W) < 2:
                continue
            lab, z, zv, zs, has = [], [], [], [], []
            for w in W:
                i0, i1 = int(round(w["start"] * 4)), min(int(round(w["end"] * 4)), len(y))
                seg = y[i0:i1]
                if not seg.size:
                    continue
                lab.append(1 if seg.mean() > 0.5 else 0)
                z.append(float(w["z"]))
                zv.append(float(w["z_visual"]) if w.get("z_visual") is not None else np.nan)
                zs.append(float(w["z_speech"]) if w.get("z_speech") is not None else np.nan)
                has.append(any(w["start"] <= t < w["end"] for t in ft))
            lab, z, zv, zs = map(np.asarray, (lab, z, zv, zs))
            has = np.asarray(has, bool)
            if lab.size < 2 or lab.min() == lab.max():
                continue
            n_win += lab.size
            n_framed += int(has.sum())

            all_p.append((lab, z)); all_v.append((lab, np.nan_to_num(zv, nan=-99.0)))
            mv = ~np.isnan(zv) & ~np.isnan(zs)
            if mv.sum() > 2 and np.ptp(zv[mv]) > 0 and np.ptp(zs[mv]) > 0:
                sp_vs.append(float(spearmanr(zv[mv], zs[mv]).correlation))
            ka, kb = has & ~np.isnan(zv), (~has) & ~np.isnan(zv)
            if (ka.sum() > 1 and kb.sum() > 1 and len(set(lab[ka])) == 2 and len(set(lab[kb])) == 2
                    and np.ptp(zv[ka]) > 0 and np.ptp(zv[kb]) > 0):
                pair_fr.append(roc_auc_score(lab[ka], zv[ka])); pair_nf.append(roc_auc_score(lab[kb], zv[kb]))
            if has.any():
                fr_p.append((lab[has], z[has]))
                m = has & ~np.isnan(zv)
                if m.any():
                    fr_v.append((lab[m], zv[m])); vis_std_fr.append(float(np.std(zv[m])))
                m = has & ~np.isnan(zs)
                if m.any():
                    fr_s.append((lab[m], zs[m])); spk_n_fr += int(m.sum())
                std_fr.append(float(np.std(z[has])))
                vis_wins_framed += int(np.sum(np.isnan(zs[has]) | (zv[has] >= np.nan_to_num(zs[has], nan=-99))))
            if (~has).any():
                nf_p.append((lab[~has], z[~has]))
                m = (~has) & ~np.isnan(zv)
                if m.any():
                    nf_v.append((lab[m], zv[m])); vis_std_nf.append(float(np.std(zv[m])))
                m = (~has) & ~np.isnan(zs)
                if m.any():
                    nf_s.append((lab[m], zs[m])); spk_n_nf += int(m.sum())
                std_nf.append(float(np.std(z[~has])))
                vis_wins_frameless += int(np.sum(np.isnan(zs[~has]) | (zv[~has] >= np.nan_to_num(zs[~has], nan=-99))))

        print(f"\n=== {ds}: {len(all_p)} videos with both window classes, {n_win} windows")
        print(f"  windows containing >=1 prefix frame: {n_framed}/{n_win} = {n_framed / max(n_win,1):.1%}")
        for name, pairs in (("composed z   all windows", all_p), ("composed z   framed only", fr_p),
                            ("composed z   frameless only", nf_p), ("visual branch  framed only", fr_v),
                            ("visual branch  frameless only", nf_v), ("speech branch  framed only", fr_s),
                            ("speech branch  frameless only", nf_s)):
            v, n = mean_auc(pairs)
            print(f"  within AUC  {name:32s} = {v:.4f}   ({n} videos)")
        print(f"  within-video std of composed z:  framed {np.mean(std_fr):.2f}   frameless {np.mean(std_nf):.2f}")
        print(f"  within-video std of visual read: framed {np.mean(vis_std_fr):.2f}   frameless {np.mean(vis_std_nf):.2f}")
        print(f"  visual branch wins the max on:   framed {vis_wins_framed}   frameless {vis_wins_frameless} windows")
        print(f"  windows carrying a speech branch: framed {spk_n_fr}   frameless {spk_n_nf}")
        print(f"  PAIRED, same {len(pair_fr)} videos (both subsets have both classes):")
        print(f"    visual branch, windows WITH a frame    = {np.mean(pair_fr):.4f}")
        print(f"    visual branch, windows WITHOUT a frame = {np.mean(pair_nf):.4f}"
              f"   ({np.mean(pair_nf) - np.mean(pair_fr):+.4f})")
        print(f"  median within-video Spearman(visual read, speech read) = {np.median(sp_vs):+.3f}")


if __name__ == "__main__":
    main()
