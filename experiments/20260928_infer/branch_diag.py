#!/usr/bin/env python3
"""Why two isolated branches instead of one joint branch (README §13; test read under rule 10).

On the raw reads, no time level: per window (8 s grid A, windows that have both a visual and a speech read), the GT
label is 'more than half of the window's frames hateful'. Reports, per corpus:
- within-video window ROC (videos with both labels), macro over videos, for z_visual, z_speech, max(z_visual,
  z_speech) = the current window score, and the joint read of the same window from a joint-branch run;
- pooled window ROC of the same four;
- among hateful windows: which branch is positive (picture only / speech only / both / neither) and, in each group,
  how often the joint read is positive; the same for non-hateful windows (false-positive side);
- the median within-video Spearman between the two branches.
Reads GT: this is a test read; the README logs it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
FPS = 4.0


def load(d):
    return {(r["dataset"], r["video_id"]): r for r in map(json.loads, open(Path(d) / "predictions.jsonl")) if not r.get("error")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dual", required=True)
    ap.add_argument("--joint", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dual, joint = load(a.dual), load(a.joint)
    lines = [f"dual {a.dual}\njoint {a.joint}"]
    for ds in a.datasets:
        g = np.load(ROOT / f"data/gt_4fps/{ds}.npz", allow_pickle=True)
        Y = {str(v): np.asarray(y, int) for v, y in zip(g["video_ids"], g["y4"])}
        rows = []  # (video, label, zv, zs, zj)
        rho = []
        for v in sorted(Y):
            if (ds, v) not in dual or (ds, v) not in joint:
                continue
            wd, wj = dual[(ds, v)]["extra"]["windows"], joint[(ds, v)]["extra"]["windows"]
            if len(wd) != len(wj):
                continue
            y = Y[v]
            vv, ss = [], []
            for d, j in zip(wd, wj):
                if "z_visual" not in d or "z_speech" not in d or "z_joint" not in j:
                    continue
                i0, i1 = int(np.floor(d["start"] * FPS)), min(len(y), int(np.ceil(d["end"] * FPS)))
                if i1 <= i0:
                    continue
                rows.append((v, int(y[i0:i1].mean() > 0.5), d["z_visual"], d["z_speech"], j["z_joint"]))
                vv.append(d["z_visual"]); ss.append(d["z_speech"])
            if len(vv) >= 3 and np.std(vv) > 0 and np.std(ss) > 0:
                rho.append(spearmanr(vv, ss).correlation)
        R = np.array([r[1:] for r in rows], float); vid = np.array([r[0] for r in rows])
        lab, zv, zs, zj = R[:, 0].astype(int), R[:, 1], R[:, 2], R[:, 3]
        zm = np.maximum(zv, zs)
        scores = {"visual": zv, "speech": zs, "max(visual, speech) [current]": zm, "joint": zj}
        lines.append(f"\n== {ds}: {len(rows)} windows with both branches in {len(set(vid))} videos; hateful windows {lab.sum()}; "
                     f"median within-video Spearman(visual, speech) {np.median(rho):.2f}")
        lines.append("  within-video window ROC (macro over videos with both labels) / pooled window ROC:")
        for k, z in scores.items():
            w = []
            for v in sorted(set(vid)):
                m = vid == v
                if 0 < lab[m].sum() < m.sum():
                    w.append(roc_auc_score(lab[m], z[m]))
            lines.append(f"    {k:32s} within {np.mean(w):.3f} (n={len(w)})   pooled {roc_auc_score(lab, z):.3f}")
        for name, sel in (("hateful", lab == 1), ("non-hateful", lab == 0)):
            groups = {"picture only (zv>0, zs<=0)": (zv > 0) & (zs <= 0), "speech only (zs>0, zv<=0)": (zs > 0) & (zv <= 0),
                      "both > 0": (zv > 0) & (zs > 0), "neither": (zv <= 0) & (zs <= 0)}
            lines.append(f"  {name} windows (n={sel.sum()}): share by branch sign, and share of them where the joint read > 0:")
            for k, m in groups.items():
                mm = m & sel
                lines.append(f"    {k:28s} {mm.sum() / max(sel.sum(), 1):.3f}   joint>0: {(zj[mm] > 0).mean() if mm.sum() else float('nan'):.3f}")
            lines.append(f"    max>0: {(zm[sel] > 0).mean():.3f}   joint>0: {(zj[sel] > 0).mean():.3f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
