#!/usr/bin/env python3
"""Per-video paired bootstrap of within-video ROC-AUC between arms (evaluation only: reads GT). Also checks that the
per-video mean equals the evaluator's within number."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "runs/20260926_twolevel"
ARMS = ["current", "m2", "full", "full_pooled", "abl_nocoupling", "abl_sharedchain", "abl_noleak", "abl_noatleast",
        "abl_vverdict", "abl_vreads"]
PAIRS = [("m2", "current"), ("full", "current"), ("full_pooled", "full"), ("abl_nocoupling", "full"),
         ("abl_sharedchain", "full"), ("abl_noleak", "full"), ("abl_noatleast", "full"), ("abl_vverdict", "full"),
         ("abl_vreads", "full")]


def per_video(arm, ds, Y):
    out = {}
    for l in open(R / arm / "predictions.jsonl"):
        r = json.loads(l)
        if r["dataset"] != ds or r["video_id"] not in Y:
            continue
        y = Y[r["video_id"]]; s = np.asarray(r["score_curve"], float)[:len(y)]
        if len(s) == len(y) and y.min() != y.max():
            out[r["video_id"]] = roc_auc_score(y, s)
    return out


def main():
    rng = np.random.default_rng(0)
    lines, summary = [], {}
    for ds in ["HateMM", "HateClipSeg"]:
        g = np.load(ROOT / f"data/gt_4fps/{ds}.npz", allow_pickle=True)
        Y = {str(v): np.asarray(y, int) for v, y in zip(g["video_ids"], g["y4"])}
        pv = {a: per_video(a, ds, Y) for a in ARMS if (R / a / "predictions.jsonl").exists()}
        lines.append(f"== {ds}")
        for a, d in pv.items():
            m = json.load(open(R / a / "metrics.json"))
            ev = [p for p in m["per_dataset"] if p["dataset"] == ds][0]
            lines.append(f"  {a:16s} pooled {ev['frame_ROC_AUC']:.4f} / {ev['frame_PR_AUC']:.4f}  within {ev['within_video_macro_ROC_AUC']:.4f} "
                         f"(per-video mean {np.mean(list(d.values())):.4f}, n {len(d)})  F1@.3/.5/.7 "
                         f"{ev['interval_F1@0.3']:.3f} / {ev['interval_F1@0.5']:.3f} / {ev['interval_F1@0.7']:.3f}")
        for a, b in PAIRS:
            if a not in pv or b not in pv:
                continue
            vids = sorted(set(pv[a]) & set(pv[b]))
            d = np.array([pv[a][v] - pv[b][v] for v in vids])
            bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)])
            lo, hi = np.quantile(bs, [.025, .975])
            summary[f"{ds}:{a}-{b}"] = [float(d.mean()), float(lo), float(hi)]
            lines.append(f"  within {a} - {b}: {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")
    (R / "analysis").mkdir(parents=True, exist_ok=True)
    (R / "analysis" / "table.txt").write_text("\n".join(lines) + "\n")
    (R / "analysis" / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
