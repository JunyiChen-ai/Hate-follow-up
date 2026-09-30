#!/usr/bin/env python3
"""README §15 diagnostic (test read, rule 10): per hateful video, is the OR of both chains or the speech chain alone
the better within-video ranking, and does the label-free gate (the model's frames-only verdict) predict which?
Inputs: predictions of the OR arm and of the speech-only arm (same time level), the gated arm (for its gate values),
GT. Reports: mean within of OR / speech-only / gated / per-video oracle; share of hateful videos where speech-only is
better; ROC of (1 - gate_visual) for 'speech-only is better'; the within reached by the hard choice gate < .5."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]


def load(d):
    return {(r["dataset"], r["video_id"]): r for r in map(json.loads, open(Path(d) / "predictions.jsonl")) if not r.get("error")}


def within(r, y):
    s = np.asarray(r["score_curve"], float); n = min(len(y), len(s))
    return roc_auc_score(y[:n], s[:n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--or-arm", required=True)
    ap.add_argument("--speech-arm", required=True)
    ap.add_argument("--gated-arm", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    A, B, G = load(a.or_arm), load(a.speech_arm), load(a.gated_arm)
    lines = []
    for ds in a.datasets:
        g = np.load(ROOT / f"data/gt_4fps/{ds}.npz", allow_pickle=True)
        Y = {str(v): np.asarray(y, int) for v, y in zip(g["video_ids"], g["y4"])}
        wa, wb, wg, gv = [], [], [], []
        for v in sorted(Y):
            k = (ds, v); y = Y[v]
            if k not in A or k not in B or k not in G or y.min() == y.max():
                continue
            wa.append(within(A[k], y)); wb.append(within(B[k], y)); wg.append(within(G[k], y))
            gv.append(G[k]["extra"].get("gate", {}).get("z_visual", 1.0))
        wa, wb, wg, gv = map(np.array, (wa, wb, wg, gv))
        better = (wb > wa + 1e-9).astype(int)
        hard = np.where(gv < .5, wb, wa)
        lines.append(f"== {ds}: {len(wa)} videos with both labels")
        lines.append(f"  within: OR {wa.mean():.4f}  speech-only {wb.mean():.4f}  gated {wg.mean():.4f}  "
                     f"oracle(OR|speech) {np.maximum(wa, wb).mean():.4f}  hard gate<.5 {hard.mean():.4f}")
        lines.append(f"  speech-only better on {better.mean():.3f} of videos; gate_visual median {np.median(gv):.3f}, "
                     f"< .5 on {np.mean(gv < .5):.3f}; ROC of (1 - gate_visual) for 'speech-only better': "
                     f"{roc_auc_score(better, 1 - gv) if 0 < better.sum() < len(better) else float('nan'):.3f}")
        for lo, hi in ((0, .25), (.25, .5), (.5, .75), (.75, 1.01)):
            m = (gv >= lo) & (gv < hi)
            if m.sum():
                lines.append(f"    gate in [{lo:.2f}, {min(hi, 1):.2f}): {m.sum():3d} videos  OR {wa[m].mean():.3f}  "
                             f"speech-only {wb[m].mean():.3f}  gated {wg[m].mean():.3f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
