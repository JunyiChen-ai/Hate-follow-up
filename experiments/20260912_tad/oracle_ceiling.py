#!/usr/bin/env python3
"""Oracle ceiling: how much within-video ordering is recoverable from the per-window reads at all?

DIAGNOSTIC ONLY (rule 10 test read; logged in the experiment README). A logistic regression is fitted on
the *test* GT with leave-one-video-out grouping and its out-of-fold score is evaluated. This is an upper
bound on what any label-free combination of the same features could reach, not a method: the fitted model
never enters a method arm, a gate, or a run's predictions.

Feature sets, each an upper bound for the corresponding family:
  reads   : the MLLM per-window reads available in the run (a, t, act_margin, and the branch scores)
  shape   : + window position, length, speech presence, transcript length
  audio   : + ImageBind audio embedding of the window (data/omsl_v6_inputs/audio_embeddings), if cached
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.video_inputs import load_asr, window_text  # noqa: E402


def window_labels(y4, wins):
    out = []
    for t1, t2 in wins:
        a, b = int(round(t1 * 4)), min(int(round(t2 * 4)), len(y4))
        seg = y4[a:b]
        out.append(1 if seg.size and float(seg.mean()) > 0.5 else 0)
    return out


def audio_window_vec(emb, t1, t2, dur):
    """Mean of the cached ImageBind audio embedding rows overlapping [t1, t2] (rows are uniform in time)."""
    if emb is None or len(emb) == 0:
        return None
    n = len(emb)
    i = int(np.floor(t1 / dur * n)); j = int(np.ceil(t2 / dur * n))
    i, j = max(0, min(i, n - 1)), max(1, min(j, n))
    return emb[i:j].mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"])
    ap.add_argument("--sets", nargs="+", default=["reads", "shape", "audio", "temporal"])
    ap.add_argument("--model", choices=["logreg", "gbt"], default="logreg")
    ap.add_argument("--audio-dims", type=int, default=32, help="PCA dims kept from the 1024-d audio vector")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(Path(args.run_dir) / "predictions.jsonl") if l.strip()]

    for ds in args.datasets:
        g = np.load(ROOT / f"data/gt_4fps/{ds}.npz", allow_pickle=True)
        gt = {str(v): np.asarray(y, float) for v, y in zip(g["video_ids"], g["y4"])}
        asr = load_asr(ds)
        R = [r for r in rows if r["dataset"] == ds and not r.get("error") and r.get("extra")]
        X, Y, G, base = [], [], [], []
        aud_dir = ROOT / f"data/omsl_v6_inputs/audio_embeddings/{ds}"
        for r in R:
            y = gt.get(r["video_id"])
            if y is None:
                continue
            W = r["extra"]["windows"]
            lab = window_labels(y, [(w["start"], w["end"]) for w in W])
            if min(lab) == max(lab):
                continue
            segs = asr.get(r["video_id"], [])
            emb = None
            p = aud_dir / f"{r['video_id']}.npy"
            if "audio" in args.sets and p.exists():
                try:
                    emb = np.load(p)
                except Exception:  # noqa: BLE001
                    emb = None
            dur = float(r["duration"])
            for k, w in enumerate(W):
                reads = [w.get("a", 0.0), w.get("t", 0.0), w.get("act_margin", 0.0),
                         w.get("z_visual", 0.0), w.get("z_speech", 0.0)]
                txt = window_text(segs, w["start"], w["end"]) or ""
                shape = [k / max(len(W) - 1, 1), w["end"] - w["start"], float(bool(txt.strip())),
                         len(txt.split())]
                av = audio_window_vec(emb, w["start"], w["end"], dur)
                X.append((reads, shape, av, k))
                Y.append(lab[k]); G.append(r["video_id"]); base.append(w.get("a", 0.0))
        Y = np.asarray(Y); G = np.asarray(G); base = np.asarray(base)
        have_audio = sum(1 for x in X if x[2] is not None)
        print(f"\n=== {ds}: {len(Y)} windows, {len(set(G))} videos, audio cached for {have_audio} windows")

        def within(scores):
            out = []
            for v in np.unique(G):
                m = G == v
                if Y[m].min() != Y[m].max():
                    out.append(roc_auc_score(Y[m], scores[m]))
            return float(np.mean(out))

        print(f"  window-level within from the act read alone (no fitting) = {within(base):.4f}")
        for name in args.sets:
            if name == "reads":
                F = np.array([x[0] for x in X], float)
            elif name == "shape":
                F = np.array([x[0] + x[1] for x in X], float)
            elif name == "temporal":  # the reads of the previous and next window as extra features
                base_f = np.array([x[0] + x[1] for x in X], float)
                nb = []
                for i in range(len(X)):
                    prev = base_f[i - 1] if i > 0 and G[i - 1] == G[i] else base_f[i]
                    nxt = base_f[i + 1] if i + 1 < len(X) and G[i + 1] == G[i] else base_f[i]
                    nb.append(np.concatenate([base_f[i], prev[:5], nxt[:5]]))
                F = np.array(nb, float)
            else:
                dim = len(next((x[2] for x in X if x[2] is not None), []))
                if not dim:
                    print("  audio: no embeddings cached, skipped"); continue
                A = np.array([x[2] if x[2] is not None else np.zeros(dim) for x in X], float)
                A = A - A.mean(0)
                U, S, Vt = np.linalg.svd(A, full_matrices=False)
                A = U[:, :args.audio_dims] * S[:args.audio_dims]
                F = np.hstack([np.array([x[0] + x[1] for x in X], float), A])
            F = StandardScaler().fit_transform(F)
            oof = np.zeros(len(Y))
            gkf = GroupKFold(n_splits=min(5, len(set(G))))
            for tr, te in gkf.split(F, Y, groups=G):
                clf = (HistGradientBoostingClassifier(max_iter=200, random_state=0) if args.model == "gbt"
                       else LogisticRegression(max_iter=2000, C=1.0))
                clf.fit(F[tr], Y[tr])
                oof[te] = clf.decision_function(F[te])
            print(f"  ORACLE (out-of-fold, GT-supervised) {name:6s} within = {within(oof):.4f}  "
                  f"[{F.shape[1]} features]")


if __name__ == "__main__":
    main()
