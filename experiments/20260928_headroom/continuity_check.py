#!/usr/bin/env python3
"""Pre-check of the discourse-continuity time level proposed in the Codex consultation (README.md §"Codex
consultation"). Reads the gold: analysis only, never part of a method.
1. At the boundary between two neighbouring 8 s windows, does speech continuity predict that both windows carry the
   same gold label (hate share >= .5), beyond what the window reads already say? Continuity: a Whisper segment runs
   across the boundary; the cosine of the two window transcripts under a frozen sentence encoder (all-MiniLM-L6-v2).
   Label change is predicted by logistic regression, 5-fold CV by video, from the reads only, the reads plus
   continuity, and continuity only.
2. Resolution limit of the reading grid: within if every 8 s window (or 4 s cell) carried its true hate share.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from headroom import FPS, ROOT, boot_diff, gold, load, per_video_within

sys.path.insert(0, str(ROOT))
from src.video_inputs import load_asr, window_text  # noqa: E402

FILL = -12.0   # a missing read (no speech or no frame in the window); the reading run fills uncovered frames with it


def share(y, s, e):
    a, b = int(round(s * FPS)), min(int(round(e * FPS)), len(y))
    return float(y[a:b].mean()) if b > a else np.nan


def boundary_rows(ds, reads, Y):
    segs, rows = load_asr(ds), []
    for v, r in sorted(reads.items()):
        if v not in Y:
            continue
        ws = sorted(r["extra"]["windows"], key=lambda w: w["start"])
        lab = np.array([share(Y[v], w["start"], w["end"]) for w in ws])
        b = lab[~np.isnan(lab)] >= .5
        if b.all() or not b.any():
            continue                                   # no label change is possible inside this video
        sg = segs.get(v, [])
        for i in range(len(ws) - 1):
            wi, wj = ws[i], ws[i + 1]
            if np.isnan(lab[i]) or np.isnan(lab[i + 1]):
                continue
            t = wi["end"]
            rd = [wi.get(m, FILL) for m in ("z_speech", "z_visual")] + [wj.get(m, FILL) for m in ("z_speech", "z_visual")]
            rows.append({"video": v, "change": int((lab[i] >= .5) != (lab[i + 1] >= .5)),
                         "reads": rd + [abs(rd[0] - rd[2]), abs(rd[1] - rd[3])]
                                  + [int(m not in w) for w in (wi, wj) for m in ("z_speech", "z_visual")],
                         "spans": int(any(s < t - .5 and e > t + .5 for s, e, _ in sg)),
                         "text": (window_text(sg, wi["start"], wi["end"]).strip(),
                                  window_text(sg, wj["start"], wj["end"]).strip())})
    return rows


def add_similarity(rows, enc):
    texts = sorted({x for r in rows for x in r["text"] if x})
    emb = dict(zip(texts, enc.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)))
    for r in rows:
        a, b = r["text"]
        r["both_speech"] = int(bool(a and b))
        r["sim"] = float(emb[a] @ emb[b]) if a and b else 0.0
        r["cont"] = [r["spans"], r["sim"], r["both_speech"], int(bool(a)), int(bool(b))]


def cv_scores(X, y, groups):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def boot_auc_diff(y, pa, pb, groups, n=1000, seed=0):
    idx = {}
    for k, g in enumerate(groups):
        idx.setdefault(g, []).append(k)
    vs, rng, d = list(idx), np.random.default_rng(seed), []
    for _ in range(n):
        s = np.concatenate([idx[vs[j]] for j in rng.integers(0, len(vs), len(vs))])
        if 0 < y[s].sum() < len(s):
            d.append(roc_auc_score(y[s], pa[s]) - roc_auc_score(y[s], pb[s]))
    return float(roc_auc_score(y, pa) - roc_auc_score(y, pb)), float(np.quantile(d, .025)), float(np.quantile(d, .975))


def oracle_within(reads, Y, cell):
    """Frame score = true hate share of its reading window (cell None) or of its fixed cell of `cell` seconds."""
    S = {}
    for v, y in Y.items():
        if cell is None:
            if v not in reads:
                continue
            ws = sorted(reads[v]["extra"]["windows"], key=lambda w: w["start"])
            edges = [(w["start"], w["end"]) for w in ws]
        else:
            edges = [(s, s + cell) for s in np.arange(0, len(y) / FPS, cell)]
        s = np.full(len(y), np.nan)
        for a, b in edges:
            s[int(round(a * FPS)):min(int(round(b * FPS)), len(y))] = share(y, a, b)
        last = s[~np.isnan(s)][-1] if (~np.isnan(s)).any() else 0.0
        S[v] = np.where(np.isnan(s), last, s)
    return per_video_within(Y, S, sorted(S))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(ROOT / "runs/20260926_twolevel/final_m2/predictions.jsonl"))
    ap.add_argument("--reads", default=str(ROOT / "runs/20260926_glr/base_gridA/predictions.jsonl"))
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out", default=str(ROOT / "runs/20260928_headroom/continuity"))
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(a.encoder, device="cpu")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lines, summary = [], {}
    for ds in ("HateMM", "HateClipSeg"):
        Y, reads, pred = gold(ds), load(a.reads, ds), load(a.pred, ds)
        rows = boundary_rows(ds, reads, Y); add_similarity(rows, enc)
        y, g = np.array([r["change"] for r in rows]), [r["video"] for r in rows]
        Xr, Xc = np.array([r["reads"] for r in rows], float), np.array([r["cont"] for r in rows], float)
        sp, both, sim = Xc[:, 0] == 1, Xc[:, 2] == 1, Xc[:, 1]
        q = np.quantile(sim[both], [.25, .5, .75]) if both.any() else [0, 0, 0]
        rate = lambda m: f"{y[m].mean():.3f} (n {int(m.sum())})"
        lines.append(f"== {ds}: {len(rows)} boundaries in {len(set(g))} videos with both labels; change rate {y.mean():.3f}")
        lines.append(f"  change rate: Whisper segment runs across the boundary {rate(sp)}; speech on both sides, "
                     f"no segment across {rate(both & ~sp)}; speech missing on a side {rate(~both)}")
        bins = np.digitize(sim, q)
        lines.append("  change rate by transcript cosine quartile (speech on both sides): "
                     + ", ".join(f"Q{k + 1} {rate(both & (bins == k))}" for k in range(4)))
        p0, p1, p2 = cv_scores(Xr, y, g), cv_scores(np.hstack([Xr, Xc]), y, g), cv_scores(Xc, y, g)
        d = boot_auc_diff(y, p1, p0, g)
        lines.append(f"  predict a label change (AUC, 5-fold CV by video): reads {roc_auc_score(y, p0):.4f}; "
                     f"reads + continuity {roc_auc_score(y, p1):.4f}; continuity only {roc_auc_score(y, p2):.4f}; "
                     f"added by continuity {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]")
        cur = per_video_within(Y, {v: np.asarray(pred[v]["score_curve"], float)[:len(Y[v])] for v in pred if v in Y},
                               sorted(v for v in pred if v in Y))
        o8, o4 = oracle_within(reads, Y, None), oracle_within(reads, Y, 4.0)
        k5 = {v for v in cur if Y[v].mean() < .25}
        m = lambda w, vs: float(np.mean([w[v] for v in vs if v in w]))
        lines.append(f"  within: method {m(cur, cur):.4f}; true share per 8 s reading window {m(o8, cur):.4f} "
                     f"(vs method {boot_diff({v: o8[v] for v in cur if v in o8}, cur)[0]:+.4f}); "
                     f"true share per 4 s cell {m(o4, cur):.4f}; n {len(cur)}")
        lines.append(f"  within, hate covers < 25 %: method {m(cur, k5):.4f}; 8 s windows {m(o8, k5):.4f}; "
                     f"4 s cells {m(o4, k5):.4f}; n {len(k5)}")
        summary[ds] = {"n_boundaries": len(rows), "n_videos": len(set(g)), "change_rate": float(y.mean()),
                       "rate_spans": float(y[sp].mean()) if sp.any() else None,
                       "rate_both_no_span": float(y[both & ~sp].mean()) if (both & ~sp).any() else None,
                       "rate_missing_side": float(y[~both].mean()) if (~both).any() else None,
                       "rate_by_sim_quartile": [float(y[both & (bins == k)].mean()) if (both & (bins == k)).any()
                                                else None for k in range(4)],
                       "auc_reads": float(roc_auc_score(y, p0)), "auc_reads_cont": float(roc_auc_score(y, p1)),
                       "auc_cont": float(roc_auc_score(y, p2)), "auc_added": d,
                       "within_method": m(cur, cur), "within_oracle_8s": m(o8, cur), "within_oracle_4s": m(o4, cur),
                       "k5": {"n": len(k5), "method": m(cur, k5), "oracle_8s": m(o8, k5), "oracle_4s": m(o4, k5)}}
    (out / "table.txt").write_text("\n".join(lines) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
