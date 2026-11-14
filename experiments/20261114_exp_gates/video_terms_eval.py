#!/usr/bin/env python3
"""Video-level terms and Integrator variants for README §6 (G3, `taxonomy`) and §10 (H3, `ordinal`). Decoder = current
m2 (centred rank from runs/20261101_paper_redecode). Labels are read only for the metrics.

- Video-level AUC: video positive = any positive 4 fps GT frame; sklearn roc_auc_score on the term.
- Spearman: term against the video's share of positive frames (all test videos with a prediction).
- Pooled ROC / PR of frame score K + lam * crank with the canonical evaluator functions (src/eval/evaluate.py).
  `sd` = each term divided by its s.d. over the corpus's videos (rank weight 1); `raw` = rank weight 8.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_table as T  # noqa: E402

TAX = "runs/20261114_exp_gates/scope_reads/tax_full_hcs/reads.jsonl"


def video_table(G, vals, vids):
    yv = np.array([G[v].max() for v in vids]); frac = np.array([G[v].mean() for v in vids])
    out = {}
    for k, x in vals.items():
        x = np.array([x[v] for v in vids], float)
        out[k] = (roc_auc_score(yv, x), spearmanr(x, frac).correlation)
    return out


def pooled_for(G, rank, K, lam):
    s = {v: K[v] + lam * rank[v] for v in K}
    p = T.pooled(G, s)
    return p["frame_ROC_AUC"], p["frame_PR_AUC"]


def base(store, ds):
    G = store.gt[ds]
    rank = store.rank("full", "", ds)
    terms = store.terms("full", "full", ds)
    vids = sorted(set(rank) & set(terms) & set(G))
    zv = {v: terms[v][0] for v in vids}; mw = {v: terms[v][1] for v in vids}; zt = {v: terms[v][3] for v in vids}
    return G, rank, vids, zv, mw, zt


def taxonomy(out):
    store = T.Store(["m2"], "prob")
    ds = "HateClipSeg"
    G, rank, vids, zv, mw, zt = base(store, ds)
    tx = {v: r["z_target"] for v, r in T.by_video(TAX, ds).items()}
    vids = [v for v in vids if v in tx]
    L = [f"### HateClipSeg, full method, decoder m2 ({len(vids)} videos)", "",
         "| term | video AUC | Spearman vs positive-frame share |", "|---|---|---|"]
    for k, (a, r) in video_table(G, {"mw": mw, "zv": zv, "z_target (protected)": zt, "z_tax": tx}, vids).items():
        L.append(f"| {k} | {a:.3f} | {r:+.3f} |")
    L += ["", "| K (raw, rank weight 8) | pooled ROC | pooled PR |", "|---|---|---|"]
    for name, K in (("mw", {v: mw[v] for v in vids}), ("mw + .75 z_tax", {v: mw[v] + .75 * tx[v] for v in vids}),
                    (".75 z_tax", {v: .75 * tx[v] for v in vids}),
                    ("mw + .5 zv + .5 z_tax", {v: mw[v] + .5 * zv[v] + .5 * tx[v] for v in vids}),
                    ("reference: mw + .75 z_target (ii)", {v: mw[v] + .75 * zt[v] for v in vids})):
        r, p = pooled_for(G, rank, K, 8.0)
        L.append(f"| {name} | {100 * r:.1f} | {100 * p:.1f} |")
    out.write_text("\n".join(L) + "\n"); print("\n".join(L))


def ordinal(out):
    store = T.Store(["m2"], "prob")
    L = []
    for ds in T.DATASETS:
        G, rank, vids, zv, mw, zt = base(store, ds)
        R = T.by_video(T.ORD[T.PART[ds]], ds)
        vids = [v for v in vids if v in R]
        ot = {v: T.ordinal_terms(R[v]["m_ord"]) for v in vids}
        zo = {v: ot[v][0] for v in vids}; sr = {v: ot[v][1] for v in vids}; zm = {v: ot[v][2] for v in vids}
        viol = np.mean([np.any(np.diff(1 / (1 + np.exp(-np.asarray(R[v]["m_ord"])))) > 1e-9) for v in vids])
        chk = max(abs(R[v]["z_target"] - zt[v]) for v in vids)
        L += [f"### {ds} ({len(vids)} videos; raw p_tau out of order in {100 * viol:.1f} % of videos; "
              f"z_target vs stored protected read max |diff| {chk:.1e})", "",
              "| term | video AUC | Spearman vs positive-frame share |", "|---|---|---|"]
        for k, (a, r) in video_table(G, {"zv": zv, "mw": mw, "z_target": zt, "z_ord": zo, "logit(s_raw)": sr,
                                         "z_mean": zm}, vids).items():
            L.append(f"| {k} | {a:.3f} | {r:+.3f} |")
        sd = lambda x: np.std([x[v] for v in vids])
        n = lambda x: {v: x[v] / sd(x) for v in vids}
        nmw, nzo, nzt, nzv = n(mw), n(zo), n(zt), n(zv)
        rows = (("O1 sd: mw + z_ord", {v: nmw[v] + nzo[v] for v in vids}, 1.0),
                ("O2 sd: z_ord", {v: nzo[v] for v in vids}, 1.0),
                ("O3 sd: mw + z_ord + z_target", {v: nmw[v] + nzo[v] + nzt[v] for v in vids}, 1.0),
                ("O4 raw: mw + .75 z_ord", {v: mw[v] + .75 * zo[v] for v in vids}, 8.0),
                ("O5 sd: mw + zv + z_ord", {v: nmw[v] + nzv[v] + nzo[v] for v in vids}, 1.0),
                ("reference: current (raw mw + zv, rank 1)", {v: mw[v] + zv[v] for v in vids}, 1.0))
        L += ["", "| Integrator | pooled ROC | pooled PR |", "|---|---|---|"]
        for name, K, lam in rows:
            r, p = pooled_for(G, rank, K, lam)
            L.append(f"| {name} | {100 * r:.1f} | {100 * p:.1f} |")
        L.append("")
    out.write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    {"taxonomy": taxonomy, "ordinal": ordinal}[sys.argv[1]](Path(sys.argv[2]))
