#!/usr/bin/env python3
"""Video-level AUC of candidate video terms (README §13, 2026-11-16; CPU, cached reads and em0 decodes; test-read,
development only). Video label = the video has at least one hateful frame in the test GT."""
import sys, json, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import gate_table as G
from sklearn.metrics import roc_auc_score, average_precision_score
lg = lambda p: np.log(np.clip(p,1e-12,1-1e-12)) - np.log1p(-np.clip(p,1e-12,1-1e-12))
st = G.Store(["em0"], "prob", taxonomy=False)
stx = G.Store(["em0"], "prob", taxonomy=True)
for ds in G.DATASETS:
    y = st.gt[ds]
    T = st.terms("full", "full", ds)
    Tx = stx.terms("full", "full", ds)
    P = G.by_video(G.decode_dir("em0", "full", "", G.PART[ds]) + "/predictions.jsonl", ds)
    vids = sorted(set(T) & set(P) & set(y))
    lab = np.array([int(y[v].max() > 0) for v in vids])
    frac = np.array([y[v].mean() for v in vids])
    feats = {}
    feats["zv"] = [T[v][0] for v in vids]
    feats["zT"] = [T[v][3] for v in vids]
    if ds == "HateClipSeg": feats["zTax"] = [Tx[v][3] for v in vids]
    feats["mw"] = [T[v][1] for v in vids]
    feats["maxw"] = [T[v][4].max() for v in vids]
    feats["top25w"] = [np.sort(T[v][4])[-max(1, len(T[v][4])//4):].mean() for v in vids]
    feats["A=mean min(z,zT)"] = [np.mean(np.minimum(T[v][4], T[v][3])) for v in vids]
    feats["dec p_video_lo"] = [P[v]["extra"]["p_video_logodds"] for v in vids]
    cp = {v: np.asarray(P[v]["extra"]["cell_prob"], float) for v in vids}
    feats["dec mean cell p"] = [cp[v].mean() for v in vids]
    feats["dec max cell lo"] = [lg(cp[v]).max() for v in vids]
    feats["dec mean cell lo"] = [lg(cp[v]).mean() for v in vids]
    feats["dec top25 cell lo"] = [np.sort(lg(cp[v]))[-max(1, len(cp[v])//4):].mean() for v in vids]
    print(f"== {ds}: {len(vids)} videos, {lab.sum()} with hate; frame base rate {np.concatenate([y[v] for v in vids]).mean():.3f}")
    F = {k: np.asarray(v, float) for k, v in feats.items()}
    for k, v in F.items():
        ok = np.isfinite(v)
        print(f"  {k:22s} video AUC {roc_auc_score(lab[ok], v[ok]):.3f}   corr with hate fraction {np.corrcoef(frac[ok], v[ok])[0,1]:+.3f}")
    z = lambda k: (F[k] - F[k].mean()) / F[k].std()
    combos = {"zv+mw": z("zv")+z("mw"), "zT+mw": z("zT")+z("mw"), "zT+maxw": z("zT")+z("maxw"), "zT+top25w": z("zT")+z("top25w"),
              "zT+A": z("zT")+z("A=mean min(z,zT)"), "zT+dec_pv": z("zT")+z("dec p_video_lo"), "zT+dec top25": z("zT")+z("dec top25 cell lo"),
              "zT+zv+mw": z("zT")+z("zv")+z("mw"), "zT+zv": z("zT")+z("zv"), "zv+dec_pv": z("zv")+z("dec p_video_lo"),
              "zT+zv+top25w": z("zT")+z("zv")+z("top25w"), "zT+zv+dec top25": z("zT")+z("zv")+z("dec top25 cell lo")}
    if ds == "HateClipSeg":
        combos.update({"zTax+mw": z("zTax")+z("mw"), "zTax+top25w": z("zTax")+z("top25w"), "zTax+zv+mw": z("zTax")+z("zv")+z("mw")})
    for k, v in combos.items():
        print(f"  {k:22s} video AUC {roc_auc_score(lab, v):.3f}")
