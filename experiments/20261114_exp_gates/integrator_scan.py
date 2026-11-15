#!/usr/bin/env python3
"""Integrator scan for README §13 (2026-11-16; CPU, cached reads and decodes; test-read, development only).

Not-below counts of the appendix rows for video-level forms x within-video weight, decoder em0, no bootstrap.
  python experiments/20261114_exp_gates/integrator_scan.py <forms> <modes> <pars> [tax]
forms: mw+zT, A+zT, mw+zv+zT, A+zv+zT, mw+zv, top25+zT, top50+zT (every term divided by its corpus s.d. on the full
       method's reads; A = mean over windows of min(window read, scope read); topQ = mean of the top Q of the windows)
modes: add (K + par * centred rank), prod (corpus rank of K x within-video quantile^par), logprod, addcdf,
       gate (K + par * corpus rank of K * centred rank)
tax:   HateClipSeg scope read = its taxonomy read (README §6); default = the protected-group read everywhere.
A cell is "not below" when the change against the full method is > -0.05 points (as gate_table.py).
"""
import sys
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import gate_table as G
G.ROWS = G.ROWS_EM0
DS = G.DATASETS
FULL = ("full", "", "full", "full", "", None)


def stats(store, ds):
    T = store.terms("full", "full", ds)
    a = lambda f: np.array([f(t) for t in T.values()], float)
    top = lambda w, q: float(np.sort(w)[-max(1, int(np.ceil(len(w) * q))):].mean()) if len(w) else 0.0
    return {"top25": a(lambda t: top(t[4], .25)).std(), "top50": a(lambda t: top(t[4], .5)).std(),
            "mw": a(lambda t: t[1]).std(), "zv": a(lambda t: t[0]).std(), "zt": a(lambda t: t[3]).std(),
            "A": a(lambda t: np.mean(np.minimum(t[4], t[3])) if len(t[4]) else 0.0).std()}

def K(form, t, integ, sg):
    zv, mw, mm, zt, wz, wm = t
    if integ == "nokey": return 0.0
    w = wm if integ == "mean_mod" else wz
    agg_mw = (mm if integ == "mean_mod" else mw) / sg["mw"]
    agg_A = (float(np.mean(np.minimum(w, zt))) if len(w) else 0.0) / sg["A"]
    S = 0.0 if integ == "no_scope" else zt / sg["zt"]
    V = zv / sg["zv"]
    if form.startswith("top"):
        q = {"top25": .25, "top50": .5}[form.split("+")[0]]
        tw = float(np.sort(w)[-max(1, int(np.ceil(len(w) * q))):].mean()) if len(w) else 0.0
        return (0.0 if integ == "no_window" else tw / sg[form.split("+")[0]]) + S + (V if "+zv" in form else 0.0)
    if form == "mw+zT":   return (0.0 if integ == "no_window" else agg_mw) + S
    if form == "A+zT":    return (mw / sg["mw"] if integ == "no_scope" else (0.0 if integ == "no_window" else agg_A)) + S
    if form == "mw+zv+zT": return (0.0 if integ == "no_window" else agg_mw) + V + S
    if form == "A+zv+zT": return (mw / sg["mw"] if integ == "no_scope" else (0.0 if integ == "no_window" else agg_A)) + V + S
    if form == "mw+zv":   return (0.0 if integ == "no_window" else agg_mw) + V
    raise ValueError(form)

def scores(store, ds, row, form, mode, par, sg):
    name, grp, rs, sp, rflag, integ = row
    rk = store.rank(rs, "" if integ else rflag, ds); T = store.terms(rs, sp, ds)
    vids = [v for v in rk if v in T]
    Kv = np.array([K(form, T[v], integ, sg) for v in vids])
    F = (rankdata(Kv, method="average") - 0.5) / len(Kv) if np.ptp(Kv) > 0 else np.full(len(Kv), .5)   # own corpus CDF
    out = {}
    for v, k, f in zip(vids, Kv, F):
        u = rk[v] + 0.5                         # within-video quantile in (0,1); constant curve -> .5
        if mode == "add":      s = k + par * rk[v]
        elif mode == "prod":   s = f * u ** par                       # P(V) x within quantile^par
        elif mode == "logprod": s = np.log(f) + par * np.log(u)
        elif mode == "addcdf": s = f + par * rk[v]
        elif mode == "gate":   s = k + par * f * rk[v]                # within spread scaled by the video belief
        out[v] = s
    return out

def table(store, form, mode, par):
    out, full = {}, {}
    for ds in DS:
        y = store.gt[ds]; sg = stats(store, ds)
        sf = scores(store, ds, FULL, form, mode, par, sg); full[ds] = G.metrics(y, sf)
        for row in G.ROWS:
            base = scores(store, ds, ("", "", "full", "full", "compose", None), form, mode, par, sg) if row[4] == "compose" else sf
            out[(row[0], ds)] = 100 * (G.metrics(y, scores(store, ds, row, form, mode, par, sg)) - G.metrics(y, base))
    return full, out

AB = {"HateMM": "MM", "HateClipSeg": "HCS", "DeHate": "DH"}
def report(tag, full, out):
    nb, cells, extra, nb_rw = 0, [], [], 0
    for row in G.ROWS:
        for ds in DS:
            for k in range(3):
                if row[5] and k == 2: continue
                d = out[(row[0], ds)][k]
                if d > -0.05:
                    s = f"{row[0].split(' ',1)[1]}:{AB[ds]}.{'RPW'[k]}{d:+.2f}"
                    if row[1].startswith("extra"): extra.append(s)
                    else:
                        nb += 1; cells.append(s); nb_rw += k != 1
    print(f"{tag:40s} full " + " | ".join("/".join(f"{100*x:.1f}" for x in full[ds]) for ds in DS) + f" | not below {nb} (ROC+W only: {nb_rw})")
    print("      ", "; ".join(cells)); print("       extra:", "; ".join(extra), flush=True)

if __name__ == "__main__":
    store = G.Store(["em0"], "prob", taxonomy=(len(sys.argv) > 4 and sys.argv[4] == "tax"))
    for form in sys.argv[1].split(","):
        for mode in sys.argv[2].split(","):
            for par in [float(x) for x in sys.argv[3].split(",")]:
                full, out = table(store, form, mode, par)
                report(f"{form} {mode} par={par:g}", full, out)
