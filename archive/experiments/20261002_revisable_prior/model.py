#!/usr/bin/env python3
"""Label-free random-effect temporal model. No ground truth enters this module."""
import argparse
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit, logsumexp
from scipy.stats import norm, rankdata

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "CLAUDE.md").is_file())
sys.path.insert(0, str(ROOT))
from src.temporal import build_chain, fb_grid, length_grid

MODS = ("z_visual", "z_speech")
VAR_FLOOR = 1e-6
PROB_FLOOR = 1e-9


def normal_log(x, mu, var):
    return -.5 * (np.log(2 * np.pi * var) + (x - mu) ** 2 / var)


def quadrature(n):
    u, w = hermgauss(n)
    return np.sqrt(2) * u, np.log(w / np.sqrt(np.pi))


def adaptive_quadrature(v, p, x, step=1.0):
    """Enclose every Gaussian component of p(u | all observations).

    All paths have equal posterior variance since emission variance is tied
    across states. Bounds include both V regimes and all local state choices.
    Weights retain the standard-normal density; never renormalize truncation.
    """
    precision = 1.0
    low = high = 0.0
    if p["arm"] != "no_global":
        precision += p["tau"]**2 / p["var"][0]
        low += (x - max(p["mu"][:2])) / p["var"][0]
        high += (x - min(p["mu"][:2])) / p["var"][0]
    for j, obs in enumerate(v["obs"]):
        y = obs["y"]; mu = p["mu"][2+2*j:4+2*j]
        precision += p["tau"]**2 * len(y) / p["var"][j+1]
        low += (y - max(mu)).sum() / p["var"][j+1]
        high += (y - min(mu)).sum() / p["var"][j+1]
    sd = precision**-.5
    left = p["tau"] * low / precision - 8*sd
    right = p["tau"] * high / precision + 8*sd
    count = max(2, int(np.ceil((right-left)/(step*sd)))+1)
    u = np.linspace(left, right, count)
    logw = normal_log(u, 0, 1) + math.log((right-left)/(count-1))
    logw[[0,-1]] -= math.log(2)
    return u, logw


def load_videos(path, datasets):
    records = {}
    for line in (Path(path) / "predictions.jsonl").open():
        r = json.loads(line)
        records[(r["dataset"], r["video_id"])] = r
    videos = {ds: [] for ds in datasets}
    for (ds, vid), r in sorted(records.items()):
        if ds not in datasets:
            continue
        assert not r.get("error"), (ds, vid, r.get("error"))
        assert r["native_rate"] == 4, (ds, vid)
        n = max(1, int(math.ceil(float(r["duration"]) / 4 - 1e-9)))
        wins = r["extra"]["windows"]
        cells, pair = [], []
        for w in wins:
            idx = [c for c in range(n) if min(w["end"], (c+1)*4) - max(w["start"], c*4) > 1e-6]
            assert 1 <= len(idx) <= 2 and (len(idx) == 1 or idx[1] == idx[0]+1), (vid, w)
            cells.append(idx[-1]); pair.append(len(idx) == 2)
        assert len(set(cells)) == len(cells), vid
        v = {"rec": r, "n": n, "L": len(r["score_curve"]), "x": float(r["extra"]["z_video"]),
             "cells": np.array(cells), "pair": np.array(pair), "obs": [], "chains": []}
        grid = length_grid(n, 4, 6)
        lw = np.array([math.log(a)+math.log(b) for a in grid for b in grid])
        v["logw"] = lw - logsumexp(lw)
        for m in MODS:
            use = np.array([i for i, w in enumerate(wins) if m in w], int)
            v["obs"].append({"use": use, "y": np.array([wins[i][m] for i in use], float)})
            v["chains"].append([build_chain(4, a, b, .5, mods=(m,)) for a in grid for b in grid])
        videos[ds].append(v)
    for ds, vs in videos.items():
        assert vs, ds
        x = norm.ppf((rankdata([v["x"] for v in vs])-.5) / len(vs))
        for v, value in zip(vs, x):
            v["x"] = float(value)
        for j in range(2):
            ys = np.concatenate([v["obs"][j]["y"] for v in vs])
            assert len(ys), (ds, MODS[j])
            scaled = norm.ppf((rankdata(ys)-.5) / len(ys))
            offset = 0
            for v in vs:
                count = len(v["obs"][j]["y"])
                v["obs"][j]["y"] = scaled[offset:offset+count].copy(); offset += count
            assert offset == len(ys)
    return videos


def initial(videos, arm):
    groups = [np.array([v["x"] for v in videos])]
    groups += [np.concatenate([v["obs"][j]["y"] for v in videos]) for j in range(2)]
    return {"mu": np.concatenate([np.percentile(g, [10,90]) for g in groups]),
            "var": np.array([max(g.var(), VAR_FLOOR) for g in groups]),
            "tau": 0.0 if arm == "independent" else .5, "pi": .5,
            "iota": np.full((2,2), .5), "arm": arm}


def terms(v, p, nodes=7, x_override=None, grid_step=None):
    x = v["x"] if x_override is None else x_override
    if p["arm"] == "independent":
        u, loguw = quadrature(1)
    elif grid_step is not None:
        u, loguw = adaptive_quadrature(v, p, x, grid_step)
    else:
        u, loguw = quadrature(nodes)
    Q = len(u); b = p["tau"] * u
    glob = np.zeros((2,Q)) if p["arm"] == "no_global" else normal_log(x, p["mu"][:2,None]+b, p["var"][0])
    ll0, ll1 = np.zeros(Q), np.zeros(Q)
    gs, ons = [], []
    for j, m in enumerate(MODS):
        obs, chains = v["obs"][j], v["chains"][j]
        for ch in chains:
            ch["pi0"][:] = 0
            ch["pi0"][:ch["S"]] = p["iota"][j][ch["ph"]] / ch["k"]
        ch = chains[0]
        use, y = obs["use"], obs["y"]
        levels = np.array([ch["lvl"][m]["pair" if v["pair"][i] else "single"] for i in use], int).reshape(len(use), -1) if len(use) else np.zeros((0, len(ch["pi0"])), int)
        ll = normal_log(y[None,:,None], p["mu"][2+2*j:4+2*j][None,None,:]+b[:,None,None], p["var"][j+1])
        E = np.zeros((Q, v["n"], len(ch["pi0"])))
        E[:,v["cells"][use],:] = ll[:,:,0,None]*(1-levels[None]) + ll[:,:,1,None]*levels[None]
        ll0 += ll[:,:,0].sum(axis=1)
        ev, g, _ = fb_grid(E, chains, v["logw"])
        ll1 += ev
        gs.append(g)
        ons.append((g[:,v["cells"][use],:] * levels[None]).sum(axis=-1))
    lp = np.stack([math.log1p(-p["pi"])+glob[0]+ll0+loguw,
                   math.log(p["pi"])+glob[1]+ll1+loguw])
    evidence = float(logsumexp(lp))
    r = np.exp(lp - evidence)
    lo = float(logsumexp(lp[1]) - logsumexp(lp[0]))
    # Conditional u posterior must be normalized within V=1. OR is evaluated
    # before mixing over u, because the modalities share the same random effect.
    q = np.exp(lp[1] - logsumexp(lp[1]))
    miss = np.ones((Q, v["n"]))
    for j, g in enumerate(gs):
        miss *= 1 - g[:,:,v["chains"][j][0]["hate"] == 1].sum(axis=-1)
    conditional = q @ (1-miss)
    return {"ll": evidence, "r": r, "u": u, "g": gs, "on": ons, "lo": lo,
            "conditional": np.clip(conditional, 0, 1), "unconditional": r[1] @ (1-miss),
            "offset_mean": float(r.sum(axis=0) @ b)}


def moments(weights, y, u):
    y = np.asarray(y)[None,:]; u = np.asarray(u)[:,None]
    return np.array([weights.sum(), (weights*y).sum(), (weights*y*y).sum(),
                     (weights*u).sum(), (weights*u*u).sum(), (weights*u*y).sum()])


def m_step(p, S, mass, starts, count, prior_moments=None):
    active = list(range(2,6)) if p["arm"] == "no_global" else list(range(6))
    effect = p["arm"] != "independent"
    d = len(active) + int(effect)
    A, rhs = np.zeros((d,d)), np.zeros(d)
    for i, k in enumerate(active):
        w, x, xx, u, uu, ux = S[k] / p["var"][k//2]
        A[i,i] += w; rhs[i] += x
        if effect:
            A[i,-1] += u; A[-1,i] += u; A[-1,-1] += uu; rhs[-1] += ux
    coef = np.linalg.lstsq(A, rhs, rcond=None)[0]
    mu = p["mu"].copy(); mu[active] = coef[:len(active)]
    tau = float(coef[-1]) if effect else 0.0
    var = p["var"].copy()
    for j in range(3):
        if j == 0 and p["arm"] == "no_global":
            continue
        rss, den = 0., 0.
        for k in (2*j, 2*j+1):
            w, x, xx, u, uu, ux = S[k]
            rss += xx + mu[k]**2*w + tau**2*uu - 2*mu[k]*x - 2*tau*ux + 2*mu[k]*tau*u
            den += w
        var[j] = max(float(rss / max(den, PROB_FLOOR)), VAR_FLOOR)
    io = np.clip(starts / max(mass, PROB_FLOOR), PROB_FLOOR, 1-PROB_FLOOR)
    io /= io.sum(axis=1, keepdims=True)
    # PX-EM: maximize an expanded N(center, spread) latent prior, then reduce
    # back to N(0,1). This represents exactly the same observation model.
    if effect and prior_moments is not None:
        center, second = prior_moments / count
        spread = max(second-center**2, VAR_FLOOR)
        mu[active] += tau*center
        tau *= math.sqrt(spread)
    return {**p, "mu": mu, "tau": abs(tau), "var": var, "pi": float(np.clip(mass/count, PROB_FLOOR, 1-PROB_FLOOR)), "iota": io}


def fit(videos, arm, nodes=7, max_it=100, log=print, start=None, grid_step=None, px=False):
    assert not px or grid_step is not None, "PX reduction requires continuous quadrature"
    p = initial(videos, arm) if start is None else start
    assert p["arm"] == arm
    previous = None; history = []
    for it in range(max_it):
        S = np.zeros((6,6)); starts = np.zeros((2,2)); total, mass = 0., 0.
        prior_moments = np.zeros(2)
        begin = time.time()
        for v in videos:
            t = terms(v,p,nodes,grid_step=grid_step); total += t["ll"]
            r, u = t["r"], t["u"]; mass += r[1].sum()
            prior_moments += [r.sum(axis=0)@u, r.sum(axis=0)@(u*u)]
            if arm != "no_global":
                for h in (0,1):
                    S[h] += moments(r[h,:,None], [v["x"]], u)
            for j in range(2):
                on = t["on"][j]
                for h, weight in ((0,r[0,:,None]+r[1,:,None]*(1-on)), (1,r[1,:,None]*on)):
                    S[2+2*j+h] += moments(weight, v["obs"][j]["y"], u)
                phase = v["chains"][j][0]["phase"]
                for h in (0,1):
                    starts[j,h] += (r[1] * t["g"][j][:,0,phase == h].sum(axis=-1)).sum()
        if previous is not None and total < previous - 1e-6*max(1,abs(previous)):
            raise AssertionError(f"EM decreased: {previous} -> {total}")
        history.append(total)
        log(f"iter {it+1} ll={total:.6f} tau={p['tau']:.4f} pi={p['pi']:.4f} seconds={time.time()-begin:.1f}")
        if previous is not None and abs(total-previous) <= 1e-6*max(1,abs(previous)):
            break
        if it < max_it-1:
            p = m_step(p,S,mass,starts,len(videos),prior_moments if px else None)
        previous = total
    p["loglik_history"] = history
    p["converged"] = len(history)>1 and abs(history[-1]-history[-2]) <= 1e-6*max(1,abs(history[-2]))
    relevant = (1,2) if arm == "no_global" else (0,1,2)
    p["ordered_means"] = all(p["mu"][2*j+1] > p["mu"][2*j] for j in relevant)
    return p


def json_params(p):
    return {k: v.tolist() if isinstance(v,np.ndarray) else v for k,v in p.items()}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run",default="runs/20260910_spvl/mllm/q3vl-8b/nostance")
    ap.add_argument("--datasets",nargs="+",default=["HateMM","HateClipSeg"])
    ap.add_argument("--arm",choices=["full","independent","no_global"],required=True)
    ap.add_argument("--nodes",type=int,default=7)
    ap.add_argument("--max-it",type=int,default=100)
    ap.add_argument("--grid-step",type=float,default=None,help="adaptive trapezoid spacing in posterior standard deviations")
    ap.add_argument("--px",action="store_true",help="equivalent parameter-expanded EM solver")
    ap.add_argument("--init-params",default="",help="numerical continuation only: same Reader, arm, datasets")
    ap.add_argument("--tag",required=True)
    ap.add_argument("--out-root",default="runs/20261002_revisable_prior")
    a=ap.parse_args(); out=ROOT/a.out_root/a.tag; out.mkdir(parents=True,exist_ok=True)
    f=(out/"run.log").open("w")
    def log(msg):
        line=f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"; print(line,flush=True);f.write(line+"\n");f.flush()
    log("host "+socket.gethostname());(out/"run.pid").write_text(str(os.getpid()))
    config={**vars(a),"code":"archive/experiments/20261002_revisable_prior/model.py + src/temporal.py; local sources 2026-10-02",
            "host":socket.gethostname(),"GT_in_fit":False}
    (out/"config.json").write_text(json.dumps(config,indent=2)+"\n")
    vs=load_videos(ROOT/a.run,a.datasets);params={};checks={};starts={}
    if a.init_params:
        src=Path(a.init_params)
        before=json.load((src.parent/"config.json").open())
        assert before["run"]==a.run and before["arm"]==a.arm and before["datasets"]==a.datasets
        starts=json.load(src.open())
        for p in starts.values():
            for k in ("mu","var","iota"):p[k]=np.asarray(p[k],float)
    with (out/"predictions.jsonl").open("w") as pred:
        for ds in a.datasets:
            p=fit(vs[ds],a.arm,a.nodes,a.max_it,lambda s:log(ds+" "+s),start=starts.get(ds),grid_step=a.grid_step,px=a.px);params[ds]=json_params(p)
            log(f"{ds} ordered_means={p['ordered_means']} converged={p['converged']} mu={p['mu'].tolist()}")
            errors=[]
            for v in vs[ds]:
                t=terms(v,p,a.nodes,grid_step=a.grid_step)
                check=terms(v,p,2*a.nodes+1,grid_step=a.grid_step/2 if a.grid_step else None) if a.arm!="independent" else t
                errors.append([abs(expit(t["lo"])-expit(check["lo"])),float(np.mean(np.abs(t["conditional"]-check["conditional"])))])
                idx=np.clip((((np.arange(v["L"])+.5)/4)//4).astype(int),0,v["n"]-1)
                values=np.clip(t["conditional"][idx],1e-12,1-1e-12)
                rank=(rankdata(values,method="average")-.5)/len(values)-.5;rank-=rank.mean()
                score=t["lo"]+rank
                rec={**v["rec"],"method":"revisable_prior__"+a.tag,"score_curve":score.tolist(),"intervals":[],
                     "code_path":"archive/experiments/20261002_revisable_prior/model.py",
                     "extra":{"z_video":v["rec"]["extra"]["z_video"],"global_nscore":v["x"],"video_logodds":t["lo"],
                              "cell_prob":t["conditional"].tolist(),"unconditional_cell_prob":t["unconditional"].tolist(),
                              "offset_mean":t["offset_mean"]}}
                pred.write(json.dumps(rec)+"\n")
            checks[ds]={"nodes_fit":a.nodes,"nodes_check":2*a.nodes+1,"median_abs_change":np.median(errors,axis=0).tolist(),
                        "grid_step_fit":a.grid_step,"grid_step_check":a.grid_step/2 if a.grid_step else None,
                        "p95_abs_change":np.quantile(errors,.95,axis=0).tolist(),
                        "max_abs_change":np.max(errors,axis=0).tolist(),
                        "needs_refit":bool(np.any(np.median(errors,axis=0)>.01) or np.any(np.quantile(errors,.95,axis=0)>.05))}
            log(ds+" quadrature "+json.dumps(checks[ds]))
            (out/"params.json").write_text(json.dumps(params,indent=2)+"\n")
            (out/"quadrature.json").write_text(json.dumps(checks,indent=2)+"\n")
    subprocess.run([sys.executable,"-m","src.eval.evaluate_four_datasets","--predictions",str(out/"predictions.jsonl"),
                    "--gt-dir",str(ROOT/"data/gt_4fps"),"--out",str(out/"metrics.json"),"--datasets",*a.datasets],check=True,cwd=ROOT)
    log("RUN_DONE")


if __name__=="__main__":
    main()
