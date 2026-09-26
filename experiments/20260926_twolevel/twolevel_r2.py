#!/usr/bin/env python3
"""Round 2 of the two-level model (CPU, cached reads). See README.md §10 in this directory.

Change against round 1 (twolevel.py): the time level uses an explicit-duration chain. A hate segment and a gap
each last a negative-binomial number of 4 s cells (shape k, declared mean), built as k sub-states in a row; k = 1
is the geometric chain of round 1. Each 8 s window's read observes "any hate in the window's cells" through the
previous cell's hate bit, as in round 1. Emissions, video level and EM are as in round 1 (twolevel.py), without
the at-least-one constraint. No labels are read here; evaluation only through src/eval/evaluate_four_datasets.py.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import expit, log_expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from twolevel import (CELL, EPS, MODS, ROOT, centered_rank, intercept, intervals_from, load_run, lognorm,  # noqa: E402
                      prep, to_frames)


# ----------------------------------------------------------------------------------------------- chain

def build_chain(k, d_gap, d_hate, iota, nocoupling=False):
    """Sub-states 0..k-1 = gap, k..2k-1 = hate. Mean durations d_gap, d_hate in seconds. Returns hate bit per
    sub-state, augmented transition matrix over (previous cell's hate bit, sub-state), initial distribution.
    nocoupling (ablation): k = 1 and independent cells with P(hate) = iota."""
    if nocoupling:
        k = 1
    S = 2 * k
    hb = (np.arange(S) >= k).astype(int)
    A = np.zeros((S, S))
    for x in range(S):
        if nocoupling:
            A[x] = [1 - iota, iota]
            continue
        q = 1.0 - k * CELL / (d_hate if hb[x] else d_gap)
        q = min(max(q, 0.0), 1 - 1e-9)
        A[x, x] = q
        A[x, (x + 1) % S] += 1.0 - q          # last gap sub-state -> first hate sub-state and vice versa
    T = np.zeros((2 * S, 2 * S))               # augmented state z = b * S + x, b = hate bit of the previous cell
    for bp in range(2):
        for xp in range(S):
            for x in range(S):
                T[bp * S + xp, hb[xp] * S + x] = A[xp, x]
    pi0 = np.zeros(2 * S)
    pi0[:k] = (1 - iota) / k                   # first cell: previous hate bit 0, phase start uniform in the phase
    pi0[k:S] = iota / k
    return {"k": k, "S": S, "hb": hb, "T": T, "pi0": pi0}


def on_masks(ch):
    S, hb = ch["S"], ch["hb"]
    b = np.repeat([0, 1], S)
    cur = np.tile(hb, 2)
    return {"pair": ((b + cur) > 0).astype(int), "single": cur}


CARRIERS = ("visual", "speech", "both")


def carrier_terms(w, P):
    """Carrier fusion: log p(reads | window hateful, carrier c) for c in CARRIERS and log p(reads | not hateful)."""
    lv = {m: (lognorm(w["y"][m], P["emit"][m]["mu11"], P["emit"][m]["s2"]),
              lognorm(w["y"][m], P["emit"][m]["mu10"], P["emit"][m]["s2"])) for m in MODS if m in w["y"]}
    off = sum(x[1] for x in lv.values())
    hot = {"visual": ("z_visual",), "speech": ("z_speech",), "both": MODS}
    on = np.array([sum(lv[m][0] if m in hot[c] else lv[m][1] for m in lv) for c in CARRIERS])
    return on, off


def emission(v, mods, P, ch, kappa=1.0):
    """log emission over augmented states, shape (n, 2S), for the chain of modalities `mods` under V = 1."""
    masks = on_masks(ch)
    E = np.zeros((v["n"], 2 * ch["S"]))
    if P.get("carrier") is not None:
        lc = np.log(np.asarray(P["carrier"]))
        for w in v["wins"]:
            on = masks["pair"] if w["pair"] else masks["single"]
            ons, off = carrier_terms(w, P)
            E[w["cell"]] += kappa * np.where(on == 1, float(np.logaddexp.reduce(lc + ons)), off)
        return E
    for w in v["wins"]:
        on = masks["pair"] if w["pair"] else masks["single"]
        for m in mods:
            if m in w["y"]:
                e = P["emit"][m]
                E[w["cell"]] += kappa * np.where(on == 1, lognorm(w["y"][m], e["mu11"], e["s2"]),
                                                 lognorm(w["y"][m], e["mu10"], e["s2"]))
    return E


def fb(E, ch):
    """Scaled forward-backward. Returns log-likelihood and the posterior over augmented states (n, 2S)."""
    n = E.shape[0]
    T = ch["T"]
    m = E.max(1, keepdims=True)
    G = np.exp(E - m)
    al = np.zeros_like(E); cs = np.zeros(n)
    a = ch["pi0"] * G[0]; cs[0] = a.sum(); al[0] = a / cs[0]
    for c in range(1, n):
        a = (al[c - 1] @ T) * G[c]; cs[c] = a.sum(); al[c] = a / cs[c]
    be = np.ones_like(E)
    for c in range(n - 2, -1, -1):
        be[c] = T @ (G[c + 1] * be[c + 1]) / cs[c + 1]
    g = al * be
    g /= g.sum(1, keepdims=True)
    return float(np.log(cs).sum() + m.sum()), g


def p_hate(g, ch):
    return g[:, np.tile(ch["hb"], 2) == 1].sum(1)


def p_on(v, g, ch):
    masks = on_masks(ch)
    return [float(g[w["cell"], (masks["pair"] if w["pair"] else masks["single"]) == 1].sum()) for w in v["wins"]]


def selftest(trials=100, seed=0):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        k = int(rng.integers(1, 3)); n = int(rng.integers(1, 6))
        ch = build_chain(k, rng.uniform(4 * k + 1, 60), rng.uniform(4 * k + 1, 60), rng.uniform(.05, .95))
        E = rng.normal(0, 2, size=(n, 2 * ch["S"]))
        ll, g = fb(E, ch)
        S = ch["S"]; tot = -np.inf; marg = np.zeros(n); lps = []; paths = list(itertools.product(range(S), repeat=n))
        A = ch["T"][:S, :S] + ch["T"][:S, S:]           # sub-state transitions (previous bit 0 rows hold A)
        for xs in paths:
            b = 0; lp = math.log(max(ch["pi0"][xs[0]], 1e-300)) + E[0, xs[0]]
            for c in range(1, n):
                b = ch["hb"][xs[c - 1]]
                t = A[xs[c - 1], xs[c]]
                lp += math.log(t) if t > 0 else -np.inf
                lp += E[c, b * S + xs[c]]
            lps.append(lp)
        lps = np.array(lps); LL = np.logaddexp.reduce(lps); post = np.exp(lps - LL)
        for c in range(n):
            marg[c] = sum(p for p, xs in zip(post, paths) if ch["hb"][xs[c]])
        worst = max(worst, abs(ll - LL), float(np.abs(marg - p_hate(g, ch)).max()))
    import twolevel as r1
    for _ in range(trials):                      # k = 1 equals round 1's geometric pair-state chain
        n = int(rng.integers(1, 30)); dg, dh, io = rng.uniform(5, 200), rng.uniform(5, 200), rng.uniform(.05, .95)
        E = rng.normal(0, 2, size=(n, 4))
        ll, g = fb(E, build_chain(1, dg, dh, io))
        ll1, g1, _, _ = r1.fb(E, io, 1 - CELL / dg, 1 - CELL / dh)
        worst = max(worst, abs(ll - ll1), float(np.abs(p_hate(g, build_chain(1, dg, dh, io)) - (g1[:, 1] + g1[:, 3])).max()))
    return worst


# ----------------------------------------------------------------------------------------------- EM

def chains_of(flags):
    return [("z_visual", "z_speech")] if flags.get("sharedchain") or flags.get("carrier") else [("z_visual",), ("z_speech",)]


def init_params(videos, flags):
    zs = np.array([v["zv"] for v in videos])
    P = {"pi": 0.5, "m0": float(np.percentile(zs, 10)), "m1": float(np.percentile(zs, 90)), "t2": float(zs.var()),
         "emit": {}, "iota": {mods: 0.5 for mods in chains_of(flags)}}
    for m in MODS:
        ys = np.array([w["y"][m] for v in videos for w in v["wins"] if m in w["y"]])
        mu00, mu10, mu11 = (float(np.percentile(ys, q)) for q in (10, 50, 90))
        if flags.get("noleak"):
            mu10 = mu00
        P["emit"][m] = {"mu00": mu00, "mu10": mu10, "mu11": mu11, "s2": float(ys.var())}
    P["carrier"] = [1 / 3, 1 / 3, 1 / 3] if flags.get("carrier") else None
    return P


def chain_for(P, mods, flags):
    return build_chain(flags["k"], flags["d_gap"], flags["d_hate"], P["iota"][mods], flags.get("nocoupling", False))


def video_terms(v, P, flags, kappa=1.0):
    lv0 = lognorm(v["zv"], P["m0"], P["t2"]); lv1 = lognorm(v["zv"], P["m1"], P["t2"])
    l0 = sum(lognorm(y, P["emit"][m]["mu00"], P["emit"][m]["s2"]) for w in v["wins"] for m, y in w["y"].items())
    per = []
    for mods in chains_of(flags):
        ch = chain_for(P, mods, flags)
        ll, g = fb(emission(v, mods, P, ch, kappa), ch)
        per.append((mods, ch, ll, g))
    l1 = sum(x[2] for x in per)
    lo_verdict = math.log(P["pi"]) - math.log(1 - P["pi"]) + lv1 - lv0
    lo = lo_verdict + l1 - l0
    total = float(np.logaddexp(math.log(P["pi"]) + lv1 + l1, math.log(1 - P["pi"]) + lv0 + l0))
    out = {"per": per, "lo": lo, "lo_verdict": lo_verdict, "total": total}
    if "icc" in P and not flags.get("sharedchain"):
        # reads of one video share its context: modality m's n reads count as n / (1 + (n - 1) icc_m) reads
        lo_r = 0.0
        for mods, ch, ll, g in per:
            m = mods[0]
            n_m = sum(1 for w in v["wins"] if m in w["y"])
            if n_m == 0:
                continue
            l0_m = sum(lognorm(w["y"][m], P["emit"][m]["mu00"], P["emit"][m]["s2"]) for w in v["wins"] if m in w["y"])
            lo_r += (ll - l0_m) / (1.0 + (n_m - 1) * P["icc"][m])
        out["lo_icc"] = lo_verdict + lo_r
    return out


def vlevel_features(v):
    """Video-level statistics: the verdict read and each modality's mean read (NaN when the modality has no read)."""
    f = [v["zv"]]
    for m in MODS:
        ys = [w["y"][m] for w in v["wins"] if m in w["y"]]
        f.append(float(np.mean(ys)) if ys else float("nan"))
    return np.array(f)


def vlevel_mixture(videos, max_it=500, tol=1e-9):
    """Two-component Gaussian mixture over (verdict, mean visual read, mean speech read), one variance per feature
    shared by the components (linear log-odds), missing features skipped. EM, label-free. Returns a function
    video -> log-odds of the violating component (the component with the larger verdict mean)."""
    X = np.array([vlevel_features(v) for v in videos]); M = ~np.isnan(X); Xz = np.where(M, X, 0.0)
    mu = np.stack([np.nanpercentile(X, 10, axis=0), np.nanpercentile(X, 90, axis=0)])
    var = np.nanvar(X, axis=0); pi = 0.5; prev = None
    for _ in range(max_it):
        ll = [np.where(M, -0.5 * (np.log(2 * np.pi * var) + (Xz - mu[j]) ** 2 / var), 0.0).sum(1) for j in (0, 1)]
        a0, a1 = math.log(1 - pi) + ll[0], math.log(pi) + ll[1]
        tot = float(np.logaddexp(a0, a1).sum()); r = expit(a1 - a0)
        if prev is not None and abs(tot - prev) <= tol * abs(prev):
            break
        prev = tot
        pi = float(np.clip(r.mean(), EPS, 1 - EPS))
        for j, wj in ((0, 1 - r), (1, r)):
            W = wj[:, None] * M
            mu[j] = (W * Xz).sum(0) / np.maximum(W.sum(0), EPS)
        var = np.maximum(((1 - r)[:, None] * M * (Xz - mu[0]) ** 2 + r[:, None] * M * (Xz - mu[1]) ** 2).sum(0)
                         / np.maximum(M.sum(0), EPS), EPS)
    hi = 1 if mu[1, 0] >= mu[0, 0] else 0

    def lo(v):
        x = vlevel_features(v); m = ~np.isnan(x); x0 = np.where(m, x, 0.0)
        l = [np.where(m, -0.5 * (x0 - mu[j]) ** 2 / var, 0.0).sum() for j in (0, 1)]
        lp = [math.log(1 - pi), math.log(pi)]
        return (lp[hi] + l[hi]) - (lp[1 - hi] + l[1 - hi])
    return lo, {"pi": pi if hi == 1 else 1 - pi, "mu_violating": mu[hi].tolist(), "mu_other": mu[1 - hi].tolist(),
                "var": var.tolist(), "loglik": prev}


def residual_icc(videos, P, flags):
    """One-way random-effects ICC of the reads' residuals (read minus its posterior expected mean), per modality.
    Label-free: uses the fitted model only."""
    res = {m: [] for m in MODS}
    for v in videos:
        t = video_terms(v, P, flags)
        w = float(expit(t["lo"]))
        pa = {}
        for mods, ch, ll, g in t["per"]:
            for m in mods:
                pa[m] = p_on(v, g, ch)
        for m in MODS:
            e = P["emit"][m]; r = []
            for i, wi in enumerate(v["wins"]):
                if m in wi["y"]:
                    mean = (1 - w) * e["mu00"] + w * ((1 - pa[m][i]) * e["mu10"] + pa[m][i] * e["mu11"])
                    r.append(wi["y"][m] - mean)
            if len(r) >= 2:
                res[m].append(np.array(r))
    icc = {}
    for m, groups in res.items():
        k = len(groups); N = sum(len(g) for g in groups); grand = np.concatenate(groups).mean()
        msb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (k - 1)
        msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (N - k)
        n0 = (N - sum(len(g) ** 2 for g in groups) / N) / (k - 1)
        icc[m] = float(np.clip((msb - msw) / (msb + (n0 - 1) * msw), 0.0, 1.0))
    return icc


def em(videos, flags, max_it=300, tol=1e-7, log=print):
    P = init_params(videos, flags)
    prev, it = None, 0
    for it in range(max_it):
        S = {m: np.zeros((3, 3)) for m in MODS}; W, Z = [], []; io = {mods: [0.0, 0.0] for mods in chains_of(flags)}
        CR = np.zeros(3)
        total = 0.0
        for v in videos:
            t = video_terms(v, P, flags); total += t["total"]
            w = float(expit(t["lo"])); W.append(w); Z.append(v["zv"])
            for wi in v["wins"]:
                for m, y in wi["y"].items():
                    S[m][0] += (1 - w) * np.array([1.0, y, y * y])
            for mods, ch, ll, g in t["per"]:
                if flags.get("nocoupling"):
                    io[mods][0] += w * float(p_hate(g, ch).sum()); io[mods][1] += w * v["n"]
                else:
                    io[mods][0] += w * float(p_hate(g[:1], ch)[0]); io[mods][1] += w
                if P.get("carrier") is not None:
                    lc = np.log(np.asarray(P["carrier"]))
                    for wi, pa in zip(v["wins"], p_on(v, g, ch)):
                        ons, off = carrier_terms(wi, P)
                        r = np.exp(lc + ons - np.logaddexp.reduce(lc + ons))       # carrier responsibilities
                        CR += w * pa * r
                        hotw = {"z_visual": r[0] + r[2], "z_speech": r[1] + r[2]}
                        for m, y in wi["y"].items():
                            S[m][1] += w * (1 - pa * hotw[m]) * np.array([1.0, y, y * y])
                            S[m][2] += w * pa * hotw[m] * np.array([1.0, y, y * y])
                    continue
                for wi, pa in zip(v["wins"], p_on(v, g, ch)):
                    for m in mods:
                        if m in wi["y"]:
                            y = wi["y"][m]
                            S[m][1] += w * (1 - pa) * np.array([1.0, y, y * y])
                            S[m][2] += w * pa * np.array([1.0, y, y * y])
        if prev is not None:
            if total < prev - 1e-6 * abs(prev):
                raise AssertionError(f"EM log-likelihood decreased at iteration {it}: {prev} -> {total}")
            if abs(total - prev) <= tol * abs(prev):
                break
        prev = total
        W, Z = np.array(W), np.array(Z)
        P["pi"] = float(np.clip(W.mean(), EPS, 1 - EPS))
        P["m1"] = float((W * Z).sum() / max(W.sum(), EPS)); P["m0"] = float(((1 - W) * Z).sum() / max((1 - W).sum(), EPS))
        P["t2"] = max(float((W * (Z - P["m1"]) ** 2 + (1 - W) * (Z - P["m0"]) ** 2).mean()), EPS)
        for m in MODS:
            mu = S[m][:, 1] / np.maximum(S[m][:, 0], EPS)
            if flags.get("noleak"):
                mu[0] = mu[1] = (S[m][0, 1] + S[m][1, 1]) / max(S[m][0, 0] + S[m][1, 0], EPS)
            ss = sum(S[m][j, 2] - 2 * mu[j] * S[m][j, 1] + mu[j] ** 2 * S[m][j, 0] for j in range(3))
            P["emit"][m] = {"mu00": float(mu[0]), "mu10": float(mu[1]), "mu11": float(mu[2]),
                            "s2": max(float(ss / max(S[m][:, 0].sum(), EPS)), EPS)}
        for mods in io:
            P["iota"][mods] = float(np.clip(io[mods][0] / max(io[mods][1], EPS), EPS, 1 - EPS))
        if P.get("carrier") is not None:
            P["carrier"] = [float(x) for x in np.clip(CR / max(CR.sum(), EPS), 1e-4, 1.0)]
            P["carrier"] = [x / sum(P["carrier"]) for x in P["carrier"]]
    log(f"EM stop after {it + 1} iterations, log-likelihood {prev:.4f}")
    return P


# ----------------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "runs/20260926_glr/base_gridA"))
    ap.add_argument("--k", type=int, default=1, help="duration shape: sub-states per segment (1 = geometric)")
    ap.add_argument("--d-gap", type=float, default=80.0, help="mean gap duration, seconds")
    ap.add_argument("--d-hate", type=float, default=80.0, help="mean hate-segment duration, seconds")
    ap.add_argument("--center", action="store_true", help="reads centred within each video (fixed video effect)")
    ap.add_argument("--sharedchain", action="store_true")
    ap.add_argument("--nocoupling", action="store_true", help="ablation: independent cells")
    ap.add_argument("--noleak", action="store_true", help="ablation: mu10 = mu00 (no stance leak term)")
    ap.add_argument("--fusion", choices=["or", "carrier"], default="or", help="per-modality chains + OR, or one chain with carrier fusion")
    ap.add_argument("--arm", choices=["m2", "full", "lexi"], default="m2")
    ap.add_argument("--vlevel", choices=["joint", "mix"], default="joint", help="video posterior: joint model (round 1) or mixture over verdict + mean reads")
    ap.add_argument("--vtemper", choices=["none", "icc"], default="none", help="video level: reads counted as ICC-effective reads")
    ap.add_argument("--kappa", type=float, default=1.0, help="diagnostic: time-level emissions tempered at inference")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-root", default=str(ROOT / "runs/20260926_twolevel"))
    ap.add_argument("--gt-dir", default=str(ROOT / "data/gt_4fps"))
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    out = Path(a.out_root) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "run.log", "w")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line); logf.write(line + "\n"); logf.flush()

    log(f"host {socket.gethostname()}")
    (out / "run.pid").write_text(str(os.getpid()))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    log(f"code experiments/20260926_twolevel/twolevel_r2.py at commit {commit} (plus uncommitted changes if any)")
    if a.selftest:
        worst = selftest()
        log(f"selftest explicit-duration forward-backward vs brute force and vs round 1 (k = 1): max abs difference {worst:.2e}")
        if worst > 1e-8:
            raise SystemExit("SELFTEST_FAILED")
    flags = {"k": a.k, "d_gap": a.d_gap, "d_hate": a.d_hate, "sharedchain": a.sharedchain, "carrier": a.fusion == "carrier",
             "nocoupling": a.nocoupling, "noleak": a.noleak}
    run = load_run(a.run)
    videos = {ds: [prep(r, a.center) for k, r in sorted(run.items()) if k[0] == ds] for ds in a.datasets}
    params = {}
    for ds in a.datasets:
        P = em(videos[ds], flags, log=lambda m, ds=ds: log(f"[{ds}] {m}"))
        if a.vtemper == "icc":
            P["icc"] = residual_icc(videos[ds], P, flags)
            log(f"[{ds}] residual ICC " + "  ".join(f"{m} {x:.3f}" for m, x in P["icc"].items()))
        if a.vlevel == "mix":
            P["vmix_fn"], info = vlevel_mixture(videos[ds])
            P["vmix"] = info
            log(f"[{ds}] video mixture: pi {info['pi']:.3f} violating means {np.round(info['mu_violating'], 2).tolist()} "
                f"other {np.round(info['mu_other'], 2).tolist()} sd {np.round(np.sqrt(info['var']), 2).tolist()}")
        params[ds] = P
        if P.get("carrier") is not None:
            log(f"[{ds}] carrier visual / speech / both {np.round(P['carrier'], 3).tolist()}")
        log(f"[{ds}] pi {P['pi']:.3f} verdict {P['m0']:.2f}/{P['m1']:.2f} sd {math.sqrt(P['t2']):.2f}  iota " +
            " ".join(f"{'+'.join(k_)} {v_:.3f}" for k_, v_ in P["iota"].items()) + "  " +
            "  ".join(f"{m}: mu00 {e['mu00']:.2f} mu10 {e['mu10']:.2f} mu11 {e['mu11']:.2f} sd {math.sqrt(e['s2']):.2f} "
                      f"slope {(e['mu11'] - e['mu10']) / e['s2']:.3f}" for m, e in P["emit"].items()))
    pred_path = out / "predictions.jsonl"
    with open(pred_path, "w") as fh:
        for ds in a.datasets:
            for v in videos[ds]:
                t = video_terms(v, params[ds], flags, a.kappa)
                miss = np.ones(v["n"])
                for mods, ch, ll, g in t["per"]:
                    miss *= 1.0 - p_hate(g, ch)
                p = np.clip(1.0 - miss, 1e-12, 1 - 1e-12)
                lo = params[ds]["vmix_fn"](v) if a.vlevel == "mix" else (t["lo_icc"] if a.vtemper == "icc" else t["lo"])
                intervals = []
                if a.arm == "m2":
                    score = intercept(v) + centered_rank(to_frames(np.log(p) - np.log1p(-p), v["L"]))
                elif a.arm == "lexi":
                    score = lo + centered_rank(to_frames(np.log(p) - np.log1p(-p), v["L"]))
                else:
                    score = float(log_expit(lo)) + np.log(to_frames(p, v["L"]))
                    intervals = intervals_from(np.exp(score))
                fh.write(json.dumps({**v["rec"], "method": f"twolevel_r2__{a.tag}", "score_curve": [float(x) for x in score],
                                     "intervals": intervals, "extra": {"z_video": v["zv"], "p_video_logodds": lo,
                                                                "cell_prob": [float(x) for x in p]}}) + "\n")
    metrics_path = out / "metrics.json"
    subprocess.run([sys.executable, str(ROOT / "src/eval/evaluate_four_datasets.py"), "--predictions", str(pred_path),
                    "--gt-dir", a.gt_dir, "--out", str(metrics_path), "--datasets", *a.datasets],
                   check=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)}, stdout=subprocess.DEVNULL)
    for P in params.values():
        P.pop("vmix_fn", None)
    (out / "params.json").write_text(json.dumps({ds: {**P, "iota": {"+".join(k_): v_ for k_, v_ in P["iota"].items()}}
                                                 for ds, P in params.items()}, indent=2))
    (out / "config.json").write_text(json.dumps({**vars(a), "flags": flags, "cell_s": CELL}, indent=2))
    d = json.load(open(metrics_path))
    log(f"{a.tag:26s} " + "  ".join(f"{p_['dataset'][:6]} {p_['frame_ROC_AUC']:.4f}/{p_['frame_PR_AUC']:.4f}/"
                                    f"{p_['within_video_macro_ROC_AUC']:.4f} F1@.3/.5/.7 {p_['interval_F1@0.3']:.3f}/"
                                    f"{p_['interval_F1@0.5']:.3f}/{p_['interval_F1@0.7']:.3f}" for p_ in d["per_dataset"]))
    log("RUN_DONE")


if __name__ == "__main__":
    main()
