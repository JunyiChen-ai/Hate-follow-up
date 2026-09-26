#!/usr/bin/env python3
"""Two-level probabilistic model over cached SPVL-r2 reads (CPU). See README.md in this directory.

Video level: V in {0, 1}; the verdict read z_video ~ N(m_V, tau^2).
Time level (V = 1): one binary Markov chain per modality on 4 s cells; a moment is hateful if any chain is in the
hate state; at least one hateful cell overall. Each 8 s window's read of modality m observes whether chain m is
hateful anywhere in the window's cells (pair states). Emissions: N(mu11, s2) hateful there, N(mu10, s2) not
hateful in a violating video, N(mu00, s2) in a non-violating video. Windows without speech force the speech-only
chain to 0 there. Parameters by EM over the unlabeled reads of a corpus (or of both corpora pooled).
No labels are read here; evaluation only through src/eval/evaluate_four_datasets.py.
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
from scipy.special import expit, log_expit, logsumexp
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
FPS = 4.0
CELL = 4.0
MODS = ("z_visual", "z_speech")
ANY = np.array([0, 1, 1, 1])   # pair state s = 2*h_prev + h_cur: any hate in the two cells
CUR = np.array([0, 1, 0, 1])   # hate in the current cell
EPS = 1e-6


# ----------------------------------------------------------------------------------------------- data

def load_run(path):
    rows = [json.loads(l) for l in open(Path(path) / "predictions.jsonl") if l.strip()]
    return {(r["dataset"], r["video_id"]): r for r in rows if not r.get("error")}


def prep(rec):
    dur = float(rec["duration"])
    n = max(1, int(math.ceil(dur / CELL - 1e-9)))
    wins = []
    for w in rec["extra"]["windows"]:
        ks = [k for k in range(n) if min(w["end"], (k + 1) * CELL) - max(w["start"], k * CELL) > 1e-6]
        if not ks:
            ks = [min(n - 1, int(w["start"] // CELL))]
        if len(ks) > 2 or (len(ks) == 2 and ks[1] != ks[0] + 1):
            raise SystemExit(f"{rec['video_id']}: window covers cells {ks}")
        wins.append({"cell": ks[-1], "pair": len(ks) == 2, "y": {m: float(w[m]) for m in MODS if m in w},
                     "z": float(w["z"])})
    return {"key": (rec["dataset"], rec["video_id"]), "n": n, "wins": wins, "zv": float(rec["extra"]["z_video"]),
            "L": len(rec["score_curve"]), "rec": rec}


# ----------------------------------------------------------------------------------------------- chains

def lognorm(y, mu, s2):
    return -0.5 * (math.log(2 * math.pi * s2) + (y - mu) ** 2 / s2)


def emission(v, chain, P):
    """log emission E[c, s] over pair states for a chain (tuple of modalities) under V = 1."""
    E = np.zeros((v["n"], 4))
    force = chain == ("z_speech",) and not P.get("noforce", False)
    for w in v["wins"]:
        c, on = w["cell"], (ANY if w["pair"] else CUR)
        ys = [(m, w["y"][m]) for m in chain if m in w["y"]]
        if not ys:
            if force:
                E[c, on == 1] = -np.inf
            continue
        for m, y in ys:
            p = P["emit"][m]
            if "lin_scale" in p:      # diagnostic: log-likelihood ratio y / corpus std, as the current method
                E[c] += np.where(on == 1, y / p["lin_scale"], 0.0)
            else:
                E[c] += np.where(on == 1, lognorm(y, p["mu11"], p["s2"]), lognorm(y, p["mu10"], p["s2"]))
    return E


def trans_matrix(a, b):
    T = np.log(np.array([[a, 1 - a], [1 - b, b]]))
    A = np.full((4, 4), -np.inf)
    for s in range(4):
        hb = s & 1
        for s2 in range(4):
            if (s2 >> 1) == hb:
                A[s, s2] = T[hb, s2 & 1]
    return A


def fb(E, iota, a, b):
    """Forward-backward over pair states (h_{-1} = 0). Returns log-likelihood, posterior over pair states,
    log joint of the all-zero path with the reads, log prior of the all-zero path."""
    n = E.shape[0]
    A = trans_matrix(a, b)
    f = np.full((n, 4), -np.inf)
    f[0, 0] = math.log(1 - iota) + E[0, 0]
    f[0, 1] = math.log(iota) + E[0, 1]
    for c in range(1, n):
        f[c] = E[c] + logsumexp(f[c - 1][:, None] + A, axis=0)
    ll = float(logsumexp(f[-1]))
    bk = np.zeros((n, 4))
    for c in range(n - 2, -1, -1):
        bk[c] = logsumexp(A + (E[c + 1] + bk[c + 1])[None, :], axis=1)
    with np.errstate(invalid="ignore"):
        g = np.exp(f + bk - ll)
    g = np.nan_to_num(g, nan=0.0)
    logp0 = math.log(1 - iota) + (n - 1) * math.log(a)
    logz = logp0 + float(E[:, 0].sum())
    return ll, g, logz, logp0


def chain_stats(v, g):
    n = v["n"]
    p_cur = g[:, 1] + g[:, 3]
    N = np.zeros((2, 2))
    for c in range(1, n):
        for s in range(4):
            N[s >> 1, s & 1] += g[c, s]
    p_any = [float(g[w["cell"], 1:].sum() if w["pair"] else p_cur[w["cell"]]) for w in v["wins"]]
    return {"p_cur": p_cur, "N": N, "g0": float(p_cur[0]), "p_any": p_any}


# ----------------------------------------------------------------------------------------------- model

def spec_chains(flags):
    return [("z_visual", "z_speech")] if flags["sharedchain"] else [("z_visual",), ("z_speech",)]


def init_params(videos, flags):
    zs = np.array([v["zv"] for v in videos])
    P = {"pi": 0.5, "m0": float(np.percentile(zs, 10)), "m1": float(np.percentile(zs, 90)), "t2": float(zs.var()),
         "emit": {}, "chain": {}}
    for m in MODS:
        ys = np.array([w["y"][m] for v in videos for w in v["wins"] if m in w["y"]])
        mu00, mu10, mu11 = (float(np.percentile(ys, q)) for q in (10, 50, 90))
        if flags["noleak"]:
            mu10 = mu00
        P["emit"][m] = {"mu00": mu00, "mu10": mu10, "mu11": mu11, "s2": float(ys.var())}
    for ch in spec_chains(flags):
        d = flags.get("fixdwell") or 80.0
        P["chain"][ch] = {"iota": 0.5, "a": 0.5, "b": 0.5} if flags["nocoupling"] else \
            {"iota": 0.5, "a": 1 - CELL / d, "b": 1 - CELL / d}
    P["noforce"] = bool(flags.get("noforce"))
    return P


def video_terms(v, P, flags):
    """Everything the E-step and the scoring need for one video."""
    chains = spec_chains(flags)
    lv = [lognorm(v["zv"], P["m0"], P["t2"]), lognorm(v["zv"], P["m1"], P["t2"])]
    l0 = sum(lognorm(y, P["emit"][m]["mu00"], P["emit"][m]["s2"]) for w in v["wins"] for m, y in w["y"].items())
    per = []
    for ch in chains:
        c = P["chain"][ch]
        ll, g, logz, logp0 = fb(emission(v, ch, P), c["iota"], c["a"], c["b"])
        per.append((ch, ll, g, logz, logp0))
    ll_u = sum(x[1] for x in per)
    logq = sum(x[3] - x[1] for x in per)                   # log P(all chains all-zero | reads, V=1, unconstrained)
    one_minus_q = -math.expm1(min(logq, 0.0))
    if flags["noatleast"]:
        l1, norm = ll_u, 1.0
    else:
        log_not0_prior = math.log(max(-math.expm1(sum(x[4] for x in per)), 1e-300))
        l1 = ll_u + math.log(max(one_minus_q, 1e-300)) - log_not0_prior
        norm = one_minus_q
    lo_verdict = math.log(P["pi"]) - math.log(1 - P["pi"]) + lv[1] - lv[0]
    lo_reads = l1 - l0
    total = float(np.logaddexp(math.log(P["pi"]) + lv[1] + l1, math.log(1 - P["pi"]) + lv[0] + l0))
    lo = lo_verdict + lo_reads
    return {"per": per, "l0": l0, "l1": l1, "lo": lo, "lo_verdict": lo_verdict, "lo_reads": lo_reads,
            "norm": norm, "logq": logq, "total": total}


def estep(videos, P, flags):
    acc = {"w": [], "wz": [], "emit": {m: np.zeros((3, 3)) for m in MODS},   # rows 00/10/11: sum w, sum w y, sum w y2
           "chain": {ch: {"N": np.zeros((2, 2)), "g0": 0.0, "w": 0.0, "hot": 0.0, "cells": 0.0} for ch in spec_chains(flags)},
           "vid": []}                                                         # per video: (w, n, {chain: (g0, N, hot)})
    total, skipped = 0.0, 0
    for v in videos:
        t = video_terms(v, P, flags)
        total += t["total"]
        w = float(expit(t["lo"]))
        acc["w"].append(w); acc["wz"].append(v["zv"])
        for wi in v["wins"]:
            for m, y in wi["y"].items():
                acc["emit"][m][0] += (1 - w) * np.array([1.0, y, y * y])
        if t["norm"] < 1e-8:          # V = 1 posterior numerically undefined; its weight is negligible
            skipped += 1
            continue
        q = 1.0 - t["norm"] if not flags["noatleast"] else 0.0
        vstats = {}
        for ch, ll, g, logz, logp0 in t["per"]:
            st = chain_stats(v, g)
            N = st["N"].copy()
            if not flags["noatleast"]:
                N[0, 0] -= q * (v["n"] - 1)
                N /= t["norm"]
            scale = 1.0 if flags["noatleast"] else 1.0 / t["norm"]
            ca = acc["chain"][ch]
            ca["N"] += w * N; ca["g0"] += w * st["g0"] * scale; ca["w"] += w
            ca["hot"] += w * float(st["p_cur"].sum()) * scale; ca["cells"] += w * v["n"]
            vstats[ch] = (st["g0"] * scale, N, float(st["p_cur"].sum()) * scale)
            for wi, pa in zip(v["wins"], st["p_any"]):
                pa = min(1.0, pa * scale)
                for m in ch:
                    if m in wi["y"]:
                        y = wi["y"][m]
                        acc["emit"][m][1] += w * (1 - pa) * np.array([1.0, y, y * y])
                        acc["emit"][m][2] += w * pa * np.array([1.0, y, y * y])
        acc["vid"].append((w, v["n"], vstats))
    return acc, total, skipped


def mstep(acc, P, flags):
    w = np.array(acc["w"]); z = np.array(acc["wz"])
    Q = {"pi": float(np.clip(w.mean(), EPS, 1 - EPS)), "emit": {}, "chain": {}, "noforce": P.get("noforce", False)}
    Q["m1"] = float((w * z).sum() / max(w.sum(), EPS)); Q["m0"] = float(((1 - w) * z).sum() / max((1 - w).sum(), EPS))
    Q["t2"] = max(float((w * (z - Q["m1"]) ** 2 + (1 - w) * (z - Q["m0"]) ** 2).mean()), EPS)
    for m in MODS:
        S = acc["emit"][m]
        mu = S[:, 1] / np.maximum(S[:, 0], EPS)
        if flags["noleak"]:
            mu[0] = mu[1] = (S[0, 1] + S[1, 1]) / max(S[0, 0] + S[1, 0], EPS)
        ss = sum(S[k, 2] - 2 * mu[k] * S[k, 1] + mu[k] ** 2 * S[k, 0] for k in range(3))
        Q["emit"][m] = {"mu00": float(mu[0]), "mu10": float(mu[1]), "mu11": float(mu[2]),
                        "s2": max(float(ss / max(S[:, 0].sum(), EPS)), EPS)}
    for ch, ca in acc["chain"].items():          # closed form (exact without the at-least-one constraint)
        if flags["nocoupling"]:
            iota = float(np.clip(ca["hot"] / max(ca["cells"], EPS), EPS, 1 - EPS))
            Q["chain"][ch] = {"iota": iota, "a": 1 - iota, "b": iota}
        else:
            N = ca["N"]
            Q["chain"][ch] = {"iota": float(np.clip(ca["g0"] / max(ca["w"], EPS), EPS, 1 - EPS)),
                              "a": float(np.clip(N[0, 0] / max(N[0].sum(), EPS), EPS, 1 - EPS)),
                              "b": float(np.clip(N[1, 1] / max(N[1].sum(), EPS), EPS, 1 - EPS))}
    if not flags["noatleast"]:
        Q["chain"] = mstep_chains_constrained(acc, Q["chain"], flags)
    if flags.get("fixdwell"):
        for ch in Q["chain"]:
            Q["chain"][ch]["a"] = Q["chain"][ch]["b"] = 1 - CELL / flags["fixdwell"]
    return Q


def mstep_chains_constrained(acc, start, flags):
    """Exact M-step of the chain parameters under the at-least-one constraint: maximise
    sum_j w_j [sum_k E log p_chain_k(h_k)] - sum_j w_j log(1 - prod_k P0_jk) numerically (logit parameters)."""
    from scipy.optimize import minimize
    chains = list(start)
    vids = acc["vid"]
    lg = lambda x: math.log(x) - math.log(1 - x)

    def unpack(x):
        out, i = {}, 0
        for ch in chains:
            if flags["nocoupling"]:
                io = float(expit(x[i])); out[ch] = {"iota": io, "a": 1 - io, "b": io}; i += 1
            else:
                out[ch] = {"iota": float(expit(x[i])), "a": float(expit(x[i + 1])), "b": float(expit(x[i + 2]))}; i += 3
        return out

    def negq(x):
        P = unpack(x); val = 0.0
        for w, n, st in vids:
            if w <= 0 or not st:
                continue
            log_p0 = 0.0
            for ch in chains:
                g0, N, hot = st[ch]; c = P[ch]
                if flags["nocoupling"]:
                    val += w * (hot * math.log(c["iota"]) + (n - hot) * math.log(1 - c["iota"]))
                    log_p0 += n * math.log(1 - c["iota"])
                else:
                    val += w * (g0 * math.log(c["iota"]) + (1 - g0) * math.log(1 - c["iota"]) +
                                N[0, 0] * math.log(c["a"]) + N[0, 1] * math.log(1 - c["a"]) +
                                N[1, 0] * math.log(1 - c["b"]) + N[1, 1] * math.log(c["b"]))
                    log_p0 += math.log(1 - c["iota"]) + (n - 1) * math.log(c["a"])
            val -= w * math.log(max(-math.expm1(log_p0), 1e-300))
        return -val

    x0 = []
    for ch in chains:
        c = start[ch]
        x0 += [lg(c["iota"])] if flags["nocoupling"] else [lg(c["iota"]), lg(c["a"]), lg(c["b"])]
    res = minimize(negq, np.array(x0), method="L-BFGS-B", bounds=[(-13.8, 13.8)] * len(x0))
    best = res.x if res.fun <= negq(np.array(x0)) else np.array(x0)
    return unpack(best)


def em(videos, flags, max_it=300, tol=1e-7, log=print):
    P = init_params(videos, flags)
    trace, prev = [], None
    for it in range(max_it):
        acc, total, skipped = estep(videos, P, flags)
        trace.append(total)
        if prev is not None:
            if total < prev - 1e-6 * abs(prev):
                raise AssertionError(f"EM log-likelihood decreased at iteration {it}: {prev} -> {total}")
            if abs(total - prev) <= tol * abs(prev):
                break
        prev = total
        P = mstep(acc, P, flags)
    log(f"EM stop after {len(trace)} iterations, log-likelihood {trace[-1]:.4f}, skipped-V1 videos {skipped}")
    return P, trace


# ----------------------------------------------------------------------------------------------- scoring

def centered_rank(v):
    v = np.asarray(v, float)
    if len(v) <= 1 or np.ptp(v) <= 1e-12:
        return np.zeros_like(v)
    r = (rankdata(v, method="average") - 0.5) / len(v) - 0.5
    return r - r.mean()


def to_frames(cell_vals, L):
    centers = (np.arange(L) + 0.5) / FPS
    idx = np.clip((centers // CELL).astype(int), 0, len(cell_vals) - 1)
    return np.asarray(cell_vals, float)[idx]


def intercept(v):
    return v["zv"] + (float(np.mean([w["z"] for w in v["wins"]])) if v["wins"] else 0.0)


def cell_prob(t, flags):
    """P(hateful cell | V = 1, reads) (constrained unless noatleast), and its log."""
    miss = np.ones(len(t["per"][0][2]))
    for ch, ll, g, logz, logp0 in t["per"]:
        miss *= 1.0 - (g[:, 1] + g[:, 3])
    p = 1.0 - miss
    if not flags["noatleast"] and t["norm"] >= 1e-8:
        p = p / t["norm"]
    return np.clip(p, 1e-12, 1 - 1e-12)


def intervals_from(prob, thr=0.5):
    on = np.r_[0, (prob >= thr).astype(np.int8), 0]
    idx = np.flatnonzero(np.diff(on)).reshape(-1, 2)
    return [[float(a) / FPS, float(b) / FPS] for a, b in idx]


def replica_current(v, scale):
    """Replica of experiments/20260922_til/til_infer.py --model average --fusion max --dwell 80 (plumbing check)."""
    n = v["n"]; p_stay = 1.0 - CELL / 80.0
    acc = np.zeros(n); cnt = np.zeros(n)
    for w in v["wins"]:
        r = max(w["y"][m] / scale[m] for m in w["y"])
        for k in ([w["cell"] - 1, w["cell"]] if w["pair"] else [w["cell"]]):
            acc[k] += r; cnt[k] += 1
    x = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)
    T = np.log(np.array([[p_stay, 1 - p_stay], [1 - p_stay, p_stay]]))
    E = np.stack([np.zeros(n), x], 1)
    f = np.zeros((n, 2)); f[0] = math.log(0.5) + E[0]
    for c in range(1, n):
        f[c] = E[c] + np.logaddexp(f[c - 1, 0] + T[0], f[c - 1, 1] + T[1])
    b = np.zeros((n, 2))
    for c in range(n - 2, -1, -1):
        b[c] = np.logaddexp(T[:, 0] + E[c + 1, 0] + b[c + 1, 0], T[:, 1] + E[c + 1, 1] + b[c + 1, 1])
    q = f + b
    return q[:, 1] - q[:, 0]


# ----------------------------------------------------------------------------------------------- self-test

def selftest(seed=0, trials=200):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        n = int(rng.integers(1, 9))
        E = rng.normal(0, 2, size=(n, 4))
        E[0, 2:] = -np.inf if rng.random() < 0.5 else E[0, 2:]
        for c in range(n):
            if rng.random() < 0.2:
                E[c, ANY == 1] = -np.inf
        iota, a, b = rng.uniform(0.05, 0.95, 3)
        ll, g, logz, logp0 = fb(E, iota, a, b)
        T = np.log(np.array([[a, 1 - a], [1 - b, b]]))
        paths, lps = list(itertools.product([0, 1], repeat=n)), []
        for h in paths:
            lp = math.log(iota if h[0] else 1 - iota) + E[0, h[0]]
            for c in range(1, n):
                lp += T[h[c - 1], h[c]] + E[c, 2 * h[c - 1] + h[c]]
            lps.append(lp)
        lps = np.array(lps); LL = logsumexp(lps); post = np.exp(lps - LL)
        marg = np.array([sum(p for p, h in zip(post, paths) if h[c]) for c in range(n)])
        z_path = lps[0]
        worst = max(worst, abs(ll - LL), float(np.abs(marg - (g[:, 1] + g[:, 3])).max()), abs(logz - z_path))
    return worst


# ----------------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "runs/20260926_glr/base_gridA"))
    ap.add_argument("--arm", required=True, choices=["current", "m2", "full", "diag_lexi"])
    ap.add_argument("--scope", choices=["corpus", "pooled"], default="corpus")
    ap.add_argument("--nocoupling", action="store_true")
    ap.add_argument("--sharedchain", action="store_true")
    ap.add_argument("--noleak", action="store_true")
    ap.add_argument("--noatleast", action="store_true")
    ap.add_argument("--video", choices=["both", "verdict", "reads"], default="both")
    ap.add_argument("--fixdwell", type=float, default=0.0, help="diagnostic: fix a = b = 1 - 4/D (transitions not estimated)")
    ap.add_argument("--noforce", action="store_true", help="diagnostic: silent windows are missing speech observations")
    ap.add_argument("--linear", action="store_true", help="diagnostic (m2 only): evidence y / corpus std, dwell 80, no EM")
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
    log(f"code experiments/20260926_twolevel/twolevel.py at commit {commit} (plus uncommitted changes if any)")
    if a.selftest:
        worst = selftest()
        log(f"selftest forward-backward vs brute force: max abs difference {worst:.2e}")
        if worst > 1e-8:
            raise SystemExit("SELFTEST_FAILED")
    flags = {"nocoupling": a.nocoupling, "sharedchain": a.sharedchain, "noleak": a.noleak, "noatleast": a.noatleast,
             "fixdwell": a.fixdwell, "noforce": a.noforce}
    run = load_run(a.run)
    videos = {ds: [prep(r) for k, r in sorted(run.items()) if k[0] == ds] for ds in a.datasets}
    params, traces = {}, {}
    if a.linear:
        for ds in a.datasets:
            P = init_params(videos[ds], {**flags, "fixdwell": 80.0})
            for m in MODS:
                P["emit"][m]["lin_scale"] = float(np.std([w["y"][m] for v in videos[ds] for w in v["wins"] if m in w["y"]]))
            params[ds] = {"_P": P, "linear": True}
    elif a.arm != "current":
        groups = {ds: videos[ds] for ds in a.datasets} if a.scope == "corpus" else {"pooled": [v for ds in a.datasets for v in videos[ds]]}
        for g, vids in groups.items():
            P, tr = em(vids, flags, log=lambda m, g=g: log(f"[{g}] {m}"))
            params[g] = {"pi": P["pi"], "m0": P["m0"], "m1": P["m1"], "t2": P["t2"], "emit": P["emit"],
                         "chain": {"+".join(k): v for k, v in P["chain"].items()}, "iterations": len(tr), "loglik": tr[-1]}
            params[g]["_P"] = P
            for m in MODS:
                e = P["emit"][m]
                if e["mu11"] <= e["mu10"]:
                    log(f"[{g}] ARM_FAILED: {m} mu11 {e['mu11']:.3f} <= mu10 {e['mu10']:.3f}")
            for ch, c in P["chain"].items():
                dw = lambda p: CELL / max(1 - p, EPS)
                log(f"[{g}] chain {'+'.join(ch)}: iota {c['iota']:.3f}  stay non-hate {c['a']:.4f} (mean dwell {dw(c['a']):.0f} s)  "
                    f"stay hate {c['b']:.4f} (mean dwell {dw(c['b']):.0f} s)")
            log(f"[{g}] pi {P['pi']:.3f}  verdict means {P['m0']:.2f} / {P['m1']:.2f} sd {math.sqrt(P['t2']):.2f}  " +
                "  ".join(f"{m}: mu00 {e['mu00']:.2f} mu10 {e['mu10']:.2f} mu11 {e['mu11']:.2f} sd {math.sqrt(e['s2']):.2f}"
                          for m, e in P["emit"].items()))
    scale = {}
    for ds in a.datasets:
        for m in MODS:
            vals = [w["y"][m] for v in videos[ds] for w in v["wins"] if m in w["y"]]
            scale[(ds, m)] = float(np.std(vals))
    pred_path = out / "predictions.jsonl"
    with open(pred_path, "w") as fh:
        for ds in a.datasets:
            for v in videos[ds]:
                rec = v["rec"]; extra = {"z_video": v["zv"]}
                if a.arm == "current":
                    lo = replica_current(v, {m: scale[(ds, m)] for m in MODS})
                    score = intercept(v) + centered_rank(to_frames(lo, v["L"]))
                    intervals = []
                else:
                    P = params[ds if a.scope == "corpus" else "pooled"]["_P"]
                    t = video_terms(v, P, flags)
                    p = cell_prob(t, flags)
                    if a.arm == "m2":
                        score = intercept(v) + centered_rank(to_frames(np.log(p) - np.log1p(-p), v["L"]))
                        intervals = []
                    elif a.arm == "diag_lexi":   # diagnostic: Bayes video log-odds as the intercept, within rank as now
                        score = t["lo"] + centered_rank(to_frames(np.log(p) - np.log1p(-p), v["L"]))
                        intervals = []
                    else:
                        lo = {"both": t["lo"], "verdict": t["lo_verdict"], "reads": t["lo_reads"]}[a.video]
                        logp = float(log_expit(lo)) + np.log(to_frames(p, v["L"]))
                        score = logp
                        intervals = intervals_from(np.exp(logp))
                    extra.update({"p_video_logodds": t["lo"], "lo_verdict": t["lo_verdict"], "lo_reads": t["lo_reads"],
                                  "cell_prob": [float(x) for x in p]})
                fh.write(json.dumps({**rec, "method": f"twolevel__{a.tag}", "score_curve": [float(x) for x in score],
                                     "intervals": intervals, "extra": extra}) + "\n")
    metrics_path = out / "metrics.json"
    subprocess.run([sys.executable, str(ROOT / "src/eval/evaluate_four_datasets.py"), "--predictions", str(pred_path),
                    "--gt-dir", a.gt_dir, "--out", str(metrics_path), "--datasets", *a.datasets],
                   check=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)}, stdout=subprocess.DEVNULL)
    for g in params.values():
        g.pop("_P", None)
    (out / "params.json").write_text(json.dumps(params, indent=2))
    (out / "config.json").write_text(json.dumps({**vars(a), "flags": flags, "cell_s": CELL,
                                                 "scale_current": {f"{k[0]}/{k[1]}": s for k, s in scale.items()}}, indent=2))
    d = json.load(open(metrics_path))
    log(f"{a.tag:22s} " + "  ".join(
        f"{p['dataset'][:6]} {p['frame_ROC_AUC']:.4f}/{p['frame_PR_AUC']:.4f}/{p['within_video_macro_ROC_AUC']:.4f}"
        f" F1@.3/.5/.7 {p['interval_F1@0.3']:.3f}/{p['interval_F1@0.5']:.3f}/{p['interval_F1@0.7']:.3f}"
        for p in d["per_dataset"]))
    log("RUN_DONE")


if __name__ == "__main__":
    main()
