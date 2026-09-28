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

# Chain kinds (experiments/20260928_infer/README.md): "two" = the r6 chain (gap / hate); "three_nested" and
# "three_free" = off topic / topic without attack / attack (§2); "joint" = one segmentation whose segments carry a
# joint (visual, speech) label (§3). A chain is described by its phases: each phase has k sub-states in a row, a mean
# length (gap or hate family), a hate bit and, per read modality, an observation level. The augmented state is
# (previous cell's level id, sub-state), so that a window covering two cells observes the higher level of the two.
KINDS = {
    # phase -> (length family, hate bit, {modality: level}); "level" indexes the emitter's list of means
    "two": {"phases": [("gap", 0, 0), ("hate", 1, 1)], "next": {0: {1: 1.0}, 1: {0: 1.0}}},
    "three_nested": {"phases": [("gap", 0, 0), ("gap", 0, 1), ("hate", 1, 2)],
                     "next": {0: {1: 1.0}, 1: {0: 0.5, 2: 0.5}, 2: {1: 1.0}}},
    "three_free": {"phases": [("gap", 0, 0), ("gap", 0, 1), ("hate", 1, 2)],
                   "next": {0: {1: .5, 2: .5}, 1: {0: .5, 2: .5}, 2: {0: .5, 1: .5}}},
    # joint label (visual bit, speech bit): 00, 01, 10, 11; levels are per modality
    "joint": {"phases": [("gap", 0, (0, 0)), ("hate", 1, (0, 1)), ("hate", 1, (1, 0)), ("hate", 1, (1, 1))],
              "next": {p: {q: 1 / 3 for q in range(4) if q != p} for p in range(4)}},
}


def build_chain(k, d_gap, d_hate, iota, nocoupling=False, kind="two", mods=None):
    """Explicit-duration chain of README §10 / §16, generalised to the kinds above. iota: start distribution over
    phases (a scalar is P(hate) of the two-phase chain). mods: the modalities this chain emits (joint kind: both).
    Returns sub-state matrix A, augmented transition matrix T over (previous level id, sub-state), start pi0, per
    augmented state the hate bit and, per modality, the observation level for single-cell and two-cell windows.
    nocoupling (ablation, two-phase): k = 1 and independent cells with P(hate) = iota."""
    spec = KINDS[kind]
    n_ph = len(spec["phases"])
    io = np.array([1 - iota, iota]) if np.ndim(iota) == 0 else np.asarray(iota, float)
    if nocoupling:
        k = 1
    S = n_ph * k
    ph = np.repeat(np.arange(n_ph), k)                      # phase of each sub-state
    hb = np.array([spec["phases"][p][1] for p in ph])
    A = np.zeros((S, S))
    for x in range(S):
        if nocoupling:
            A[x] = io[ph] / k
            continue
        fam = spec["phases"][ph[x]][0]
        q = 1.0 - k * CELL / (d_hate if fam == "hate" else d_gap)
        q = min(max(q, 0.0), 1 - 1e-9)
        A[x, x] = q
        if (x + 1) % k:                                     # next sub-state of the same phase
            A[x, x + 1] += 1.0 - q
        else:                                               # last sub-state: leave to the first sub-state of a next phase
            for p2, pr in spec["next"][ph[x]].items():
                A[x, p2 * k] += (1.0 - q) * pr
    # level ids carried to the next cell: the hate bit for the two-phase chain (as in round 2), else the phase
    plv = hb if kind == "two" else ph
    n_plv = 2 if kind == "two" else n_ph
    T = np.zeros((n_plv * S, n_plv * S))
    for bp in range(n_plv):
        for xp in range(S):
            T[bp * S + xp, plv[xp] * S:plv[xp] * S + S] = A[xp]
    pi0 = np.zeros(n_plv * S)
    pi0[:S] = io[ph] / k                                    # first cell: previous level id 0, start uniform in the phase
    mods = tuple(mods) if mods else tuple(MODS)
    lvl = {}
    for j, m in enumerate(mods):
        cur = np.array([spec["phases"][p][2][j] if kind == "joint" else spec["phases"][p][2] for p in ph])
        prev = np.array([spec["phases"][p][2][j] if kind == "joint" else spec["phases"][p][2]
                         for p in (range(n_ph) if kind != "two" else (0, 1))])   # level of each previous-level id
        single = np.tile(cur, n_plv)
        pair = np.maximum(np.repeat(prev, S), single)
        lvl[m] = {"single": single, "pair": pair}
    return {"k": k, "S": S, "n_plv": n_plv, "n_ph": n_ph, "kind": kind, "hb": hb, "A": A, "plv": plv, "ph": ph,
            "T": T, "pi0": pi0, "hate": np.tile(hb, n_plv), "phase": np.tile(ph, n_plv), "lvl": lvl,
            "n_lvl": {m: int(max(v["pair"].max(), v["single"].max())) + 1 for m, v in lvl.items()}}


def on_masks(ch):
    """Two-level view (carrier and linear paths): a window's read is 'on' when its level is above 0."""
    m = next(iter(ch["lvl"]))
    return {"pair": (ch["lvl"][m]["pair"] > 0).astype(int), "single": (ch["lvl"][m]["single"] > 0).astype(int)}


def ek(w, m):
    """Emitter key of modality m in window w: the modality, or modality|condition (README of 20260928_infer §1)."""
    c = w.get("cond", {}).get(m)
    return m if c is None else f"{m}|{c}"


def p_levels(v, g, ch, m):
    """Per window, the posterior over modality m's observation levels (n_windows, n_levels)."""
    L = ch["n_lvl"][m]
    out = np.zeros((len(v["wins"]), L))
    for i, w in enumerate(v["wins"]):
        lv = ch["lvl"][m]["pair" if w["pair"] else "single"]
        for l_ in range(L):
            out[i, l_] = g[w["cell"], lv == l_].sum()
    return out


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
    E = np.zeros((v["n"], ch["n_plv"] * ch["S"]))
    if P.get("linear"):
        for w in v["wins"]:
            on = masks["pair"] if w["pair"] else masks["single"]
            ys = [w["y"][m] / P["linear"][m] for m in mods if m in w["y"]]
            if ys:
                E[w["cell"]] += kappa * np.where(on == 1, max(ys), 0.0)
        return E
    if P.get("carrier") is not None:
        lc = np.log(np.asarray(P["carrier"]))
        for w in v["wins"]:
            on = masks["pair"] if w["pair"] else masks["single"]
            ons, off = carrier_terms(w, P)
            E[w["cell"]] += kappa * np.where(on == 1, float(np.logaddexp.reduce(lc + ons)), off)
        return E
    for w in v["wins"]:
        for m in mods:
            if m in w["y"]:
                e = P["emit"][ek(w, m)]
                lv = ch["lvl"][m]["pair" if w["pair"] else "single"]
                E[w["cell"]] += kappa * lognorm(w["y"][m], np.asarray(e["mu"], float)[lv], e["s2"])
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


def bma_fb(E, chains, logw):
    """Model averaging over chains that share one state space (README §16): log p(E) = logsumexp_j(logw_j + ll_j),
    posterior = sum_j w_j g_j with w_j proportional to exp(logw_j + ll_j). Returns (log p, posterior, w)."""
    lls, gs = [], []
    for ch in chains:
        ll, g = fb(E, ch)
        lls.append(ll); gs.append(g)
    a = np.asarray(logw, float) + np.asarray(lls)
    LL = float(np.logaddexp.reduce(a))
    w = np.exp(a - LL)
    return LL, np.tensordot(w, np.stack(gs), axes=1), w


def length_grid(n_cells, k, G, fixed=None):
    """Mean segment lengths (seconds) averaged over for one video (README §16): G values log-spaced from the smallest
    possible mean, k cells, to the video's length. `fixed` pins a single value (plumbing check only)."""
    if fixed:
        return np.array([float(fixed)])
    lo, hi = k * CELL, max(n_cells * CELL, k * CELL)
    return np.array([lo]) if hi <= lo * (1 + 1e-9) or G == 1 else np.geomspace(lo, hi, G)


def corpus_combine(A, B, Lc):
    """Round 5 (README §18): one (gap, hate) pair per chain shared by all videos, uniform prior over each chain's J
    pairs. A[v] = log pi + log p(verdict | hateful), B[v] = log(1 - pi) + log p(verdict | not) + log p(reads | not),
    Lc[i][v, j] = log p(chain i's reads of video v | hateful, pair j). Returns the corpus log-likelihood, the
    posterior over joint pairs (J, ..., J) and M[v, j...] = P(joint pair, video v hateful | all reads)."""
    C = len(Lc); nv, J = Lc[0].shape
    X = np.asarray(A, float).reshape((nv,) + (1,) * C)
    for i, L in enumerate(Lc):
        X = X + L.reshape((nv,) + tuple(J if d == i else 1 for d in range(C)))
    Y = np.logaddexp(X, np.asarray(B, float).reshape((nv,) + (1,) * C))
    lp = Y.sum(0) - C * math.log(J)
    LL = float(np.logaddexp.reduce(lp.ravel()))
    w = np.exp(lp - LL)
    return LL, w, w[None] * np.exp(X - Y)


def corpus_terms(videos, P, flags, kappa=1.0):
    """Round 5 (README §18). Per video: P(V = 1 | all reads) and, per chain, the pair-averaged P(hate) per cell,
    P(on) per window and P(hate) in the first cell, given V = 1. Also the corpus log-likelihood and the per-chain
    posterior over pairs."""
    grid, cs = P["grid"], chains_of(flags)
    A, B, L, proj = [], [], [[] for _ in cs], []
    for v in videos:
        lv0 = lognorm(v["zv"], P["m0"], P["t2"]); lv1 = lognorm(v["zv"], P["m1"], P["t2"])
        l0 = sum(lognorm(y, P["emit"][m]["mu00"], P["emit"][m]["s2"]) for w in v["wins"] for m, y in w["y"].items())
        A.append(math.log(P["pi"]) + lv1); B.append(math.log(1 - P["pi"]) + lv0 + l0)
        pv = []
        for i, mods in enumerate(cs):
            chs = [build_chain(flags["k"], dg, dh, P["iota"][mods]) for dg in grid for dh in grid]
            E = emission(v, mods, P, chs[0], kappa)
            lls, ph, pon = [], [], []
            for ch in chs:
                ll, g = fb(E, ch)
                lls.append(ll); ph.append(p_hate(g, ch)); pon.append(p_on(v, g, ch))
            L[i].append(lls); pv.append((chs[0], np.array(ph), np.array(pon).reshape(len(chs), -1)))
        proj.append(pv)
    LL, w, M = corpus_combine(A, B, [np.array(x) for x in L])
    C = len(cs); others = lambda i: tuple(d for d in range(C) if d != i)
    out = []
    for n_, v in enumerate(videos):
        Wv = float(M[n_].sum()); per = []
        for i, mods in enumerate(cs):
            a = M[n_].sum(axis=others(i)) if C > 1 else M[n_]
            a = a / a.sum() if a.sum() > 1e-300 else (w.sum(axis=others(i)) if C > 1 else w)
            ch, ph, pon = proj[n_][i]
            per.append((mods, ch, a @ ph, a @ pon))
        out.append({"W": Wv, "per": per})
    marg = {mods: (w.sum(axis=others(i)) if C > 1 else w) for i, mods in enumerate(cs)}
    return LL, marg, out


def p_hate(g, ch):
    return g[:, ch["hate"] == 1].sum(1)


def p_phase(g_cells, ch):
    """Posterior over phases, summed over the given cells' rows of g: shape (n_phases,)."""
    return np.array([g_cells[:, ch["phase"] == p].sum() for p in range(ch["n_ph"])])


def p_on(v, g, ch):
    m = next(iter(ch["lvl"]))
    return [float(1.0 - x) for x in p_levels(v, g, ch, m)[:, 0]]


def selftest(trials=100, seed=0):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for kind in KINDS:                           # every chain kind vs brute force over sub-state paths
        n_ph = len(KINDS[kind]["phases"])
        for _ in range(trials if kind == "two" else trials // 2):
            k = int(rng.integers(1, 3)); n = int(rng.integers(1, 5 if kind == "joint" else 6))
            io = rng.dirichlet(np.ones(n_ph))
            ch = build_chain(k, rng.uniform(4 * k + 1, 60), rng.uniform(4 * k + 1, 60), io, False, kind, MODS)
            S = ch["S"]; E = rng.normal(0, 2, size=(n, ch["n_plv"] * S))
            ll, g = fb(E, ch)
            marg = np.zeros(n); lps = []; paths = list(itertools.product(range(S), repeat=n))
            A = ch["A"]
            for xs in paths:
                lp = math.log(max(ch["pi0"][xs[0]], 1e-300)) + E[0, xs[0]]
                for c in range(1, n):
                    t = A[xs[c - 1], xs[c]]
                    lp += math.log(t) if t > 0 else -np.inf
                    lp += E[c, ch["plv"][xs[c - 1]] * S + xs[c]]
                lps.append(lp)
            lps = np.array(lps); LL = np.logaddexp.reduce(lps); post = np.exp(lps - LL)
            for c in range(n):
                marg[c] = sum(p for p, xs in zip(post, paths) if ch["hb"][xs[c]])
            worst = max(worst, abs(ll - LL), float(np.abs(marg - p_hate(g, ch)).max()))
            # the observation levels: a two-cell window sees the higher level of its two cells
            for m in MODS:
                lv = ch["lvl"][m]
                for z in range(ch["n_plv"] * S):
                    bp, x = divmod(z, S)
                    spec = KINDS[kind]["phases"]; j = MODS.index(m)
                    cur = spec[ch["ph"][x]][2][j] if kind == "joint" else spec[ch["ph"][x]][2]
                    pl = [spec[p][2][j] if kind == "joint" else spec[p][2] for p in range(n_ph)]
                    prv = (0, 1)[bp] if kind == "two" else pl[bp]
                    assert lv["single"][z] == cur and lv["pair"][z] == max(prv, cur), (kind, m, z)
    import twolevel as r1
    for _ in range(trials):                      # k = 1 equals round 1's geometric pair-state chain
        n = int(rng.integers(1, 30)); dg, dh, io = rng.uniform(5, 200), rng.uniform(5, 200), rng.uniform(.05, .95)
        E = rng.normal(0, 2, size=(n, 4))
        ll, g = fb(E, build_chain(1, dg, dh, io))
        ll1, g1, _, _ = r1.fb(E, io, 1 - CELL / dg, 1 - CELL / dh)
        worst = max(worst, abs(ll - ll1), float(np.abs(p_hate(g, build_chain(1, dg, dh, io)) - (g1[:, 1] + g1[:, 3])).max()))
    for _ in range(trials):                      # model averaging (README §16) vs brute force over (chain, path)
        k = int(rng.integers(1, 3)); n = int(rng.integers(1, 5)); io = rng.uniform(.05, .95)
        chs = [build_chain(k, rng.uniform(4 * k + 1, 60), rng.uniform(4 * k + 1, 60), io) for _ in range(3)]
        logw = np.log(rng.dirichlet(np.ones(3)))
        E = rng.normal(0, 2, size=(n, 2 * chs[0]["S"]))
        LL, g, _ = bma_fb(E, chs, logw)
        S = chs[0]["S"]; paths = list(itertools.product(range(S), repeat=n)); lps, owner = [], []
        for j, ch in enumerate(chs):
            A = ch["T"][:S, :S] + ch["T"][:S, S:]
            for xs in paths:
                lp = logw[j] + math.log(max(ch["pi0"][xs[0]], 1e-300)) + E[0, xs[0]]
                for c in range(1, n):
                    t = A[xs[c - 1], xs[c]]
                    lp += (math.log(t) if t > 0 else -np.inf) + E[c, ch["hb"][xs[c - 1]] * S + xs[c]]
                lps.append(lp); owner.append(xs)
        lps = np.array(lps); LLb = np.logaddexp.reduce(lps); post = np.exp(lps - LLb)
        marg = np.array([sum(p for p, xs in zip(post, owner) if chs[0]["hb"][xs[c]]) for c in range(n)])
        worst = max(worst, abs(LL - LLb), float(np.abs(marg - p_hate(g, chs[0])).max()))

    def paths(E, ch):                              # brute force over one chain's paths: log p and P(hate) per cell
        S = ch["S"]; A = ch["T"][:S, :S] + ch["T"][:S, S:]; xs_all = list(itertools.product(range(S), repeat=E.shape[0]))
        lps = []
        for xs in xs_all:
            lp = math.log(max(ch["pi0"][xs[0]], 1e-300)) + E[0, xs[0]]
            for c in range(1, E.shape[0]):
                t = A[xs[c - 1], xs[c]]
                lp += (math.log(t) if t > 0 else -np.inf) + E[c, ch["hb"][xs[c - 1]] * S + xs[c]]
            lps.append(lp)
        lps = np.array(lps); LLp = np.logaddexp.reduce(lps); post = np.exp(lps - LLp)
        return LLp, np.array([sum(p for p, xs in zip(post, xs_all) if ch["hb"][xs[c]]) for c in range(E.shape[0])])
    for _ in range(trials // 5):                   # corpus-shared pairs (README §18) vs brute force over (pair, V, paths)
        k = int(rng.integers(1, 3)); grid = rng.uniform(4 * k + 1, 60, size=2); J = 4
        ios = rng.uniform(.05, .95, size=2); nv = 3
        chs = [[build_chain(k, dg, dh, ios[i]) for dg in grid for dh in grid] for i in range(2)]
        A = rng.normal(0, 2, nv); B = rng.normal(0, 2, nv); ns = rng.integers(1, 4, nv)
        Es = [[rng.normal(0, 2, size=(ns[v], 2 * chs[0][0]["S"])) for _ in range(2)] for v in range(nv)]
        Lc = [np.array([[fb(Es[v][i], ch)[0] for ch in chs[i]] for v in range(nv)]) for i in range(2)]
        LL, w, M = corpus_combine(A, B, Lc)
        br = [[[paths(Es[v][i], ch) for ch in chs[i]] for i in range(2)] for v in range(nv)]
        lp = np.zeros((J, J)); r = np.zeros((nv, J, J))
        for j1 in range(J):
            for j2 in range(J):
                x = [A[v] + br[v][0][j1][0] + br[v][1][j2][0] for v in range(nv)]
                lp[j1, j2] = -2 * math.log(J) + sum(np.logaddexp(x[v], B[v]) for v in range(nv))
                r[:, j1, j2] = [1 / (1 + math.exp(B[v] - x[v])) for v in range(nv)]
        LLb = float(np.logaddexp.reduce(lp.ravel())); wb = np.exp(lp - LLb)
        worst = max(worst, abs(LL - LLb), float(np.abs(M - wb[None] * r).max()))
        for v in range(nv):                        # P(hate) per cell of chain 0 given V = 1, averaged over pairs
            a = (wb * r[v]).sum(1); a = a / a.sum()
            fast = sum(a[j] * p_hate(fb(Es[v][0], chs[0][j])[1], chs[0][j]) for j in range(J))
            slow = sum(a[j] * br[v][0][j][1] for j in range(J))
            worst = max(worst, float(np.abs(fast - slow).max()))
    return worst


# ----------------------------------------------------------------------------------------------- EM

def chains_of(flags):
    return [tuple(MODS)] if flags.get("sharedchain") or flags.get("carrier") or flags.get("kind") == "joint" \
        else [(m,) for m in MODS]


def emitters(videos):
    """All emitter keys present in the videos, per modality (README of 20260928_infer §1: modality|condition)."""
    out = {m: sorted({ek(w, m) for v in videos for w in v["wins"] if m in w["y"]}) for m in MODS}
    return out


def n_levels(flags):
    return len(KINDS[flags.get("kind", "two")]["phases"]) if flags.get("kind", "two") != "joint" else 2


def set_modalities(run):
    """Branch keys of a cached run (README §12). The two-branch runs keep the order (z_visual, z_speech) of the main
    runs; other runs (e.g. a joint branch) use their own keys, sorted."""
    import twolevel as r1
    keys = sorted({k for rec in run.values() for w in rec["extra"]["windows"] for k in w if k.startswith("z_")})
    mods = ("z_visual", "z_speech") if set(keys) == {"z_visual", "z_speech"} else tuple(keys)
    globals()["MODS"] = mods
    r1.MODS = mods
    return mods


def emit_entry(mu00, mu, s2):
    """Emitter parameters: the V = 0 mean, the list of level means (level 0 = non-hate in a violating video, the last
    level = hate) and one variance. mu10 / mu11 stay as aliases of the first / last level for the older code paths."""
    return {"mu00": float(mu00), "mu": [float(x) for x in mu], "mu10": float(mu[0]), "mu11": float(mu[-1]), "s2": float(s2)}


def init_params(videos, flags):
    zs = np.array([v["zv"] for v in videos])
    kind = flags.get("kind", "two"); n_ph = len(KINDS[kind]["phases"])
    if flags.get("iota_stationary"):
        st = flags["d_hate"] / (flags["d_hate"] + flags["d_gap"]); io = [1 - st, st]
    else:
        io = [1.0 / n_ph] * n_ph
    P = {"pi": 0.5, "m0": float(np.percentile(zs, 10)), "m1": float(np.percentile(zs, 90)), "t2": max(float(zs.var()), EPS),
         "emit": {}, "iota": {mods: list(io) for mods in chains_of(flags)}}   # EPS floor: runs without a verdict (constant z_video)
    L = n_levels(flags)
    for m in MODS:
        for e in emitters(videos)[m]:
            ys = np.array([w["y"][m] for v in videos for w in v["wins"] if m in w["y"] and ek(w, m) == e])
            qs = np.linspace(10, 90, L + 1)
            mu00, *mu = (float(np.percentile(ys, q)) for q in qs)
            if flags.get("noleak"):
                mu[0] = mu00
            P["emit"][e] = emit_entry(mu00, mu, ys.var())
    P["carrier"] = [1 / 3, 1 / 3, 1 / 3] if flags.get("carrier") else None
    return P


def chain_for(P, mods, flags):
    return build_chain(flags["k"], flags["d_gap"], flags["d_hate"], P["iota"][mods], flags.get("nocoupling", False),
                       flags.get("kind", "two"), mods)


def video_terms(v, P, flags, kappa=1.0):
    lv0 = lognorm(v["zv"], P["m0"], P["t2"]); lv1 = lognorm(v["zv"], P["m1"], P["t2"])
    l0 = sum(lognorm(y, P["emit"][ek(w, m)]["mu00"], P["emit"][ek(w, m)]["s2"]) for w in v["wins"] for m, y in w["y"].items())
    per, lengths = [], {}
    kind = flags.get("kind", "two")
    for mods in chains_of(flags):
        if flags.get("duration") == "bma" and not flags.get("nocoupling"):
            grid = length_grid(v["n"], flags["k"], flags["bma_grid"], flags.get("bma_fixed"))
            chs = [build_chain(flags["k"], dg, dh, P["iota"][mods], False, kind, mods) for dg in grid for dh in grid]
            if flags.get("bma_prior") == "length":    # README §19: uniform in length; a log-grid point spans a length ∝ it
                lw = np.array([math.log(dg) + math.log(dh) for dg in grid for dh in grid])
                lw = lw - np.logaddexp.reduce(lw)
            else:                                        # README §16: uniform in log length
                lw = np.full(len(chs), -math.log(len(chs)))
            ll, g, w = bma_fb(emission(v, mods, P, chs[0], kappa), chs, lw)
            W2 = w.reshape(len(grid), len(grid))            # rows: gap length, columns: hate length
            lengths[mods] = (float(np.exp(W2.sum(0) @ np.log(grid))), float(np.exp(W2.sum(1) @ np.log(grid))))
            per.append((mods, chs[0], ll, g))
            continue
        ch = chain_for(P, mods, flags)
        ll, g = fb(emission(v, mods, P, ch, kappa), ch)
        per.append((mods, ch, ll, g))
    l1 = sum(x[2] for x in per)
    lo_verdict = math.log(P["pi"]) - math.log(1 - P["pi"]) + lv1 - lv0
    lo = lo_verdict + l1 - l0
    total = float(np.logaddexp(math.log(P["pi"]) + lv1 + l1, math.log(1 - P["pi"]) + lv0 + l0))
    out = {"per": per, "lo": lo, "lo_verdict": lo_verdict, "total": total, "lengths": lengths}
    if "icc" in P and not flags.get("sharedchain"):
        # reads of one video share its context: modality m's n reads count as n / (1 + (n - 1) icc_m) reads
        lo_r = 0.0
        for mods, ch, ll, g in per:
            m = mods[0]
            n_m = sum(1 for w in v["wins"] if m in w["y"])
            if n_m == 0:
                continue
            l0_m = sum(lognorm(w["y"][m], P["emit"][ek(w, m)]["mu00"], P["emit"][ek(w, m)]["s2"]) for w in v["wins"] if m in w["y"])
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


def key_calibration(keys, it=2000):
    """Label-free calibration of the video key K = z_video + mean window z: a two-component 1-D Gaussian mixture with
    a shared variance (EM from the 10th / 90th percentiles). Returns (a, b) with logit P(V = 1 | K) = a K + b."""
    x = np.asarray(keys, float)
    mu = np.percentile(x, [10, 90]).astype(float); var = float(x.var()); pi = 0.5
    for _ in range(it):
        l0 = math.log(1 - pi) - 0.5 * (x - mu[0]) ** 2 / var; l1 = math.log(pi) - 0.5 * (x - mu[1]) ** 2 / var
        r = expit(l1 - l0); pi = float(np.clip(r.mean(), EPS, 1 - EPS))
        mu = np.array([((1 - r) * x).sum() / (1 - r).sum(), (r * x).sum() / r.sum()])
        var = float(((1 - r) * (x - mu[0]) ** 2 + r * (x - mu[1]) ** 2).mean())
    if mu[1] < mu[0]:
        mu = mu[::-1]; pi = 1 - pi
    a = (mu[1] - mu[0]) / var
    b = math.log(pi) - math.log(1 - pi) - (mu[1] ** 2 - mu[0] ** 2) / (2 * var)
    return float(a), float(b)


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
                pa[m] = p_levels(v, g, ch, m)
        for m in MODS:
            r = []
            for i, wi in enumerate(v["wins"]):
                if m in wi["y"]:
                    e = P["emit"][ek(wi, m)]
                    mean = (1 - w) * e["mu00"] + w * float(pa[m][i] @ np.asarray(e["mu"]))
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
    if flags.get("duration") == "bma_corpus" and not flags.get("nocoupling"):   # README §18: one grid per corpus
        P["grid"] = length_grid(max(v["n"] for v in videos), flags["k"], flags["bma_grid"], flags.get("bma_fixed"))
    prev, it = None, 0
    L = n_levels(flags); n_ph = len(KINDS[flags.get("kind", "two")]["phases"])
    for it in range(max_it):
        # sufficient statistics per emitter: row 0 = V = 0, rows 1..L = observation levels 0..L-1 under V = 1
        S = {e: np.zeros((1 + L, 3)) for e in P["emit"]}; W, Z = [], []
        io = {mods: [np.zeros(n_ph), 0.0] for mods in chains_of(flags)}
        CR = np.zeros(3)
        total = 0.0
        if flags.get("duration") == "bma_corpus" and not flags.get("nocoupling"):
            total, P["pair_post"], terms = corpus_terms(videos, P, flags)
            for v, t in zip(videos, terms):
                w = t["W"]; W.append(w); Z.append(v["zv"])
                for wi in v["wins"]:
                    for m, y in wi["y"].items():
                        S[ek(wi, m)][0] += (1 - w) * np.array([1.0, y, y * y])
                for mods, ch, ph, pon in t["per"]:
                    io[mods][0] += w * np.array([1 - float(ph[0]), float(ph[0])]); io[mods][1] += w
                    for wi, pa in zip(v["wins"], pon):
                        for m in mods:
                            if m in wi["y"]:
                                y = wi["y"][m]
                                S[ek(wi, m)][1] += w * (1 - pa) * np.array([1.0, y, y * y])
                                S[ek(wi, m)][2] += w * pa * np.array([1.0, y, y * y])
        for v in ([] if flags.get("duration") == "bma_corpus" and not flags.get("nocoupling") else videos):
            t = video_terms(v, P, flags); total += t["total"]
            w = float(expit(t["lo"])); W.append(w); Z.append(v["zv"])
            for wi in v["wins"]:
                for m, y in wi["y"].items():
                    S[ek(wi, m)][0] += (1 - w) * np.array([1.0, y, y * y])
            for mods, ch, ll, g in t["per"]:
                if flags.get("nocoupling"):
                    io[mods][0] += w * p_phase(g, ch); io[mods][1] += w * v["n"]
                else:
                    io[mods][0] += w * p_phase(g[:1], ch); io[mods][1] += w
                if P.get("carrier") is not None:
                    lc = np.log(np.asarray(P["carrier"]))
                    for wi, pa in zip(v["wins"], p_on(v, g, ch)):
                        ons, off = carrier_terms(wi, P)
                        r = np.exp(lc + ons - np.logaddexp.reduce(lc + ons))       # carrier responsibilities
                        CR += w * pa * r
                        hotw = {"z_visual": r[0] + r[2], "z_speech": r[1] + r[2]}
                        for m, y in wi["y"].items():
                            S[ek(wi, m)][1] += w * (1 - pa * hotw[m]) * np.array([1.0, y, y * y])
                            S[ek(wi, m)][2] += w * pa * hotw[m] * np.array([1.0, y, y * y])
                    continue
                for m in mods:
                    pl = p_levels(v, g, ch, m)
                    for wi, pa in zip(v["wins"], pl):
                        if m in wi["y"]:
                            y = wi["y"][m]; e = ek(wi, m)
                            for l_ in range(L):
                                S[e][1 + l_] += w * pa[l_] * np.array([1.0, y, y * y])
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
        for e in P["emit"]:
            mu = S[e][:, 1] / np.maximum(S[e][:, 0], EPS)
            if flags.get("noleak"):
                mu[0] = mu[1] = (S[e][0, 1] + S[e][1, 1]) / max(S[e][0, 0] + S[e][1, 0], EPS)
            ss = sum(S[e][j, 2] - 2 * mu[j] * S[e][j, 1] + mu[j] ** 2 * S[e][j, 0] for j in range(1 + L))
            P["emit"][e] = emit_entry(mu[0], mu[1:], max(float(ss / max(S[e][:, 0].sum(), EPS)), EPS))
        for mods in io:
            vec = np.clip(io[mods][0] / max(io[mods][1], EPS), EPS, 1 - EPS)
            P["iota"][mods] = [float(x) for x in vec / vec.sum()]
            if flags.get("iota_stationary"):   # diagnostic (§12): start distribution = stationary phase share, not fitted
                st = flags["d_hate"] / (flags["d_hate"] + flags["d_gap"]); P["iota"][mods] = [1 - st, st]
        if P.get("carrier") is not None:
            P["carrier"] = [float(x) for x in np.clip(CR / max(CR.sum(), EPS), 1e-4, 1.0)]
            P["carrier"] = [x / sum(P["carrier"]) for x in P["carrier"]]
    log(f"EM stop after {it + 1} iterations, log-likelihood {prev:.4f}")
    return P


# ----------------------------------------------------------------------------------------------- main

DVD_FIELD = {"T": "z_target", "E": "z_endorse", "A": "z_attack"}   # experiments/20260927_dvd reads

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
    ap.add_argument("--evidence", choices=["em", "linear"], default="em",
                    help="diagnostic: linear = y / corpus std (current method's evidence), no EM; with --fusion max one chain on the max")
    ap.add_argument("--transform", choices=["none", "nscore"], default="none")
    ap.add_argument("--iota-stationary", action="store_true", help="diagnostic: start distribution fixed to the stationary phase share")
    ap.add_argument("--fusion", choices=["or", "carrier", "max"], default="or", help="per-modality chains + OR, or one chain with carrier fusion")
    ap.add_argument("--arm", choices=["m2", "full", "lexi"], default="m2")
    ap.add_argument("--key", choices=["raw", "calib", "scaled", "none"], default="raw",
                    help="video key: raw z_video + mean window z; calib = its label-free log-odds (key_calibration); "
                         "scaled = raw times --key-scale (declared scan); none = no video term (ablation)")
    ap.add_argument("--key-scale", type=float, default=1.0)
    ap.add_argument("--norank", action="store_true", help="ablation: no within-video term (video key only)")
    ap.add_argument("--vlevel", choices=["joint", "mix"], default="joint", help="video posterior: joint model (round 1) or mixture over verdict + mean reads")
    ap.add_argument("--vtemper", choices=["none", "icc"], default="none", help="video level: reads counted as ICC-effective reads")
    ap.add_argument("--kappa", type=float, default=1.0, help="diagnostic: time-level emissions tempered at inference")
    ap.add_argument("--duration", choices=["fixed", "bma", "bma_corpus"], default="fixed",
                    help="fixed: --d-gap / --d-hate; bma: mean lengths averaged per video over a log grid (README §16); "
                         "bma_corpus: one pair per chain shared by the corpus, posterior from all videos (README §18)")
    ap.add_argument("--bma-grid", type=int, default=6, help="bma: grid points per mean length")
    ap.add_argument("--bma-prior", choices=["log", "length"], default="log",
                    help="bma: prior over the mean lengths, uniform in log length (README §16) or in length (§19)")
    ap.add_argument("--bma-fixed", type=float, default=0.0, help="plumbing check only: pin the bma grid to one value (s)")
    ap.add_argument("--min-windows", type=float, default=0.0,
                    help="if > 0: shape k = min_windows x (window length of the reads) / cell length (README §16)")
    ap.add_argument("--dvd-reads", nargs="*", default=[],
                    help="experiments/20260927_dvd reads.jsonl files; with --dvd-conds the calibrated key becomes the "
                         "logit of a product of calibrated condition probabilities (noisy AND)")
    ap.add_argument("--dvd-conds", default="", help="comma list of viol (the current key K), T, E, A")
    ap.add_argument("--kind", choices=list(KINDS), default="two",
                    help="chain kind (experiments/20260928_infer/README.md): two = r6; three_nested / three_free = off topic / "
                         "topic / attack (§2); joint = one segmentation with a joint (visual, speech) label (§3)")
    ap.add_argument("--conditions", default="", help="window_conditions.py output (20260928_infer §1): emitters per "
                                                     "modality|reading condition")
    ap.add_argument("--conditions-mode", choices=["both", "vis", "sp", "shuf"], default="both",
                    help="which conditions to use; shuf = the labels permuted within the corpus (seed 0), a control")
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
    flags = {"k": a.k, "d_gap": a.d_gap, "d_hate": a.d_hate, "sharedchain": a.sharedchain or a.fusion == "max", "carrier": a.fusion == "carrier",
             "nocoupling": a.nocoupling, "noleak": a.noleak, "iota_stationary": a.iota_stationary,
             "duration": a.duration, "bma_grid": a.bma_grid, "bma_fixed": a.bma_fixed, "bma_prior": a.bma_prior,
             "kind": a.kind}
    if a.kind != "two" and (a.nocoupling or a.sharedchain or a.fusion != "or" or a.evidence != "em" or a.duration == "bma_corpus"):
        raise SystemExit("--kind other than two supports only the EM evidence, OR / joint fusion and fixed or bma durations")
    run = load_run(a.run)
    log(f"modalities {set_modalities(run)}")
    dvd_conds = [c for c in a.dvd_conds.split(",") if c]
    dvd = {}
    for path in a.dvd_reads:
        for line in open(path):
            r = json.loads(line)
            if not r.get("error"):
                dvd[(r["dataset"], r["video_id"])] = r
    if dvd_conds:
        need = [k_ for k_ in run if k_[0] in a.datasets]
        missing = [x for x in need if x not in dvd]
        if missing:
            raise SystemExit(f"DVD reads missing for {len(missing)} videos, e.g. {missing[:3]}")
        log(f"DVD key: conditions {dvd_conds} from {len(dvd)} read records")
    if a.min_windows > 0:
        wl = float(np.median([w["end"] - w["start"] for rec in run.values() for w in rec["extra"]["windows"]]))
        flags["k"] = max(1, int(round(a.min_windows * wl / CELL)))
        log(f"shape from the reading grid: {a.min_windows:g} windows x {wl:g} s / {CELL:g} s cells -> k = {flags['k']}")
    videos = {ds: [prep(r, a.center) for k, r in sorted(run.items()) if k[0] == ds] for ds in a.datasets}
    if a.conditions:
        cond = json.load(open(a.conditions))["conditions"]
        use = {"z_visual": a.conditions_mode in ("both", "vis", "shuf"), "z_speech": a.conditions_mode in ("both", "sp", "shuf")}
        rng = np.random.default_rng(0)
        for ds in a.datasets:
            for v in videos[ds]:
                cv = cond[ds][v["rec"]["video_id"]]
                if len(cv) != len(v["wins"]):
                    raise SystemExit(f"{ds} {v['rec']['video_id']}: {len(cv)} conditions for {len(v['wins'])} windows")
                for w, (f, s) in zip(v["wins"], cv):
                    w["cond"] = {m: c for m, c in (("z_visual", f), ("z_speech", s)) if use[m] and c is not None and m in w["y"]}
            if a.conditions_mode == "shuf":        # control: same label counts, labels permuted over the corpus's windows
                for m in MODS:
                    ws = [w for v in videos[ds] for w in v["wins"] if m in w.get("cond", {})]
                    labs = [w["cond"][m] for w in ws]
                    for w, c in zip(ws, rng.permutation(labs)):
                        w["cond"][m] = str(c)
            counts = {}
            for v in videos[ds]:
                for w in v["wins"]:
                    for m in w["y"]:
                        counts[ek(w, m)] = counts.get(ek(w, m), 0) + 1
            log(f"[{ds}] emitters ({a.conditions_mode}): " + "  ".join(f"{e} {n}" for e, n in sorted(counts.items())))
    if a.transform == "nscore":
        # diagnostic (§12): each modality's reads -> normal scores of their rank within the corpus (only the order of
        # reads is used, not their scale); the key keeps the raw reads
        from scipy.stats import norm, rankdata
        for ds in a.datasets:
            for m in MODS:
                refs = [w for v in videos[ds] for w in v["wins"] if m in w["y"]]
                ys = np.array([w["y"][m] for w in refs])
                sc = norm.ppf((rankdata(ys) - 0.5) / len(ys))
                for w, z in zip(refs, sc):
                    w["y"][m] = float(z)
    params = {}
    for ds in a.datasets:
        if a.evidence == "linear":
            P = init_params(videos[ds], flags)
            P["linear"] = {m: float(np.std([w["y"][m] for v in videos[ds] for w in v["wins"] if m in w["y"]])) for m in MODS}
            P["iota"] = {mods: 0.5 for mods in chains_of(flags)}
            params[ds] = P
            if a.key == "calib":
                P["key_ab"] = key_calibration([intercept(v) for v in videos[ds]])
            log(f"[{ds}] linear evidence, corpus std " + " ".join(f"{m} {x:.3f}" for m, x in P["linear"].items()))
            continue
        P = em(videos[ds], flags, log=lambda m, ds=ds: log(f"[{ds}] {m}"))
        if a.vtemper == "icc":
            P["icc"] = residual_icc(videos[ds], P, flags)
            log(f"[{ds}] residual ICC " + "  ".join(f"{m} {x:.3f}" for m, x in P["icc"].items()))
        if a.vlevel == "mix":
            P["vmix_fn"], info = vlevel_mixture(videos[ds])
            P["vmix"] = info
            log(f"[{ds}] video mixture: pi {info['pi']:.3f} violating means {np.round(info['mu_violating'], 2).tolist()} "
                f"other {np.round(info['mu_other'], 2).tolist()} sd {np.round(np.sqrt(info['var']), 2).tolist()}")
        if a.key == "calib":
            P["key_ab"] = key_calibration([intercept(v) for v in videos[ds]])
            log(f"[{ds}] key calibration: logit P(V = 1 | K) = {P['key_ab'][0]:.4f} K {P['key_ab'][1]:+.4f}")
            if dvd_conds:
                P["dvd_ab"] = {}
                for c in dvd_conds:
                    if c == "viol":
                        continue
                    xs = [dvd[(ds, v["rec"]["video_id"])][DVD_FIELD[c]] for v in videos[ds]]
                    P["dvd_ab"][c] = key_calibration(xs)
                    log(f"[{ds}] condition {c}: logit P = {P['dvd_ab'][c][0]:.4f} z {P['dvd_ab'][c][1]:+.4f}; "
                        f"share P > .5 {np.mean([expit(P['dvd_ab'][c][0] * x + P['dvd_ab'][c][1]) > .5 for x in xs]):.3f}")
        params[ds] = P
        if P.get("carrier") is not None:
            log(f"[{ds}] carrier visual / speech / both {np.round(P['carrier'], 3).tolist()}")
        if flags["duration"] == "bma_corpus" and not flags["nocoupling"]:
            if a.vtemper != "none" or a.vlevel != "joint" or a.fusion == "carrier":
                raise SystemExit("bma_corpus supports only the default video level and OR / shared-chain fusion")
            g_ = P["grid"]
            for mods, wm in P["pair_post"].items():
                W2 = wm.reshape(len(g_), len(g_)); j = int(np.argmax(wm))
                log(f"[{ds}] {'+'.join(mods)} corpus posterior over (gap, hate) means: mode ({g_[j // len(g_)]:.1f} s, "
                    f"{g_[j % len(g_)]:.1f} s) weight {wm[j]:.3f}; posterior geometric mean hate "
                    f"{np.exp(W2.sum(0) @ np.log(g_)):.1f} s, gap {np.exp(W2.sum(1) @ np.log(g_)):.1f} s; grid "
                    f"{g_[0]:.0f}-{g_[-1]:.0f} s")
        if flags["duration"] == "bma" and not flags["nocoupling"]:
            ln = [video_terms(v, P, flags)["lengths"] for v in videos[ds]]
            for mods in chains_of(flags):
                h = np.array([x[mods][0] for x in ln]); gp = np.array([x[mods][1] for x in ln])
                log(f"[{ds}] {'+'.join(mods)} posterior mean length per video (geometric mean over the grid), "
                    f"hate: median {np.median(h):.1f} s [10-90% {np.percentile(h, 10):.1f}-{np.percentile(h, 90):.1f}]  "
                    f"gap: median {np.median(gp):.1f} s [10-90% {np.percentile(gp, 10):.1f}-{np.percentile(gp, 90):.1f}]")
        log(f"[{ds}] pi {P['pi']:.3f} verdict {P['m0']:.2f}/{P['m1']:.2f} sd {math.sqrt(P['t2']):.2f}  iota " +
            " ".join(f"{'+'.join(k_)} {np.round(v_, 3).tolist()}" for k_, v_ in P["iota"].items()) + "  " +
            "  ".join(f"{m}: mu00 {e['mu00']:.2f} levels {np.round(e['mu'], 2).tolist()} sd {math.sqrt(e['s2']):.2f} "
                      f"slope {(e['mu11'] - e['mu10']) / e['s2']:.3f}" for m, e in P["emit"].items()))
        if a.kind != "two" and not (flags["duration"] == "bma_corpus"):
            ph_share = np.zeros(len(KINDS[a.kind]["phases"]))
            for v in videos[ds]:
                for mods, ch, ll, g in video_terms(v, P, flags)["per"]:
                    ph_share += p_phase(g, ch)
            log(f"[{ds}] share of cells per phase under V = 1: {np.round(ph_share / ph_share.sum(), 3).tolist()}")
    pred_path = out / "predictions.jsonl"
    corpus_mode = flags["duration"] == "bma_corpus" and not flags["nocoupling"]
    with open(pred_path, "w") as fh:
        for ds in a.datasets:
            cterms = corpus_terms(videos[ds], params[ds], flags, a.kappa)[2] if corpus_mode else None
            for n_v, v in enumerate(videos[ds]):
                miss = np.ones(v["n"])
                if corpus_mode:
                    ct = cterms[n_v]; Wv = float(np.clip(ct["W"], 1e-12, 1 - 1e-12))
                    t = {"lo": math.log(Wv) - math.log1p(-Wv)}
                    for mods, ch, ph, pon in ct["per"]:
                        miss *= 1.0 - ph
                else:
                    t = video_terms(v, params[ds], flags, a.kappa)
                    for mods, ch, ll, g in t["per"]:
                        miss *= 1.0 - p_hate(g, ch)
                p = np.clip(1.0 - miss, 1e-12, 1 - 1e-12)
                lo = params[ds]["vmix_fn"](v) if a.vlevel == "mix" else (t["lo_icc"] if a.vtemper == "icc" else t["lo"])
                intervals = []
                K = intercept(v)
                if a.key == "calib" and dvd_conds:
                    lp = 0.0
                    for c in dvd_conds:
                        if c == "viol":
                            ka, kb = params[ds]["key_ab"]; lp += float(log_expit(ka * K + kb))
                        else:
                            ca, cb = params[ds]["dvd_ab"][c]
                            lp += float(log_expit(ca * dvd[(ds, v["rec"]["video_id"])][DVD_FIELD[c]] + cb))
                    key = lp - float(np.log(-np.expm1(min(lp, -1e-12)))); lo = key     # logit of the product
                elif a.key == "calib":
                    ka, kb = params[ds]["key_ab"]; key = ka * K + kb; lo = key
                elif a.key == "scaled":
                    key = a.key_scale * K
                elif a.key == "none":
                    key = 0.0
                else:
                    key = K
                if a.arm == "m2":
                    score = key + (0.0 if a.norank else centered_rank(to_frames(np.log(p) - np.log1p(-p), v["L"])))
                    score = np.full(v["L"], score) if np.ndim(score) == 0 else score
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
        if "pair_post" in P:                       # README §18: corpus grid and posterior over (gap, hate) pairs
            P["grid"] = [float(x) for x in P["grid"]]
            P["pair_post"] = {"+".join(k_): [float(x) for x in v_] for k_, v_ in P["pair_post"].items()}
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
