"""Shared explicit-duration chain kernels, promoted unchanged from twolevel_r2 on 2026-10-02.

No data loading, fitting or evaluation. Four-second cells are the existing method grid.
"""
import math
import numpy as np

CELL = 4.0
MODS = ("z_visual", "z_speech")

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


def fb_grid_numpy(E, chains, logw):
    """Batched forward-backward over observation variants and transition grids.

    E: (Q,n,S), chains: G transition/prior variants, logw: normalized grid prior.
    Returns log evidence (Q), posterior states after grid integration (Q,n,S),
    and grid posterior weights (Q,G). Same calculation as bma_fb per Q.
    """
    Q, n, S = E.shape
    T = np.stack([c["T"] for c in chains])
    pi = np.stack([c["pi0"] for c in chains])
    mx = E.max(axis=2)
    likelihood = np.exp(E - mx[:, :, None])
    alpha = np.empty((Q, len(chains), n, S))
    scale = np.empty((Q, len(chains), n))
    state = pi[None] * likelihood[:, None, 0]
    for t in range(n):
        if t:
            state = np.einsum("qgi,gij->qgj", state, T) * likelihood[:, None, t]
        scale[:, :, t] = state.sum(axis=-1)
        if not np.all(scale[:, :, t] > 0):
            raise FloatingPointError("forward probability underflow")
        state = state / scale[:, :, t, None]
        alpha[:, :, t] = state
    lls = np.log(scale).sum(axis=-1) + mx.sum(axis=1)[:, None]
    weighted = lls + np.asarray(logw)[None]
    evidence = np.logaddexp.reduce(weighted, axis=1)
    postw = np.exp(weighted - evidence[:, None])
    posterior = np.empty_like(E)
    beta = np.ones((Q, len(chains), S))
    for t in range(n - 1, -1, -1):
        if t < n - 1:
            beta = np.einsum("gij,qgj->qgi", T, likelihood[:, None, t+1] * beta)
            beta /= scale[:, :, t+1, None]
        g = alpha[:, :, t] * beta
        g /= g.sum(axis=-1, keepdims=True)
        posterior[:, t] = np.einsum("qg,qgs->qs", postw, g)
    return evidence, posterior, postw


def _fb_sparse(E, pi, edges_from, edges_to, weights):
    """Same scaled recursion, sparse edges; JIT compilation is optional."""
    Q, n, S = E.shape
    G = len(pi)
    lls = np.zeros((Q,G))
    gs = np.zeros((Q,G,n,S))
    for q in range(Q):
        obs = np.empty((n,S))
        shift = 0.0
        for t in range(n):
            mx = np.max(E[q,t])
            shift += mx
            obs[t] = np.exp(E[q,t]-mx)
        for g in range(G):
            alpha = np.zeros((n,S)); scales = np.zeros(n)
            alpha[0] = pi[g]*obs[0]
            scales[0] = np.sum(alpha[0])
            if scales[0] <= 0: raise FloatingPointError("forward probability underflow")
            alpha[0] /= scales[0]
            for t in range(1,n):
                for e in range(len(edges_from)):
                    alpha[t,edges_to[e]] += alpha[t-1,edges_from[e]]*weights[g,e]
                alpha[t] *= obs[t]
                scales[t] = np.sum(alpha[t])
                if scales[t] <= 0: raise FloatingPointError("forward probability underflow")
                alpha[t] /= scales[t]
            lls[q,g] = np.sum(np.log(scales))+shift
            beta = np.ones(S)
            for t in range(n-1,-1,-1):
                if t < n-1:
                    prev = np.zeros(S)
                    for e in range(len(edges_from)):
                        i, j = edges_from[e], edges_to[e]
                        prev[i] += weights[g,e]*obs[t+1,j]*beta[j]
                    beta = prev/scales[t+1]
                gs[q,g,t] = alpha[t]*beta
                gs[q,g,t] /= np.sum(gs[q,g,t])
    return lls, gs


try:
    from numba import njit
    _fb_sparse = njit(cache=False)(_fb_sparse)
except ImportError:
    _fb_sparse = None


def fb_grid(E, chains, logw):
    """Accelerated batch kernel; scalar and NumPy implementations are references."""
    if _fb_sparse is None:
        return fb_grid_numpy(E, chains, logw)
    T = np.stack([c["T"] for c in chains])
    pi = np.stack([c["pi0"] for c in chains])
    src, dst = np.nonzero(np.any(T != 0,axis=0))
    ll, gs = _fb_sparse(E,pi,src,dst,T[:,src,dst])
    weighted = ll + np.asarray(logw)[None]
    evidence = np.logaddexp.reduce(weighted,axis=1)
    weights = np.exp(weighted-evidence[:,None])
    posterior = np.einsum("qg,qgns->qns",weights,gs)
    return evidence, posterior, weights
