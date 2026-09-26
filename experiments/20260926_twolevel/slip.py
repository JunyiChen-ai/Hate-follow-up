#!/usr/bin/env python3
"""Three-phase time level with learned durations (README §15). CPU, cached reads.

Per modality, each 4 s cell is in one of three phases: normal, hate, or slip (the MLLM reads the window as hateful
although it is not). Hate and slip windows have the same read distribution N(mu1, s2); normal windows N(mu0, s2).
They differ only in how long they last and where they can occur:
  - slip can occur in any video; hate only in a violating video (V = 1);
  - every phase lasts a geometric number of cells whose mean is learned by EM (no hand-set duration or shape).
Transitions: normal -> hate (rate h, V = 1 only), normal -> slip (rate s, any video), hate -> normal, slip -> normal.
Each 8 s window's read observes "hate or slip anywhere in the window's cells" (pair states). Reads are normal scores
of their rank within the corpus (as round 3). Video level, key calibration and composition as `r3_m2`.
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
from scipy.special import expit, log_expit
from scipy.stats import norm, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from twolevel import (CELL, EPS, ROOT, centered_rank, intercept, intervals_from, load_run, lognorm,  # noqa: E402
                      prep, to_frames)
import twolevel_r2 as r2  # noqa: E402

NORMAL, HATE, SLIP = 0, 1, 2
HIGH = np.array([0, 1, 1])        # phases whose windows read high
ISHATE = np.array([0, 1, 0])
Z = 6                              # augmented state z = b * 3 + x, b = "previous cell high"
B_OF = np.repeat([0, 1], 3)
X_OF = np.tile([0, 1, 2], 2)
ON_PAIR = ((B_OF + HIGH[X_OF]) > 0).astype(int)
ON_SINGLE = HIGH[X_OF]


# ----------------------------------------------------------------------------------------------- chain

def phase_matrix(tr, v1, nocoupling=False):
    h = tr["h"] if v1 else 0.0
    s = tr["s"]
    row_n = [1.0 - h - s, h, s]
    if nocoupling:
        return np.array([row_n, row_n, row_n])
    return np.array([row_n, [1.0 - tr["pH"], tr["pH"], 0.0], [1.0 - tr["pS"], 0.0, tr["pS"]]])


def augment(A, iota):
    T = np.zeros((Z, Z))
    for bp in range(2):
        for xp in range(3):
            for x in range(3):
                T[bp * 3 + xp, HIGH[xp] * 3 + x] = A[xp, x]
    pi0 = np.zeros(Z)
    pi0[:3] = iota
    return T, pi0


def fb(E, T, pi0):
    """Scaled forward-backward. Returns log-likelihood, state posteriors (n, Z), expected transitions (Z, Z)."""
    n = E.shape[0]
    m = E.max(1, keepdims=True)
    G = np.exp(E - m)
    al = np.zeros_like(E); cs = np.zeros(n)
    a = pi0 * G[0]; cs[0] = a.sum(); al[0] = a / cs[0]
    for c in range(1, n):
        a = (al[c - 1] @ T) * G[c]; cs[c] = a.sum(); al[c] = a / cs[c]
    be = np.ones_like(E)
    for c in range(n - 2, -1, -1):
        be[c] = T @ (G[c + 1] * be[c + 1]) / cs[c + 1]
    g = al * be
    g /= g.sum(1, keepdims=True)
    Xi = np.zeros((Z, Z))
    for c in range(1, n):
        Xi += np.outer(al[c - 1], G[c] * be[c]) * T / cs[c]
    return float(np.log(cs).sum() + m.sum()), g, Xi


def phase_counts(Xi):
    X = np.zeros((3, 3))
    for zp in range(Z):
        for z in range(Z):
            X[X_OF[zp], X_OF[z]] += Xi[zp, z]
    return X


def selftest(trials=60, seed=0):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        n = int(rng.integers(1, 6)); v1 = bool(rng.random() < 0.7)
        tr = {"h": rng.uniform(.01, .3), "s": rng.uniform(.01, .3), "pH": rng.uniform(.3, .97), "pS": rng.uniform(.1, .9)}
        iota = rng.dirichlet(np.ones(3))
        if not v1:
            iota[HATE] = 0.0; iota /= iota.sum()
        A = phase_matrix(tr, v1)
        T, pi0 = augment(A, iota)
        E = rng.normal(0, 2, size=(n, Z))
        ll, g, Xi = fb(E, T, pi0)
        paths = list(itertools.product(range(3), repeat=n)); lps = []
        for xs in paths:
            lp = math.log(iota[xs[0]]) if iota[xs[0]] > 0 else -np.inf
            lp += E[0, xs[0]]
            for c in range(1, n):
                t = A[xs[c - 1], xs[c]]
                lp += (math.log(t) if t > 0 else -np.inf) + E[c, HIGH[xs[c - 1]] * 3 + xs[c]]
            lps.append(lp)
        lps = np.array(lps); LL = np.logaddexp.reduce(lps); post = np.exp(lps - LL)
        marg = np.array([sum(p for p, xs in zip(post, paths) if xs[c] == HATE) for c in range(n)])
        cnt = np.zeros((3, 3))
        for p, xs in zip(post, paths):
            for c in range(1, n):
                cnt[xs[c - 1], xs[c]] += p
        worst = max(worst, abs(ll - LL), float(np.abs(marg - g[:, X_OF == HATE].sum(1)).max()),
                    float(np.abs(cnt - phase_counts(Xi)).max()))
    return worst


# ----------------------------------------------------------------------------------------------- model

def emission(v, m, P):
    E = np.zeros((v["n"], Z))
    e = P["emit"][m]
    for w in v["wins"]:
        if m in w["y"]:
            on = ON_PAIR if w["pair"] else ON_SINGLE
            E[w["cell"]] += np.where(on == 1, lognorm(w["y"][m], e["mu1"], e["s2"]), lognorm(w["y"][m], e["mu0"], e["s2"]))
    return E


def init_params(videos, mods, flags):
    zs = np.array([v["zv"] for v in videos])
    P = {"pi": 0.5, "m0": float(np.percentile(zs, 10)), "m1": float(np.percentile(zs, 90)), "t2": max(float(zs.var()), EPS),
         "emit": {}, "tr": {}, "iota1": {}, "iota0": {}}
    for m in mods:
        ys = np.array([w["y"][m] for v in videos for w in v["wins"] if m in w["y"]])
        P["emit"][m] = {"mu0": float(np.percentile(ys, 10)), "mu1": float(np.percentile(ys, 90)), "s2": float(ys.var())}
        s0 = 0.0 if flags["noslip"] else 0.025
        P["tr"][m] = {"h": 0.025, "s": s0, "pH": 1 - CELL / 80.0, "pS": 1 - CELL / 8.0}
        P["iota1"][m] = np.array([.5, .5, 0.0]) if flags["noslip"] else np.array([.5, .25, .25])
        P["iota0"][m] = np.array([1.0, 0.0, 0.0]) if flags["noslip"] else np.array([.75, 0.0, .25])
    return P


def video_terms(v, P, mods, flags):
    lv0 = lognorm(v["zv"], P["m0"], P["t2"]); lv1 = lognorm(v["zv"], P["m1"], P["t2"])
    per, l1, l0 = {}, 0.0, 0.0
    for m in mods:
        E = emission(v, m, P)
        o1 = fb(E, *augment(phase_matrix(P["tr"][m], True, flags["nocoupling"]), P["iota1"][m]))
        o0 = fb(E, *augment(phase_matrix(P["tr"][m], False, flags["nocoupling"]), P["iota0"][m]))
        per[m] = (o1, o0); l1 += o1[0]; l0 += o0[0]
    lo = math.log(P["pi"]) - math.log(1 - P["pi"]) + lv1 - lv0 + l1 - l0
    total = float(np.logaddexp(math.log(P["pi"]) + lv1 + l1, math.log(1 - P["pi"]) + lv0 + l0))
    return {"per": per, "lo": lo, "total": total}


def em(videos, mods, flags, max_it=300, tol=1e-7, log=print):
    P = init_params(videos, mods, flags)
    prev, it = None, 0
    for it in range(max_it):
        S = {m: np.zeros((2, 3)) for m in mods}                 # rows mu0 / mu1: sum w, sum w y, sum w y^2
        C1 = {m: np.zeros((3, 3)) for m in mods}; C0 = {m: np.zeros((3, 3)) for m in mods}
        I1 = {m: np.zeros(3) for m in mods}; I0 = {m: np.zeros(3) for m in mods}
        W, Zv, total = [], [], 0.0
        for v in videos:
            t = video_terms(v, P, mods, flags); total += t["total"]
            w = float(expit(t["lo"])); W.append(w); Zv.append(v["zv"])
            for m in mods:
                (ll1, g1, X1), (ll0, g0, X0) = t["per"][m]
                C1[m] += w * phase_counts(X1); C0[m] += (1 - w) * phase_counts(X0)
                for x in range(3):
                    I1[m][x] += w * g1[0, X_OF == x].sum(); I0[m][x] += (1 - w) * g0[0, X_OF == x].sum()
                for wi in v["wins"]:
                    if m not in wi["y"]:
                        continue
                    on = ON_PAIR if wi["pair"] else ON_SINGLE
                    p_on = w * g1[wi["cell"], on == 1].sum() + (1 - w) * g0[wi["cell"], on == 1].sum()
                    y = wi["y"][m]
                    S[m][0] += (1 - p_on) * np.array([1.0, y, y * y]); S[m][1] += p_on * np.array([1.0, y, y * y])
        if prev is not None:
            if total < prev - 1e-6 * abs(prev):
                raise AssertionError(f"EM log-likelihood decreased at iteration {it}: {prev} -> {total}")
            if abs(total - prev) <= tol * abs(prev):
                break
        prev = total
        W, Zv = np.array(W), np.array(Zv)
        P["pi"] = float(np.clip(W.mean(), EPS, 1 - EPS))
        P["m1"] = float((W * Zv).sum() / max(W.sum(), EPS)); P["m0"] = float(((1 - W) * Zv).sum() / max((1 - W).sum(), EPS))
        P["t2"] = max(float((W * (Zv - P["m1"]) ** 2 + (1 - W) * (Zv - P["m0"]) ** 2).mean()), EPS)
        for m in mods:
            mu = S[m][:, 1] / np.maximum(S[m][:, 0], EPS)
            ss = sum(S[m][j, 2] - 2 * mu[j] * S[m][j, 1] + mu[j] ** 2 * S[m][j, 0] for j in range(2))
            P["emit"][m] = {"mu0": float(mu[0]), "mu1": float(mu[1]), "s2": max(float(ss / max(S[m][:, 0].sum(), EPS)), EPS)}
            c1, c0 = C1[m], C0[m]
            if flags["nocoupling"]:              # every cell's phase is an independent draw from the normal row
                d1, d0 = c1.sum(0), c0.sum(0)
                a, b, c, d = d1[NORMAL], d1[HATE], d1[SLIP] + d0[SLIP], d0[NORMAL]
            else:
                a, b, c, d = c1[NORMAL, NORMAL], c1[NORMAL, HATE], c1[NORMAL, SLIP] + c0[NORMAL, SLIP], c0[NORMAL, NORMAL]
            s_ = 0.0 if flags["noslip"] else float(np.clip(c / max(a + b + c + d, EPS), EPS, .5))
            h_ = float(np.clip(b * (1 - s_) / max(a + b, EPS), EPS, .5))
            pH = float(np.clip(c1[HATE, HATE] / max(c1[HATE, HATE] + c1[HATE, NORMAL], EPS), EPS, 1 - EPS))
            sS, sN = c1[SLIP, SLIP] + c0[SLIP, SLIP], c1[SLIP, NORMAL] + c0[SLIP, NORMAL]
            pS = float(np.clip(sS / max(sS + sN, EPS), EPS, 1 - EPS))
            P["tr"][m] = {"h": h_, "s": s_, "pH": pH, "pS": pS}
            if flags["nocoupling"]:
                P["iota1"][m] = np.array([1 - h_ - s_, h_, s_]); P["iota0"][m] = np.array([1 - s_, 0.0, s_])
            else:
                i1 = np.clip(I1[m] / max(I1[m].sum(), EPS), 0, 1); i0 = np.clip(I0[m] / max(I0[m].sum(), EPS), 0, 1)
                if flags["noslip"]:
                    i1[SLIP] = 0.0; i0[:] = [1.0, 0.0, 0.0]
                i0[HATE] = 0.0
                P["iota1"][m] = np.clip(i1 / i1.sum(), 0, 1); P["iota0"][m] = np.clip(i0 / i0.sum(), 0, 1)
    log(f"EM stop after {it + 1} iterations, log-likelihood {prev:.4f}")
    return P


def durations(tr):
    f = lambda p: CELL / max(1 - p, EPS)
    return {"normal_s": CELL / max(tr["h"] + tr["s"], EPS), "hate_s": f(tr["pH"]), "slip_s": f(tr["pS"]),
            "hate_starts_per_min": 60.0 / CELL * tr["h"], "slip_starts_per_min": 60.0 / CELL * tr["s"]}


# ----------------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "runs/20260926_glr/base_gridA"))
    ap.add_argument("--arm", choices=["m2", "full"], default="m2")
    ap.add_argument("--noslip", action="store_true", help="ablation: no slip phase (two phases, learned durations)")
    ap.add_argument("--nocoupling", action="store_true", help="ablation: independent cells")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-root", default=str(ROOT / "runs/20260926_twolevel"))
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
    log(f"code experiments/20260926_twolevel/slip.py at commit {commit} (plus uncommitted changes if any)")
    if a.selftest:
        worst = selftest()
        log(f"selftest three-phase forward-backward vs brute force (likelihood, hate marginals, transition counts): {worst:.2e}")
        if worst > 1e-8:
            raise SystemExit("SELFTEST_FAILED")
    flags = {"noslip": a.noslip, "nocoupling": a.nocoupling}
    run = load_run(a.run)
    mods = r2.set_modalities(run)
    log(f"modalities {mods}")
    videos = {ds: [prep(r) for k, r in sorted(run.items()) if k[0] == ds] for ds in a.datasets}
    for ds in a.datasets:                       # normal scores of each modality's reads within the corpus
        for m in mods:
            refs = [w for v in videos[ds] for w in v["wins"] if m in w["y"]]
            ys = np.array([w["y"][m] for w in refs])
            for w, zsc in zip(refs, norm.ppf((rankdata(ys) - 0.5) / len(ys))):
                w["y"][m] = float(zsc)
    params = {}
    for ds in a.datasets:
        P = em(videos[ds], mods, flags, log=lambda msg, ds=ds: log(f"[{ds}] {msg}"))
        P["key_ab"] = r2.key_calibration([intercept(v) for v in videos[ds]])
        params[ds] = P
        for m in mods:
            e, d = P["emit"][m], durations(P["tr"][m])
            log(f"[{ds}] {m}: mu0 {e['mu0']:.3f} mu1 {e['mu1']:.3f} sd {math.sqrt(e['s2']):.3f}  mean durations: normal "
                f"{d['normal_s']:.0f} s, hate {d['hate_s']:.1f} s, slip {d['slip_s']:.1f} s; starts per minute: hate "
                f"{d['hate_starts_per_min']:.3f}, slip {d['slip_starts_per_min']:.3f}; iota1 {np.round(P['iota1'][m], 3).tolist()}")
        log(f"[{ds}] pi {P['pi']:.3f} verdict {P['m0']:.2f}/{P['m1']:.2f} sd {math.sqrt(P['t2']):.2f}  key calibration "
            f"{P['key_ab'][0]:.4f} K {P['key_ab'][1]:+.4f}")
    pred = out / "predictions.jsonl"
    with open(pred, "w") as fh:
        for ds in a.datasets:
            P = params[ds]; ka, kb = P["key_ab"]
            for v in videos[ds]:
                t = video_terms(v, P, mods, flags)
                miss = np.ones(v["n"]); slip = np.ones(v["n"])
                for m in mods:
                    g1 = t["per"][m][0][1]
                    miss *= 1.0 - g1[:, X_OF == HATE].sum(1); slip *= 1.0 - g1[:, X_OF == SLIP].sum(1)
                p = np.clip(1.0 - miss, 1e-12, 1 - 1e-12)
                key = ka * intercept(v) + kb
                intervals = []
                if a.arm == "m2":
                    score = key + centered_rank(to_frames(np.log(p) - np.log1p(-p), v["L"]))
                else:
                    score = float(log_expit(key)) + np.log(to_frames(p, v["L"]))
                    intervals = intervals_from(np.exp(score))
                fh.write(json.dumps({**v["rec"], "method": f"twolevel_slip__{a.tag}", "score_curve": [float(x) for x in score],
                                     "intervals": intervals,
                                     "extra": {"z_video": v["zv"], "key": key, "p_hate_cell": [float(x) for x in p],
                                               "p_slip_cell": [float(x) for x in 1.0 - slip]}}) + "\n")
    mp = out / "metrics.json"
    subprocess.run([sys.executable, str(ROOT / "src/eval/evaluate_four_datasets.py"), "--predictions", str(pred),
                    "--gt-dir", str(ROOT / "data/gt_4fps"), "--out", str(mp), "--datasets", *a.datasets],
                   check=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)}, stdout=subprocess.DEVNULL)
    ser = {ds: {"pi": P["pi"], "m0": P["m0"], "m1": P["m1"], "t2": P["t2"], "emit": P["emit"], "tr": P["tr"],
                "durations": {m: durations(P["tr"][m]) for m in mods}, "key_ab": P["key_ab"],
                "iota1": {m: P["iota1"][m].tolist() for m in mods}, "iota0": {m: P["iota0"][m].tolist() for m in mods}}
           for ds, P in params.items()}
    (out / "params.json").write_text(json.dumps(ser, indent=2))
    (out / "config.json").write_text(json.dumps({**vars(a), "cell_s": CELL, "modalities": list(mods)}, indent=2))
    d = json.load(open(mp))
    log(f"{a.tag:22s} " + "  ".join(f"{p_['dataset'][:6]} {p_['frame_ROC_AUC']:.4f}/{p_['frame_PR_AUC']:.4f}/"
                                    f"{p_['within_video_macro_ROC_AUC']:.4f} F1@.3/.5/.7 {p_['interval_F1@0.3']:.3f}/"
                                    f"{p_['interval_F1@0.5']:.3f}/{p_['interval_F1@0.7']:.3f}" for p_ in d["per_dataset"]))
    log("RUN_DONE")


if __name__ == "__main__":
    main()
