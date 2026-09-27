#!/usr/bin/env python3
"""Headroom of the open concerns K3, K4, K5, K7 under the current protocol (README.md). Reads the gold: analysis only,
never part of any method. On the current method `final_m2` (r6_bma) and its reads:
A. video level (K3): pooled ROC / PR if the video score were perfect, keeping the method's within-video ranking;
   A3: a video score fitted on the gold (5-fold CV) from the current reads, the method's outputs and the DVD reads:
   how much of that headroom the existing reads can reach;
B. within level (K4, K5): a model fitted on the gold (5-fold cross-validation by video, same corpus) on the same reads,
   alone and with the method's own outputs as extra inputs, within-video features only, best of three settings. If it
   cannot beat the label-free method, the reads are the limit, not the way the method uses them;
C. visual branch (K7): the same fitted bound on visual reads only, speech reads only and both; and on the older reads
   with and without one frame per window.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
CELL, FPS = 4.0, 4
MODS = ("z_visual", "z_speech")


def load(path, ds):
    return {r["video_id"]: r for r in map(json.loads, open(path)) if not r.get("error") and r["dataset"] == ds}


def gold(ds):
    g = np.load(ROOT / f"data/gt_4fps/{ds}.npz", allow_pickle=True)
    return {str(v): np.asarray(y, int) for v, y in zip(g["video_ids"], g["y4"])}


def pooled(Y, S, vids):
    y = np.concatenate([Y[v] for v in vids]); s = np.concatenate([S[v] for v in vids])
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def per_video_within(Y, S, vids):
    return {v: float(roc_auc_score(Y[v], S[v])) for v in vids if 0 < Y[v].sum() < len(Y[v])}


def boot_diff(a, b, n=2000, seed=0):
    """Paired bootstrap over videos of mean(a - b); a, b: dicts video -> value on the same videos."""
    vs = sorted(set(a) & set(b)); d = np.array([a[v] - b[v] for v in vs]); rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(d.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975)), len(vs)


# ------------------------------------------------------------------------------------------------ A. video level

def part_a(ds, pred, Y):
    vids = sorted(v for v in Y if v in pred)
    S, R, lab = {}, {}, {}
    for v in vids:
        s = np.asarray(pred[v]["score_curve"], float)[:len(Y[v])]
        assert len(s) == len(Y[v]), (v, len(s), len(Y[v]))
        S[v] = s; R[v] = s - float(pred[v]["extra"]["p_video_logodds"]); lab[v] = bool(Y[v].any())
    low = min(s.min() for s in S.values()) - 1e6
    out = {"n_videos": len(vids), "n_hateful": int(sum(lab.values())), "current": pooled(Y, S, vids)}
    # A1: every non-hateful video below every hateful one; hateful videos keep the method's scores
    out["perfect_separation"] = pooled(Y, {v: S[v] if lab[v] else np.full(len(S[v]), low) for v in vids}, vids)
    # A2: also order hateful videos by their true hate coverage (key = scale * logit coverage), best scale by PR
    best = None
    for sc in (0, .25, .5, 1, 2, 4, 8, 16, 64, 1e4):
        S2 = {}
        for v in vids:
            c = min(max(Y[v].mean(), 1e-3), 1 - 1e-3)
            S2[v] = sc * math.log(c / (1 - c)) + R[v] if lab[v] else np.full(len(S[v]), low)
        m = pooled(Y, S2, vids)
        if best is None or m[1] > best[1][1]:
            best = (sc, m)
    out["perfect_video_level"] = best[1]; out["perfect_video_level_scale"] = best[0]
    return out


# ------------------------------------------------------------------------------------------------ B/C. within level

def shift(a, k):
    """a moved by k cells (k > 0: value of the cell k earlier), NaN where undefined."""
    out = np.full(len(a), np.nan)
    if k == 0:
        out[:] = a
    elif 0 < k < len(a):
        out[k:] = a[:-k]
    elif 0 < -k < len(a):
        out[:k] = a[-k:]
    return out


def cell_rows(reads, pred, Y, mods, with_method):
    """One row per 4 s cell of every video with both classes. Only features that can change the order inside a video:
    each modality's read minus the video's mean read, its rank in the video, the neighbouring windows' reads minus the
    mean, the position; with the method, its cell log-odds minus the video mean, its rank and its neighbours."""
    X, T, W, G, idx = [], [], [], [], []
    for v in sorted(Y):
        y = Y[v]
        if v not in reads or (with_method and v not in pred) or not 0 < y.sum() < len(y):
            continue
        n_c = int(math.ceil(len(y) / (CELL * FPS)))
        z = {m: np.full(n_c, np.nan) for m in mods}
        for w in reads[v]["extra"]["windows"]:
            for c in range(int(w["start"] // CELL), min(n_c, int(math.ceil(w["end"] / CELL)))):
                if w["start"] <= c * CELL + CELL / 2 < w["end"]:
                    for m in mods:
                        if w.get(m) is not None:
                            z[m][c] = float(w[m])
        feats = []
        for m in mods:
            x = z[m]; ok = np.isfinite(x); mu = x[ok].mean() if ok.any() else 0.0
            feats += [x - mu, rank_in(x), shift(x, 2) - mu, shift(x, -2) - mu, shift(x, 4) - mu, shift(x, -4) - mu]
        feats += [np.arange(n_c) / max(n_c - 1, 1)]
        if with_method:
            p = np.asarray(pred[v]["extra"]["cell_prob"], float)[:n_c]
            p = np.r_[p, np.full(n_c - len(p), p[-1])]      # the method's cell grid may be one cell shorter
            lp = np.log(np.clip(p, 1e-12, 1)) - np.log(np.clip(1 - p, 1e-12, 1)); lp = lp - lp.mean()
            feats += [lp, rank_in(lp), shift(lp, 1), shift(lp, -1), shift(lp, 2), shift(lp, -2)]
        F = np.stack(feats, 1)
        for c in range(n_c):
            f0, f1 = int(c * CELL * FPS), min(len(y), int((c + 1) * CELL * FPS))
            if f1 <= f0:
                continue
            X.append(F[c]); T.append(y[f0:f1].mean()); W.append(f1 - f0); G.append(v); idx.append((v, f0, f1))
    return np.array(X), np.array(T), np.array(W), np.array(G), idx


def rank_in(x):
    r = np.full(len(x), np.nan); ok = np.isfinite(x)
    if ok.sum() > 1:
        r[ok] = (np.argsort(np.argsort(x[ok])) + .5) / ok.sum()
    return r


MODELS = {
    "boosted trees": lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=.05, max_leaf_nodes=15,
                                                           min_samples_leaf=50, l2_regularization=1.0, random_state=0),
    "small boosted trees": lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=.05, max_leaf_nodes=7,
                                                                 min_samples_leaf=100, l2_regularization=1.0,
                                                                 random_state=0),
    "ridge": lambda: make_pipeline(SimpleImputer(strategy="constant", fill_value=0.0), StandardScaler(), Ridge(1.0)),
}


def fitted_bound(reads, pred, Y, mods, with_method):
    """Within of a model fitted on the gold, 5-fold cross-validation by video; the best of three model settings (a
    generous bound). Returns (per-video within of the best setting, name of the setting, mean within per setting)."""
    X, T, W, G, idx = cell_rows(reads, pred, Y, mods, with_method)
    res = {}
    for name, make in MODELS.items():
        P = np.zeros(len(T))
        for tr, te in GroupKFold(5).split(X, T, G):
            mdl = make()
            if name == "ridge":
                mdl.fit(X[tr], T[tr], ridge__sample_weight=W[tr])
            else:
                mdl.fit(X[tr], T[tr], sample_weight=W[tr])
            P[te] = mdl.predict(X[te])
        S = {v: np.zeros(len(Y[v])) for v in set(G)}
        for (v, f0, f1), p in zip(idx, P):
            S[v][f0:f1] = p
        res[name] = per_video_within(Y, S, sorted(S))
    best = max(res, key=lambda k: np.mean(list(res[k].values())))
    return res[best], best, {k: float(np.mean(list(r.values()))) for k, r in res.items()}


# ------------------------------------------------------------------------------------------------ A3. video level, fitted

def video_features(v, reads, pred, dvd):
    f = [float(reads[v]["extra"]["z_video"])]
    for m in MODS:
        x = np.array([w[m] for w in reads[v]["extra"]["windows"] if w.get(m) is not None], float)
        f += [x.mean(), np.median(x), x.max(), np.quantile(x, .25), np.quantile(x, .75), x.std(), (x > 0).mean()] \
            if len(x) else [np.nan] * 7
    p = np.asarray(pred[v]["extra"]["cell_prob"], float)
    f += [float(pred[v]["extra"]["p_video_logodds"]), p.mean(), (p > .5).mean(), len(p)]
    d = dvd.get(v, {})
    f += [d.get("z_target", np.nan), d.get("z_endorse", np.nan), d.get("z_attack", np.nan)]
    return f


def part_a3(ds, pred, reads, dvd, Y):
    """Video score fitted on the gold (5-fold CV over videos) from the current reads, the method's outputs and the DVD
    reads; target = the video's hate coverage. Frame score = scale x logit(predicted coverage) + the method's
    within term; best model setting and scale by pooled PR (a generous bound)."""
    vids = sorted(v for v in Y if v in pred and v in reads)
    X = np.array([video_features(v, reads, pred, dvd) for v in vids]); cov = np.array([Y[v].mean() for v in vids])
    R = {v: np.asarray(pred[v]["score_curve"], float)[:len(Y[v])] - float(pred[v]["extra"]["p_video_logodds"])
         for v in vids}
    lab = (cov > 0).astype(int)
    best = None
    for name, make in (("ridge", lambda: make_pipeline(SimpleImputer(), StandardScaler(), Ridge(1.0))),
                       ("boosted trees", lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=.05,
                                                                              max_leaf_nodes=7, min_samples_leaf=10,
                                                                              random_state=0))):
        P = np.zeros(len(vids))
        for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
            mdl = make(); mdl.fit(X[tr], cov[tr]); P[te] = mdl.predict(X[te])
        pc = np.clip(P, 1e-3, 1 - 1e-3); key = np.log(pc / (1 - pc))
        for sc in (.25, .5, 1, 2, 4, 8, 16):
            m = pooled(Y, {v: sc * k + R[v] for v, k in zip(vids, key)}, vids)
            if best is None or m[1] > best[0][1]:
                best = (m, name, sc, float(roc_auc_score(lab, P)))
    return {"fitted": best[0], "model": best[1], "scale": best[2], "video_auc": best[3]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(ROOT / "runs/20260926_twolevel/final_m2/predictions.jsonl"))
    ap.add_argument("--reads", default=str(ROOT / "runs/20260926_glr/base_gridA/predictions.jsonl"))
    ap.add_argument("--reads-k20", default=str(ROOT / "runs/20260910_spvl/full2_dual_evid_stance/predictions.jsonl"))
    ap.add_argument("--reads-w8", default=str(ROOT / "runs/20260910_spvl/full3_dual_evid_stance_w8/predictions.jsonl"))
    ap.add_argument("--dehate-pred", default=str(ROOT / "runs/20260926_twolevel/final_dehate/final_m2/predictions.jsonl"))
    ap.add_argument("--dehate-reads", default=str(ROOT / "runs/20260927_dehate_external/reads_gridA/predictions.jsonl"))
    ap.add_argument("--dvd-main", default=str(ROOT / "runs/20260927_dvd/reads_main/reads.jsonl"))
    ap.add_argument("--dvd-dehate", default=str(ROOT / "runs/20260927_dvd/reads_dehate/reads.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "runs/20260928_headroom"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lines, summary = [], {}
    fmt = lambda m: f"{m[0]:.4f} / {m[1]:.4f}"
    lines.append("A. video level: pooled ROC / PR (within unchanged)")
    for ds, pp, rp, dp in (("HateMM", a.pred, a.reads, a.dvd_main), ("HateClipSeg", a.pred, a.reads, a.dvd_main),
                           ("DeHate", a.dehate_pred, a.dehate_reads, a.dvd_dehate)):
        Y, pred = gold(ds), load(pp, ds)
        r = part_a(ds, pred, Y); summary[f"A:{ds}"] = r
        dvd = {x["video_id"]: x for x in map(json.loads, open(dp)) if x["dataset"] == ds and not x.get("error")}
        r3 = part_a3(ds, pred, load(rp, ds), dvd, Y); summary[f"A3:{ds}"] = r3
        lines.append(f"  {ds:11s} ({r['n_hateful']}/{r['n_videos']} hateful)  current {fmt(r['current'])}")
        lines.append(f"    perfect split of hateful / non-hateful videos      {fmt(r['perfect_separation'])}")
        lines.append(f"    plus hateful videos ordered by true hate share     {fmt(r['perfect_video_level'])} "
                     f"(scale {r['perfect_video_level_scale']:g})")
        lines.append(f"    video score fitted on the gold from current reads  {fmt(r3['fitted'])} "
                     f"({r3['model']}, scale {r3['scale']:g}; video AUC {r3['video_auc']:.3f})")
    lines.append("B/C. within: label-free method vs a model fitted on the gold (5-fold CV by video); "
                 "difference [95% interval], videos; K5 = hate covers < 25 %")
    for ds in ("HateMM", "HateClipSeg"):
        Y, pred, reads = gold(ds), load(a.pred, ds), load(a.reads, ds)
        cur = per_video_within(Y, {v: np.asarray(pred[v]["score_curve"], float)[:len(Y[v])] for v in pred if v in Y},
                               sorted(v for v in pred if v in Y))
        k5 = {v for v in cur if Y[v].mean() < .25}
        arms = {"fitted: reads (both modalities)": fitted_bound(reads, pred, Y, MODS, False),
                "fitted: reads + method outputs": fitted_bound(reads, pred, Y, MODS, True),
                "fitted: visual reads only": fitted_bound(reads, pred, Y, ("z_visual",), False),
                "fitted: speech reads only": fitted_bound(reads, pred, Y, ("z_speech",), False)}
        arms, settings = {k: v[0] for k, v in arms.items()}, {k: (v[1], v[2]) for k, v in arms.items()}
        lines.append(f"  == {ds}: label-free method within {np.mean(list(cur.values())):.4f} (n {len(cur)}); "
                     f"K5 subset {np.mean([cur[v] for v in k5]):.4f} (n {len(k5)})")
        for name, w in arms.items():
            d = boot_diff(w, cur); dk = boot_diff({v: w[v] for v in k5 if v in w}, {v: cur[v] for v in k5})
            summary[f"B:{ds}:{name}"] = {"within": float(np.mean(list(w.values()))), "diff": d, "k5_diff": dk,
                                         "best_setting": settings[name][0], "per_setting": settings[name][1]}
            lines.append(f"    {name:34s} {np.mean(list(w.values())):.4f}  vs method {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]"
                         f"   K5 {dk[0]:+.4f} [{dk[1]:+.4f}, {dk[2]:+.4f}]   ({settings[name][0]})")
        for tag, path in (("20 shared frames (old ASR)", a.reads_k20), ("plus a frame per window (old ASR)", a.reads_w8)):
            rd = load(path, ds)
            for mods, mname in ((("z_visual",), "visual only"), (MODS, "both")):
                w = fitted_bound(rd, pred, Y, mods, False)[0]
                summary[f"C:{ds}:{tag}:{mname}"] = float(np.mean(list(w.values())))
                lines.append(f"    fitted, {tag}, {mname:11s} {np.mean(list(w.values())):.4f}")
    (out / "table.txt").write_text("\n".join(lines) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
