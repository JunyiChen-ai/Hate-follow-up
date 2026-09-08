"""Scale-emergence analysis: 3 model scales x 2 test corpora x 5 frozen readouts.

Pre-registration: docs/duplex/PREREG_scale_emergence.md (frozen and committed
before any 4B weight was downloaded). Every convention below is quoted from it
and none may be changed here.

CPU only. No model call. Reads the per-video raw z and the per-video stored
hidden states written by the frozen judge, plus the unembedding rows and the
final RMSNorm weight of each checkpoint from the local Hugging Face cache.

Readouts, identical for every model and corpus:
  (a) z distribution: sd, dynamic range, de-quantized KDE relative trough depth
      on the exact crossbench_analyze.kde_valley convention, bf16 trough depth
      alongside;
  (b) angular commitment: sd of cos(h, d) where h is the final state after the
      model's own RMSNorm and d = mean(W[yes]) - mean(W[no]);
  (c) ranking AUC of the de-quantized z against gold labels;
  (d) five-fold ridge probe AUC on the hidden state at layer round(0.75*L);
  (e) macro-F1 at the label-free valley against macro-F1 at the oracle
      threshold.

Disconfirmers D1, D2 and D3 are evaluated exactly as frozen.

Writes results/scale_emergence/results.json. No video id reaches the output.

Usage:
  python scripts/duplex/scale_emergence_analyze.py
"""

import glob
import json
import math
import os
import sys

import numpy as np
import torch
from safetensors import safe_open
from scipy.stats import gaussian_kde

ROOT = "/home/jehc223/Hate-follow-up"
sys.path.insert(0, os.path.join(ROOT, "src", "our_method"))
os.environ.setdefault("HVD_DATA_ROOT", "/home/jehc223/data")
from data_utils import load_annotations  # noqa: E402

OUT = os.path.join(ROOT, "results", "scale_emergence", "results.json")

# ---------------------------------------------------------------- frozen ----
YES_IDS = [7414, 9454, 9693, 9834, 14004, 14080]
NO_IDS = [902, 2152, 2308, 2753, 5664, 8996]
RMS_EPS = 1e-6
PROBE_DEPTH_FRAC = 0.75
PROBE_LAMBDA = 1.0
N_FOLDS = 5
FOLD_SEED = 0
N_GRID = 4001
GRID_PAD = 2.0
D1_RATIO = 0.7
D2_MARGIN = 0.05

MODELS = [
    ("2b", "Qwen/Qwen3-VL-2B-Instruct", "judge_2b", 28),
    ("4b", "Qwen/Qwen3-VL-4B-Instruct", "judge_4b", 36),
    ("8b", "Qwen/Qwen3-VL-8B-Instruct", "judge_8b", 36),
]
CORPORA = [
    ("implihatevid", "ImpliHateVid", {"Hateful": 1, "Normal": 0}),
    ("hatemm", "HateMM", {"Hate": 1, "Non Hate": 0}),
]
DESCRIPTIVE = [("8b_thinking", "Qwen/Qwen3-VL-8B-Thinking",
                "judge_8b_thinking", 36, "implihatevid")]


# ---------------------------------------------------------------- helpers ---
def auc(scores, labels):
    """Mann-Whitney ROC-AUC with midranks for ties."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if npos == 0 or nneg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(s)
    ss = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def kde_valley(z):
    """Frozen E7/KDE recipe, byte-for-byte the crossbench_analyze convention."""
    z = np.asarray(z, dtype=float)
    kde = gaussian_kde(z, bw_method="scott")
    grid = np.linspace(z.min() - GRID_PAD, z.max() + GRID_PAD, N_GRID)
    dens = kde(grid)
    loc = [i for i in range(1, N_GRID - 1)
           if dens[i] > dens[i - 1] and dens[i] > dens[i + 1]]
    out = {"bandwidth_scott": float(kde.factor * np.std(z, ddof=1)),
           "n_grid_local_maxima": len(loc)}
    if len(loc) < 2:
        out["value"] = None
        out["relative_trough_depth"] = None
        out["note"] = "fewer than two modes: the recipe finds no valley"
        return out
    a, b = sorted(sorted(loc, key=lambda i: -dens[i])[:2])
    j = a + 1 + int(np.argmin(dens[a + 1:b]))
    out.update({
        "value": float(grid[j]),
        "mode_locations": [float(grid[a]), float(grid[b])],
        "relative_trough_depth": float(1.0 - dens[j] / min(dens[a], dens[b])),
    })
    return out


def macro_f1(y, pred):
    y = np.asarray(y, int)
    pred = np.asarray(pred, int)
    fs = []
    for c in (0, 1):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(fs))


def oracle_threshold(z, y):
    """Macro-F1-maximising threshold. Diagnostic ceiling, never label-free."""
    cands = np.unique(np.asarray(z, float))
    mids = np.concatenate([[cands[0] - 1.0],
                           (cands[:-1] + cands[1:]) / 2.0,
                           [cands[-1] + 1.0]])
    best, best_t = -1.0, None
    for t in mids:
        f = macro_f1(y, (np.asarray(z) >= t).astype(int))
        if f > best:
            best, best_t = f, float(t)
    return best_t, best


def load_head(repo):
    """Unembedding rows for the frozen id sets, plus the final RMSNorm weight."""
    from huggingface_hub import snapshot_download
    d = snapshot_download(repo, allow_patterns=["*.safetensors"],
                          local_files_only=True)
    want_norm = "model.language_model.norm.weight"
    found = {}
    lm_key = None
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with safe_open(f, framework="pt") as sf:
            keys = list(sf.keys())
            if want_norm in keys:
                found[want_norm] = sf.get_tensor(want_norm)
            for k in keys:
                if k == "lm_head.weight":
                    lm_key = k
                    found[k] = sf.get_tensor(k)
    if lm_key is None:
        # Tied embeddings: the unembedding is the text embedding table.
        lm_key = "model.language_model.embed_tokens.weight"
        for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
            with safe_open(f, framework="pt") as sf:
                if lm_key in sf.keys():
                    found[lm_key] = sf.get_tensor(lm_key)
    if lm_key not in found or want_norm not in found:
        raise SystemExit(f"ABORT: {repo} missing {lm_key} or {want_norm}")
    W = found[lm_key].float()
    g = found[want_norm].float()
    return W[YES_IDS], W[NO_IDS], g, lm_key


def final_state(x, g):
    """The state that produces the logits: the model's own final RMSNorm."""
    return x * torch.rsqrt(x.pow(2).mean() + RMS_EPS) * g


def ridge_probe_auc(X, y, layer):
    """Five-fold stratified ridge probe; AUC over the held-out predictions."""
    y = np.asarray(y, int)
    rng = np.random.default_rng(FOLD_SEED)
    folds = np.empty(len(y), dtype=int)
    for c in (0, 1):
        idx = np.where(y == c)[0]
        idx = idx[rng.permutation(len(idx))]
        folds[idx] = np.arange(len(idx)) % N_FOLDS
    held = np.empty(len(y), dtype=float)
    for k in range(N_FOLDS):
        te = folds == k
        tr = ~te
        Xtr, Xte = X[tr], X[te]
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0)
        sd = np.where(sd <= 0, 1.0, sd)
        Ztr = (Xtr - mu) / sd
        Zte = (Xte - mu) / sd
        ttr = y[tr].astype(float) - y[tr].mean()
        d = Ztr.shape[1]
        if d <= Ztr.shape[0]:
            A = Ztr.T @ Ztr + PROBE_LAMBDA * np.eye(d)
            w = np.linalg.solve(A, Ztr.T @ ttr)
        else:  # dual form, cheaper when d > n
            K = Ztr @ Ztr.T + PROBE_LAMBDA * np.eye(Ztr.shape[0])
            w = Ztr.T @ np.linalg.solve(K, ttr)
        held[te] = Zte @ w + y[tr].mean()
    return auc(held, y), int(layer)


# ------------------------------------------------------------------- cell ---
def measure(arm, repo, judge_dir, n_layers, slug, dataset, labmap, ann):
    d_dir = os.path.join(ROOT, "results", "testruns", slug, judge_dir)
    spath = os.path.join(d_dir, "scores.jsonl")
    hdir = os.path.join(d_dir, "hidden")
    if not os.path.exists(spath):
        return None

    z_stored = {}
    with open(spath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if isinstance(r.get("z"), (int, float)) and math.isfinite(r["z"]):
                z_stored[r["video_id"]] = float(r["z"])
    ids = sorted(z_stored)
    for v in ids:
        if not os.path.isfile(os.path.join(hdir, v + ".npy")):
            raise SystemExit(f"ABORT: {slug}/{arm}: {v} has z but no hidden state")

    Wy, Wn, g, lm_key = load_head(repo)
    dvec = Wy.mean(0) - Wn.mean(0)
    probe_layer = int(round(PROBE_DEPTH_FRAC * n_layers))

    zd, cosv, hn, Xp = [], [], [], []
    for v in ids:
        a = np.load(os.path.join(hdir, v + ".npy"))
        if a.shape != (n_layers + 1, g.shape[0]):
            raise SystemExit(f"ABORT: {slug}/{arm}: hidden shape {a.shape}")
        h = final_state(torch.from_numpy(a[-1]).float(), g)
        zd.append(float(torch.logsumexp(Wy @ h, 0) - torch.logsumexp(Wn @ h, 0)))
        cosv.append(float(torch.dot(h, dvec) / (h.norm() * dvec.norm())))
        hn.append(float(h.norm()))
        Xp.append(a[probe_layer].astype(np.float32))
    zd = np.array(zd)
    zs = np.array([z_stored[v] for v in ids])
    cosv = np.array(cosv)
    Xp = np.stack(Xp)

    y = np.array([labmap[ann[v]["label"]] for v in ids], dtype=int)

    v_dq = kde_valley(zd)
    v_bf = kde_valley(zs)
    probe_auc, used_layer = ridge_probe_auc(Xp, y, probe_layer)
    rank_auc = auc(zd, y)
    o_thr, o_f1 = oracle_threshold(zd, y)

    if v_dq.get("value") is None:
        valley_f1 = None
        valley_counts = None
    else:
        pred = (zd >= v_dq["value"]).astype(int)
        valley_f1 = macro_f1(y, pred)
        valley_counts = {"fn": int(((pred == 0) & (y == 1)).sum()),
                         "fp": int(((pred == 1) & (y == 0)).sum())}

    return {
        "arm": arm, "checkpoint": repo, "corpus": dataset,
        "n_videos": len(ids), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
        "unembedding_key": lm_key, "tied_embeddings": lm_key != "lm_head.weight",
        "n_layers": n_layers, "probe_layer": used_layer,
        "z_distribution": {
            "sd_dequantized": float(zd.std(ddof=1)),
            "sd_stored_bf16": float(zs.std(ddof=1)),
            "dynamic_range_dequantized": float(zd.max() - zd.min()),
            "min": float(zd.min()), "max": float(zd.max()),
            "max_abs_dequantization_shift": float(np.abs(zd - zs).max()),
            "n_modes_dequantized": v_dq["n_grid_local_maxima"],
            "relative_trough_depth_dequantized": v_dq["relative_trough_depth"],
            "relative_trough_depth_stored_bf16": v_bf["relative_trough_depth"],
            "valley_dequantized": v_dq.get("value"),
            "n_modes_stored_bf16": v_bf["n_grid_local_maxima"],
        },
        "angular_commitment": {
            "cos_sd": float(cosv.std(ddof=1)),
            "cos_mean": float(cosv.mean()),
            "cos_min": float(cosv.min()), "cos_max": float(cosv.max()),
            "answer_direction_norm": float(dvec.norm()),
            "mean_final_state_norm": float(np.mean(hn)),
        },
        "ranking_auc": rank_auc,
        "probe": {"auc": probe_auc, "layer": used_layer,
                  "n_folds": N_FOLDS, "l2_lambda": PROBE_LAMBDA,
                  "seed": FOLD_SEED},
        "operating_point": {
            "valley_macro_f1_label_free": valley_f1,
            "valley_errors": valley_counts,
            "oracle_macro_f1": o_f1, "oracle_threshold": o_thr,
            "gap": None if valley_f1 is None else float(o_f1 - valley_f1),
        },
    }


# -------------------------------------------------------------------- main --
def main():
    cells = {}
    ann_cache = {}
    for slug, dataset, labmap in CORPORA:
        ann_cache[dataset] = load_annotations(dataset)
        for arm, repo, jdir, nl in MODELS:
            c = measure(arm, repo, jdir, nl, slug, dataset, labmap,
                        ann_cache[dataset])
            if c is None:
                raise SystemExit(f"ABORT: no scores for {slug}/{arm}")
            cells[f"{slug}|{arm}"] = c
            print(f"{slug:14s} {arm:3s} n={c['n_videos']:4d} "
                  f"trough={c['z_distribution']['relative_trough_depth_dequantized']} "
                  f"cos_sd={c['angular_commitment']['cos_sd']:.5f} "
                  f"auc={c['ranking_auc']:.4f} probe={c['probe']['auc']:.4f}")

    # coverage: the 4B arm must match the 8B arm on every corpus
    for slug, _, _ in CORPORA:
        if cells[f"{slug}|4b"]["n_videos"] != cells[f"{slug}|8b"]["n_videos"]:
            raise SystemExit(f"ABORT: coverage mismatch on {slug}")

    def trough(slug, arm):
        return cells[f"{slug}|{arm}"]["z_distribution"][
            "relative_trough_depth_dequantized"]

    # ---- D1: emergence or smooth scaling ---------------------------------
    d1_rows = []
    for slug, _, _ in CORPORA:
        t4, t8 = trough(slug, "4b"), trough(slug, "8b")
        ratio = None if (t4 is None or t8 is None or t8 == 0) else t4 / t8
        d1_rows.append({"corpus": slug, "trough_4b": t4, "trough_8b": t8,
                        "ratio": ratio,
                        "at_or_above_0.7x": bool(ratio is not None
                                                 and ratio >= D1_RATIO)})
    d1_fires = all(r["at_or_above_0.7x"] for r in d1_rows)

    # ---- D2: construct presence at 4B ------------------------------------
    d2_rows = []
    for slug, _, _ in CORPORA:
        p4 = cells[f"{slug}|4b"]["probe"]["auc"]
        p8 = cells[f"{slug}|8b"]["probe"]["auc"]
        d2_rows.append({"corpus": slug, "probe_4b": p4, "probe_8b": p8,
                        "delta": p4 - p8,
                        "materially_below": bool(p8 - p4 > D2_MARGIN)})
    d2_fires = any(r["materially_below"] for r in d2_rows)

    # ---- D3: does angular spread track trough depth? ---------------------
    d3_rows = []
    for slug, _, _ in CORPORA:
        arms = ["2b", "4b", "8b"]
        cos = {a: cells[f"{slug}|{a}"]["angular_commitment"]["cos_sd"]
               for a in arms}
        tr = {a: trough(slug, a) for a in arms}
        cos_order = sorted(arms, key=lambda a: cos[a])
        # a corpus with no valley sorts below every corpus that has one
        tr_order = sorted(arms, key=lambda a: (-1.0 if tr[a] is None
                                               else tr[a]))
        d3_rows.append({"corpus": slug,
                        "cos_sd": cos, "trough": tr,
                        "cos_sd_ascending": cos_order,
                        "trough_ascending": tr_order,
                        "orders_agree": cos_order == tr_order})
    d3_fires = any(not r["orders_agree"] for r in d3_rows)

    res = {
        "protocol": "docs/duplex/PREREG_scale_emergence.md",
        "compute": ("CPU only for this analysis; no model call. Per-video z and "
                    "hidden states written by the frozen judge are read from "
                    "disk, together with the unembedding rows and the final "
                    "RMSNorm weight of each checkpoint."),
        "frozen_constants": {
            "yes_ids": YES_IDS, "no_ids": NO_IDS,
            "final_state": ("hidden_states[-1] is pre-final-norm; the model's "
                            "own RMSNorm with eps=1e-6 is applied before any "
                            "geometry is read"),
            "answer_direction": "mean(W[yes_ids]) - mean(W[no_ids])",
            "kde": {"n_grid": N_GRID, "grid_pad": GRID_PAD,
                    "bandwidth": "scott",
                    "trough": "1 - density(valley)/min(density of the two modes)"},
            "probe": {"layer_rule": "round(0.75 * n_layers)",
                      "folds": N_FOLDS, "l2_lambda": PROBE_LAMBDA,
                      "seed": FOLD_SEED,
                      "standardisation": "per fold, training-half statistics"},
            "d1_ratio": D1_RATIO, "d2_margin": D2_MARGIN,
        },
        "cells": cells,
        "disconfirmers": {
            "D1_smooth_scaling": {
                "rule": ("fires iff the 4B de-quantized relative trough depth "
                         f"is >= {D1_RATIO}x the 8B's on BOTH corpora; firing "
                         "collapses emergence to smooth capability scaling"),
                "rows": d1_rows, "fires": bool(d1_fires),
                "verdict": ("SMOOTH SCALING" if d1_fires
                            else "NOT DISCONFIRMED: the 4B trough does not "
                                 "reach 0.7x the 8B's on both corpora")},
            "D2_construct_absent_at_4b": {
                "rule": (f"fires iff the 4B probe AUC is more than {D2_MARGIN} "
                         "below the 8B's on either corpus"),
                "rows": d2_rows, "fires": bool(d2_fires),
                "verdict": ("CONSTRUCT HALF DIES" if d2_fires
                            else "NOT DISCONFIRMED: the construct is present "
                                 "at 4B to within the frozen margin")},
            "D3_angular_mechanism": {
                "rule": ("fires iff the ascending rank order of cosine spread "
                         "differs from the ascending rank order of trough "
                         "depth on either corpus"),
                "rows": d3_rows, "fires": bool(d3_fires),
                "verdict": ("MECHANISM DIES" if d3_fires
                            else "NOT DISCONFIRMED: angular spread and trough "
                                 "depth order the three scales identically")},
        },
    }

    # ---- optional descriptive arm, reported apart ------------------------
    desc = {}
    for arm, repo, jdir, nl, slug in DESCRIPTIVE:
        dataset, labmap = next((d, m) for s, d, m in CORPORA if s == slug)
        try:
            c = measure(arm, repo, jdir, nl, slug, dataset, labmap,
                        ann_cache[dataset])
        except SystemExit as e:
            c = {"not_run": str(e)}
        if c is not None:
            desc[f"{slug}|{arm}"] = c
    res["descriptive_arms"] = desc
    res["descriptive_arms_note"] = (
        "The pre-registered non-instruct control could not be run: no base "
        "Qwen3-VL checkpoint is published, only Instruct, Thinking, FP8 and "
        "GGUF variants. The declared substitute, Qwen3-VL-8B-Thinking, was "
        "checked and rejected before it was scored: its chat template appends "
        "'<think>\\n' to the generation prompt, so the final prompt position "
        "is the opening of a reasoning block rather than the Yes-or-No answer "
        "position, and its z would not be the same measurement. No "
        "descriptive arm ran.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
        f.write("\n")
    print(json.dumps(res["disconfirmers"], indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
