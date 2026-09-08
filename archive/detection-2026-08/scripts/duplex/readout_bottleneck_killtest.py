"""Readout-bottleneck kill test on the frozen judge's hidden states.

Pre-registration: docs/duplex/PREREG_readout_bottleneck_killtest.md.

CPU only. No model call, no rescoring: the hidden states written by the frozen
single-call Qwen3-VL-8B judge runs are read from disk and nothing else.

Frozen decision rule, all three clauses required for SURVIVES:
  1. Logistic probe on MHClip-EN layer-27 hidden states, blind-coded
     no-protected-target positives (34) versus protected-target positives (15),
     leave-one-out cross-validation, ROC-AUC >= 0.75.
  2. At least one of the 24 unsupervised MHClip-EN axes (3 layers x 8 PCA
     components, fit on the corpus's own hidden states with no labels) reaches
     ROC-AUC >= 0.799 on no-protected-target positives versus shipped Normals.
  3. At the layer of the best clause-2 axis, at least one of HateClipSeg's own
     top-3 PCA axes reaches ROC-AUC >= 0.687 on insulting-only videos versus
     clean normals.

Frozen constants: layers {18, 27, 36} of the 37 stored rows (row 0 is the
embedding output); 8 components per layer per corpus; every feature matrix is
standardized per dimension with the mean and standard deviation of that
corpus's own unlabeled hidden states before both the PCA fit and the probe fit;
axis sign is flipped when needed so that the Spearman correlation with that
corpus's z is non-negative; the probe's L2 penalty is fixed at lambda = 1.0 and
is never tuned. HateClipSeg's `lexicons.json` is never read.

Writes results/readout_bottleneck/results.json. No video id reaches the output.

Usage:
  python scripts/duplex/readout_bottleneck_killtest.py
"""

import ast
import csv
import glob
import json
import math
import os

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

ROOT = "/home/jehc223/Hate-follow-up"
EN_HIDDEN = os.path.join(ROOT, "results", "testruns", "mhclip_en", "judge_8b",
                         "hidden")
EN_SCORES = os.path.join(ROOT, "results", "testruns", "mhclip_en", "judge_8b",
                         "scores.jsonl")
HCS_HIDDEN = os.path.join(ROOT, "results", "hateclipseg", "judge_8b", "hidden")
HCS_SCORES = os.path.join(ROOT, "results", "hateclipseg", "judge_8b",
                          "scores.jsonl")
AV = os.path.join(ROOT, "results", "annotation_validity")
EN_ANN = "/home/jehc223/data/Multihateclip/English/annotation(new).json"
HCS_ANN = os.path.join(ROOT, "idea-stage", "pilots", "b1_coverage_audit",
                       "data", "video_level_annotation.csv")
OUT = os.path.join(ROOT, "results", "readout_bottleneck", "results.json")

LAYERS = [18, 27, 36]
N_PC = 8
N_PC_CLAUSE3 = 3
PROBE_LAYER = 27
PROBE_LAMBDA = 1.0
CLAUSE1_FLOOR = 0.75
CLAUSE2_FLOOR = 0.799
CLAUSE3_FLOOR = 0.687
EXPECTED_ROWS = 37
EXPECTED_DIM = 4096

HCS_IDX = ["normal", "hateful", "insulting", "sexual", "violence", "harm"]
HCS_OFFENSIVE = set(HCS_IDX[1:])


# ------------------------------------------------------------------ helpers --
def auc(scores, labels):
    """Mann-Whitney ROC-AUC with midranks for ties (same routine as the audit)."""
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


def auc_pos_neg(pos, neg):
    return auc(list(pos) + list(neg), [1] * len(pos) + [0] * len(neg))


def load_z(path):
    d = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if isinstance(r.get("z"), (int, float)) and math.isfinite(r["z"]):
                d[r["video_id"]] = float(r["z"])
    return d


def load_hidden(hidden_dir, ids):
    """(n, 37, 4096) float32 stack in the given id order; structure verified."""
    out = np.empty((len(ids), EXPECTED_ROWS, EXPECTED_DIM), dtype=np.float32)
    for k, v in enumerate(ids):
        p = os.path.join(hidden_dir, v + ".npy")
        a = np.load(p)
        if a.shape != (EXPECTED_ROWS, EXPECTED_DIM):
            raise SystemExit(f"ABORT: {os.path.basename(p)} has shape {a.shape}, "
                             f"expected {(EXPECTED_ROWS, EXPECTED_DIM)}")
        if not np.isfinite(a.astype(np.float32)).all():
            raise SystemExit(f"ABORT: non-finite values in {os.path.basename(p)}")
        out[k] = a.astype(np.float32)
    return out


def standardize(X):
    """Per-dimension z-scoring with the corpus's own unlabeled statistics."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    dead = sd <= 0
    sd = np.where(dead, 1.0, sd)
    return (X - mu) / sd, int(dead.sum())


def pca_axes(Xs, n_pc):
    """Top-n_pc PCA projections of the already-standardized matrix Xs."""
    Xc = Xs - Xs.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(n_pc, Vt.shape[0])
    proj = U[:, :k] * S[:k]
    var = (S ** 2) / max(Xc.shape[0] - 1, 1)
    evr = (var[:k] / var.sum()).tolist()
    return proj, evr


def orient(proj, z):
    """Flip each axis so its Spearman correlation with z is non-negative."""
    rhos = []
    for j in range(proj.shape[1]):
        rho = float(spearmanr(proj[:, j], z)[0])
        if not np.isfinite(rho):
            rho = 0.0
        if rho < 0:
            proj[:, j] = -proj[:, j]
            rho = -rho
        rhos.append(float(rho))
    return proj, rhos


# ------------------------------------------------------------------- probe ---
def logistic_fit(X, y, lam):
    """L2-penalised logistic regression, unpenalised intercept, L-BFGS."""
    n, d = X.shape
    t = np.where(np.asarray(y) == 1, 1.0, -1.0)

    def obj(theta):
        w, b = theta[:d], theta[d]
        m = t * (X @ w + b)
        # log(1 + exp(-m)) computed stably
        loss = np.logaddexp(0.0, -m).sum() + 0.5 * lam * float(w @ w)
        s = -t / (1.0 + np.exp(m))
        g = np.empty(d + 1)
        g[:d] = X.T @ s + lam * w
        g[d] = s.sum()
        return loss, g

    r = minimize(obj, np.zeros(d + 1), jac=True, method="L-BFGS-B",
                 options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-9})
    return r.x[:d], float(r.x[d]), bool(r.success), int(r.nit)


def loo_probe(X, y, lam):
    """Leave-one-out decision values; AUC over the held-out predictions."""
    n = X.shape[0]
    held = np.empty(n)
    nits, ok = [], True
    for i in range(n):
        keep = np.arange(n) != i
        w, b, success, nit = logistic_fit(X[keep], np.asarray(y)[keep], lam)
        held[i] = float(X[i] @ w + b)
        nits.append(nit)
        ok = ok and success
    return held, ok, nits


# ------------------------------------------------------------------ strata ---
def en_strata(ids):
    manifest = json.load(open(os.path.join(AV, "manifest.json")))
    codes = {}
    for p in sorted(glob.glob(os.path.join(AV, "coding", "*.jsonl"))):
        for line in open(p):
            line = line.strip()
            if line:
                r = json.loads(line)
                codes[r["item_id"]] = r
    target = [m for m in manifest
              if m["corpus"] == "EN" and m["stratum"] == "target_union_positive"]
    no_pt = [m["video_id"] for m in target
             if codes[m["item_id"]].get("protected_group_targeted") is False]
    pt = [m["video_id"] for m in target
          if codes[m["item_id"]].get("protected_group_targeted") is True]
    shipped = {x["Video_ID"]: x["Label"] for x in json.load(open(EN_ANN))}
    normals = [v for v in ids if shipped.get(v) == "Normal"]
    if (len(no_pt), len(pt), len(normals)) != (34, 15, 112):
        raise SystemExit(f"ABORT: EN strata are {len(no_pt)}/{len(pt)}/"
                         f"{len(normals)}, expected 34/15/112")
    idset = set(ids)
    for v in no_pt + pt:
        if v not in idset:
            raise SystemExit("ABORT: a coded EN video has no hidden state")
    return no_pt, pt, normals


def hcs_strata(ids):
    labels = {}
    with open(HCS_ANN) as f:
        for row in csv.DictReader(f):
            labs = ast.literal_eval(row["Video-Level Label"])
            unknown = set(labs) - set(HCS_IDX)
            if unknown:
                raise SystemExit(f"ABORT: unknown HateClipSeg label {unknown}")
            labels[row["Video Id"].strip()] = labs
    missing = [v for v in ids if v not in labels]
    if missing:
        raise SystemExit(f"ABORT: {len(missing)} scored HateClipSeg videos have "
                         "no shipped annotation row")
    insulting_only = [v for v in ids if labels[v] == ["insulting"]]
    clean_normal = [v for v in ids if not (set(labels[v]) & HCS_OFFENSIVE)]
    union_pos = [v for v in ids if set(labels[v]) & HCS_OFFENSIVE]
    hateful = [v for v in ids if "hateful" in labels[v]]
    if (len(insulting_only), len(clean_normal)) != (59, 50):
        raise SystemExit(f"ABORT: HateClipSeg strata are {len(insulting_only)}/"
                         f"{len(clean_normal)}, expected 59/50")
    return insulting_only, clean_normal, union_pos, hateful


# -------------------------------------------------------------------- main ---
def main():
    # ---- MHClip-EN -------------------------------------------------------
    z_en = load_z(EN_SCORES)
    en_files = sorted(glob.glob(os.path.join(EN_HIDDEN, "*.npy")))
    en_ids = [os.path.splitext(os.path.basename(p))[0] for p in en_files]
    if len(en_ids) != 161 or set(en_ids) != set(z_en):
        raise SystemExit(f"ABORT: EN hidden={len(en_ids)} scores={len(z_en)}; "
                         "id sets must match at 161")
    H_en = load_hidden(EN_HIDDEN, en_ids)
    zv_en = np.array([z_en[v] for v in en_ids])
    no_pt, pt, normals = en_strata(en_ids)
    pos_en = {v: i for i, v in enumerate(en_ids)}
    i_no_pt = [pos_en[v] for v in no_pt]
    i_pt = [pos_en[v] for v in pt]
    i_norm = [pos_en[v] for v in normals]

    # ---- HateClipSeg -----------------------------------------------------
    z_hcs = load_z(HCS_SCORES)
    hcs_files = sorted(glob.glob(os.path.join(HCS_HIDDEN, "*.npy")))
    hcs_ids = [os.path.splitext(os.path.basename(p))[0] for p in hcs_files]
    if len(hcs_ids) != 394 or set(hcs_ids) != set(z_hcs):
        raise SystemExit(f"ABORT: HCS hidden={len(hcs_ids)} scores={len(z_hcs)}; "
                         "id sets must match at 394")
    H_hcs = load_hidden(HCS_HIDDEN, hcs_ids)
    zv_hcs = np.array([z_hcs[v] for v in hcs_ids])
    ins, clean, union_pos, hateful = hcs_strata(hcs_ids)
    pos_hcs = {v: i for i, v in enumerate(hcs_ids)}
    i_ins = [pos_hcs[v] for v in ins]
    i_clean = [pos_hcs[v] for v in clean]
    i_union = [pos_hcs[v] for v in union_pos]
    i_hateful = [pos_hcs[v] for v in hateful]

    res = {
        "protocol": "docs/duplex/PREREG_readout_bottleneck_killtest.md",
        "compute": "CPU only, no model call; frozen hidden states read from disk",
        "frozen_constants": {
            "layers": LAYERS,
            "n_components_per_layer": N_PC,
            "probe_layer": PROBE_LAYER,
            "probe_l2_lambda": PROBE_LAMBDA,
            "probe_lambda_rule": ("fixed at 1.0 before any result was seen; "
                                  "never tuned on any outcome"),
            "standardisation": ("per-dimension mean and standard deviation of "
                                "the corpus's own hidden states, all videos, "
                                "no labels; applied before PCA and before the "
                                "probe"),
            "sign_rule": "flip axis if Spearman with that corpus's z is negative",
            "clause_floors": {"clause1": CLAUSE1_FLOOR, "clause2": CLAUSE2_FLOOR,
                              "clause3": CLAUSE3_FLOOR},
        },
        "data": {
            "en_corpus": "MHClip_EN test_clean",
            "en_n_videos": len(en_ids),
            "en_hidden_shape": [EXPECTED_ROWS, EXPECTED_DIM],
            "en_strata": {"no_protected_target_positives": len(no_pt),
                          "protected_target_positives": len(pt),
                          "shipped_normals": len(normals)},
            "hcs_corpus": "HateClipSeg",
            "hcs_n_videos": len(hcs_ids),
            "hcs_hidden_shape": [EXPECTED_ROWS, EXPECTED_DIM],
            "hcs_strata": {"insulting_only": len(ins), "clean_normal": len(clean),
                           "offensive_union": len(union_pos),
                           "hateful_any": len(hateful)},
            "lexicons_json": "quarantined, never read",
        },
    }

    # ---- reference AUCs of the shipped scalar readout ---------------------
    res["reference_z_auc"] = {
        "en_no_target_vs_normal": auc_pos_neg(zv_en[i_no_pt], zv_en[i_norm]),
        "en_protected_vs_normal": auc_pos_neg(zv_en[i_pt], zv_en[i_norm]),
        "en_no_target_vs_protected": auc_pos_neg(zv_en[i_no_pt], zv_en[i_pt]),
        "hcs_insulting_only_vs_clean_normal": auc_pos_neg(zv_hcs[i_ins],
                                                          zv_hcs[i_clean]),
        "hcs_union_vs_clean_normal": auc_pos_neg(zv_hcs[i_union], zv_hcs[i_clean]),
        "prereg_quoted": {"en_no_target": 0.749, "en_protected": 0.866,
                          "hcs_insulting_only": 0.637},
    }

    # ---- unsupervised axes, both corpora ---------------------------------
    en_axes, hcs_axes = {}, {}
    en_table, hcs_table = {}, {}
    for layer in LAYERS:
        Xs, dead = standardize(H_en[:, layer, :])
        proj, evr = pca_axes(Xs, N_PC)
        proj, rhos = orient(proj, zv_en)
        en_axes[layer] = proj
        rows = []
        for j in range(proj.shape[1]):
            a = proj[:, j]
            rows.append({
                "pc": j + 1,
                "explained_variance_ratio": float(evr[j]),
                "spearman_with_z": rhos[j],
                "auc_no_target_vs_normal": auc_pos_neg(a[i_no_pt], a[i_norm]),
                "auc_protected_vs_normal": auc_pos_neg(a[i_pt], a[i_norm]),
                "auc_no_target_vs_protected": auc_pos_neg(a[i_no_pt], a[i_pt]),
            })
        en_table[str(layer)] = {"n_zero_variance_dims": dead, "axes": rows}

        Xs2, dead2 = standardize(H_hcs[:, layer, :])
        proj2, evr2 = pca_axes(Xs2, N_PC)
        proj2, rhos2 = orient(proj2, zv_hcs)
        hcs_axes[layer] = proj2
        rows2 = []
        for j in range(proj2.shape[1]):
            a = proj2[:, j]
            rows2.append({
                "pc": j + 1,
                "explained_variance_ratio": float(evr2[j]),
                "spearman_with_z": rhos2[j],
                "auc_insulting_only_vs_clean_normal": auc_pos_neg(a[i_ins],
                                                                  a[i_clean]),
                "auc_union_vs_clean_normal": auc_pos_neg(a[i_union], a[i_clean]),
                "auc_hateful_vs_clean_normal": auc_pos_neg(a[i_hateful],
                                                           a[i_clean]),
            })
        hcs_table[str(layer)] = {"n_zero_variance_dims": dead2, "axes": rows2}

    res["en_axis_table"] = en_table
    res["hcs_axis_table"] = hcs_table

    # ---- clause 1: supervised ceiling ------------------------------------
    Xs27, _ = standardize(H_en[:, PROBE_LAYER, :])
    idx = i_no_pt + i_pt
    Xp = Xs27[idx]
    yp = [1] * len(i_no_pt) + [0] * len(i_pt)
    held, ok, nits = loo_probe(Xp, yp, PROBE_LAMBDA)
    probe_auc = auc(held, yp)
    # in-sample fit, reported only to show the probe can memorise the split
    w, b, _, _ = logistic_fit(Xp, yp, PROBE_LAMBDA)
    insample_auc = auc((Xp @ w + b).tolist(), yp)
    clause1 = bool(probe_auc >= CLAUSE1_FLOOR)
    res["clause1_information_exists"] = {
        "rule": (f"logistic probe, EN layer {PROBE_LAYER}, no-target positives "
                 "vs protected-target positives, leave-one-out CV, "
                 f"AUC >= {CLAUSE1_FLOOR}"),
        "n_pos_no_target": len(i_no_pt),
        "n_neg_protected_target": len(i_pt),
        "n_folds": len(idx),
        "l2_lambda": PROBE_LAMBDA,
        "loo_auc": probe_auc,
        "in_sample_auc": insample_auc,
        "all_folds_converged": ok,
        "median_lbfgs_iterations": float(np.median(nits)),
        "passes": clause1,
    }

    # ---- clause 2: label-free access on EN --------------------------------
    cand = []
    for layer in LAYERS:
        for row in en_table[str(layer)]["axes"]:
            cand.append((row["auc_no_target_vs_normal"], layer, row["pc"], row))
    cand.sort(key=lambda t: (-t[0], t[1], t[2]))
    best_auc, best_layer, best_pc, best_row = cand[0]
    clause2 = bool(best_auc >= CLAUSE2_FLOOR)
    res["clause2_label_free_access"] = {
        "rule": (f"best of {len(cand)} unsupervised EN axes reaches AUC >= "
                 f"{CLAUSE2_FLOOR} on no-target positives vs shipped Normals"),
        "n_candidates": len(cand),
        "best_layer": best_layer,
        "best_pc": best_pc,
        "best_auc_no_target_vs_normal": best_auc,
        "best_auc_protected_vs_normal": best_row["auc_protected_vs_normal"],
        "best_auc_no_target_vs_protected": best_row["auc_no_target_vs_protected"],
        "best_spearman_with_z": best_row["spearman_with_z"],
        "best_explained_variance_ratio": best_row["explained_variance_ratio"],
        "runner_up": [{"layer": l, "pc": p, "auc_no_target_vs_normal": a}
                      for a, l, p, _ in cand[1:6]],
        "n_axes_at_or_above_floor": int(sum(1 for a, _, _, _ in cand
                                            if a >= CLAUSE2_FLOOR)),
        "n_axes_above_z": int(sum(
            1 for a, _, _, _ in cand
            if a > res["reference_z_auc"]["en_no_target_vs_normal"])),
        "passes": clause2,
    }

    # ---- clause 3: cross-corpus replication -------------------------------
    hcs_rows = hcs_table[str(best_layer)]["axes"][:N_PC_CLAUSE3]
    hcs_best = max(hcs_rows, key=lambda r: r["auc_insulting_only_vs_clean_normal"])
    clause3 = bool(hcs_best["auc_insulting_only_vs_clean_normal"] >= CLAUSE3_FLOOR)
    res["clause3_cross_corpus_replication"] = {
        "rule": (f"at the clause-2 winner's layer, one of HateClipSeg's top-"
                 f"{N_PC_CLAUSE3} PCA axes reaches AUC >= {CLAUSE3_FLOOR} on "
                 "insulting-only vs clean-normal"),
        "layer": best_layer,
        "candidates": [{"pc": r["pc"],
                        "auc_insulting_only_vs_clean_normal":
                            r["auc_insulting_only_vs_clean_normal"],
                        "spearman_with_z": r["spearman_with_z"]}
                       for r in hcs_rows],
        "best_pc": hcs_best["pc"],
        "best_auc": hcs_best["auc_insulting_only_vs_clean_normal"],
        "passes": clause3,
    }

    # ---- descriptive: layer 36 alone, against the z bottleneck ------------
    l36 = en_table["36"]["axes"]
    best36 = max(l36, key=lambda r: r["auc_no_target_vs_normal"])
    hcs36 = hcs_table["36"]["axes"]
    hcs_best36 = max(hcs36, key=lambda r: r["auc_insulting_only_vs_clean_normal"])
    res["descriptive_layer36_only"] = {
        "note": ("layer 36 is the row the scalar readout is taken from, so the "
                 "gap between its best unsupervised axis and z measures what "
                 "the single logit contrast discards at its own depth"),
        "en_best_pc": best36["pc"],
        "en_best_auc_no_target_vs_normal": best36["auc_no_target_vs_normal"],
        "en_z_auc_no_target_vs_normal":
            res["reference_z_auc"]["en_no_target_vs_normal"],
        "en_best_minus_z": (best36["auc_no_target_vs_normal"]
                            - res["reference_z_auc"]["en_no_target_vs_normal"]),
        "en_best_auc_protected_vs_normal": best36["auc_protected_vs_normal"],
        "hcs_best_pc": hcs_best36["pc"],
        "hcs_best_auc_insulting_only_vs_clean_normal":
            hcs_best36["auc_insulting_only_vs_clean_normal"],
        "hcs_z_auc_insulting_only_vs_clean_normal":
            res["reference_z_auc"]["hcs_insulting_only_vs_clean_normal"],
    }

    # ---- descriptive: construct specificity of qualifying axes ------------
    qual = [(a, l, p, r) for a, l, p, r in cand if a >= CLAUSE2_FLOOR]
    res["descriptive_qualifying_axes"] = {
        "floor": CLAUSE2_FLOOR,
        "n": len(qual),
        "axes": [{"layer": l, "pc": p, "auc_no_target_vs_normal": a,
                  "auc_protected_vs_normal": r["auc_protected_vs_normal"],
                  "spearman_with_z": r["spearman_with_z"]}
                 for a, l, p, r in qual],
    }

    verdict = "SURVIVES" if (clause1 and clause2 and clause3) else "DEAD"
    res["verdict"] = verdict
    res["verdict_reason"] = {
        "clause1_passes": clause1, "clause2_passes": clause2,
        "clause3_passes": clause3,
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
        f.write("\n")
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("en_axis_table", "hcs_axis_table")}, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
