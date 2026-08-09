"""Spec-displacement pilot: CPU analysis of the layer-27 displacement between
two committed policy specs.

Pre-registration: docs/duplex/PREREG_spec_displacement_pilot.md. Every constant,
threshold, stratum, control and decision rule below is frozen there and was
committed before the first union-arm or off-construct-arm forward pass.

CPU only. Hidden states are read from disk and nothing is rescored.

Writes results/spec_displacement/results.json. No video id, title or transcript
text reaches the output.

Usage:
  python scripts/duplex/spec_displacement_analyze.py
"""

import ast
import csv
import glob
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(ROOT, "src", "our_method"))
os.environ.setdefault("HVD_DATA_ROOT", "/home/jehc223/data")

from crossbench_analyze import kde_valley  # noqa: E402
from anchored_operating_point import (  # noqa: E402
    _fit_gmm,
    decision_boundaries,
    macro_f1,
    posterior_pos,
)

# ------------------------------------------------------------------ frozen --
SEED = 20260808
LAYERS_REPORTED = [18, 27, 36]
PRIMARY_LAYER = 27
EXPECTED_ROWS = 37
EXPECTED_DIM = 4096

C1A_FLOOR = {"c1": 0.72, "c2": 0.74}      # MHClip-EN flip separation
C1B_FLOOR = {"c1": 0.70, "c2": 0.72}      # HateClipSeg flip separation
C2_MARGIN = 0.10                          # real minus shuffled
C3_CEIL = 0.60                            # off-construct arm
C4_UNION_FLOOR = 0.6522
C4_STRICT_FLOOR = 0.6767

PROBE_LAMBDA = 1.0

STRICT_HIDDEN = {
    "mhclip_en": os.path.join(ROOT, "results", "testruns", "mhclip_en",
                              "judge_8b", "hidden"),
    "hateclipseg": os.path.join(ROOT, "results", "hateclipseg", "judge_8b",
                                "hidden"),
}
STRICT_SCORES = {
    "mhclip_en": os.path.join(ROOT, "results", "testruns", "mhclip_en",
                              "judge_8b", "scores.jsonl"),
    "hateclipseg": os.path.join(ROOT, "results", "hateclipseg", "judge_8b",
                                "scores.jsonl"),
}
NEW_ROOT = os.path.join(ROOT, "results", "spec_displacement")
DUAL_AXIS_Z = os.path.join(ROOT, "results", "dual_axis", "z_off_scores.jsonl")
AV = os.path.join(ROOT, "results", "annotation_validity")
EN_ANN = "/home/jehc223/data/Multihateclip/English/annotation(new).json"
HCS_ANN = os.path.join(ROOT, "idea-stage", "pilots", "b1_coverage_audit",
                       "data", "video_level_annotation.csv")
OUT = os.path.join(NEW_ROOT, "results.json")

HCS_IDX = ["normal", "hateful", "insulting", "sexual", "violence", "harm"]
HCS_OFFENSIVE = set(HCS_IDX[1:])

# Committed references the run self-checks against before using them.
REF_VALLEY_Z = -5.4449999999999985
REF_VALLEY_UNION_F1 = 0.6521892655367232
REF_VALLEY_STRICT_F1 = 0.4478965766634523
REF_ANCHORED_STRICT_F1 = 0.6767466649439582


# ----------------------------------------------------------------- helpers --
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


def load_hidden(hidden_dir, ids, layers):
    """(n, len(layers), 4096) float32 in the given id order; shape verified."""
    out = np.empty((len(ids), len(layers), EXPECTED_DIM), dtype=np.float32)
    for k, v in enumerate(ids):
        a = np.load(os.path.join(hidden_dir, v + ".npy"))
        if a.shape != (EXPECTED_ROWS, EXPECTED_DIM):
            raise SystemExit(f"ABORT: {v}.npy has shape {a.shape}")
        f = a[layers].astype(np.float32)
        if not np.isfinite(f).all():
            raise SystemExit(f"ABORT: non-finite values in {v}.npy")
        out[k] = f
    return out


def derangement(n, seed):
    """A permutation with no fixed point, drawn by rejection from one stream."""
    rng = np.random.default_rng(seed)
    for _ in range(10000):
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p
    raise SystemExit("ABORT: no derangement drawn")


# ------------------------------------------------------------------ strata --
def en_strata(ids):
    codes = {}
    for p in sorted(glob.glob(os.path.join(AV, "coding", "*.jsonl"))):
        for line in open(p):
            line = line.strip()
            if line:
                r = json.loads(line)
                codes[r["item_id"]] = r
    manifest = json.load(open(os.path.join(AV, "manifest.json")))
    target = [m for m in manifest
              if m["corpus"] == "EN" and m["stratum"] == "target_union_positive"]
    no_pt = [m["video_id"] for m in target
             if codes[m["item_id"]].get("protected_group_targeted") is False]
    pt = [m["video_id"] for m in target
          if codes[m["item_id"]].get("protected_group_targeted") is True]
    shipped = {x["Video_ID"]: x["Label"] for x in json.load(open(EN_ANN))}
    normals = [v for v in ids if shipped.get(v) == "Normal"]
    if (len(no_pt), len(pt), len(normals)) != (34, 15, 112):
        raise SystemExit(f"ABORT: EN strata {len(no_pt)}/{len(pt)}/{len(normals)}")
    idset = set(ids)
    if any(v not in idset for v in no_pt + pt):
        raise SystemExit("ABORT: a coded EN video has no hidden state")
    return no_pt, pt, normals


def hcs_labels(ids):
    labels = {}
    with open(HCS_ANN) as f:
        for row in csv.DictReader(f):
            labs = ast.literal_eval(row["Video-Level Label"])
            if set(labs) - set(HCS_IDX):
                raise SystemExit("ABORT: unknown HateClipSeg label")
            labels[row["Video Id"].strip()] = labs
    if any(v not in labels for v in ids):
        raise SystemExit("ABORT: a scored HateClipSeg video has no annotation row")
    union = np.array([bool(set(labels[v]) & HCS_OFFENSIVE) for v in ids])
    strict = np.array(["hateful" in labels[v] for v in ids])
    flip = union & ~strict
    if (int(union.sum()), int(strict.sum()), int(flip.sum())) != (344, 180, 164):
        raise SystemExit(f"ABORT: HCS strata {int(union.sum())}/{int(strict.sum())}"
                         f"/{int(flip.sum())}, expected 344/180/164")
    return union, strict, flip


# ------------------------------------------------------------------- probe --
def logistic_fit(X, y, lam):
    n, d = X.shape
    t = np.where(np.asarray(y) == 1, 1.0, -1.0)

    def obj(theta):
        w, b = theta[:d], theta[d]
        m = t * (X @ w + b)
        loss = np.logaddexp(0.0, -m).sum() + 0.5 * lam * float(w @ w)
        s = -t / (1.0 + np.exp(m))
        g = np.empty(d + 1)
        g[:d] = X.T @ s + lam * w
        g[d] = s.sum()
        return loss, g

    r = minimize(obj, np.zeros(d + 1), jac=True, method="L-BFGS-B",
                 options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-9})
    return r.x[:d], float(r.x[d]), bool(r.success)


# ------------------------------------------------------- displacement maths --
def decompose(H_a, H_b, keep_directions):
    """Displacement H_a - H_b at one layer, its carrier and its residuals.

    Returns per-video scores and scalar summaries. The 4096-vectors are kept
    only for the layers that are reported, so the whole sweep fits in memory.
    """
    dh = H_a - H_b
    m = dh.mean(axis=0)
    nm = float(np.linalg.norm(m))
    d_unit = m / nm if nm > 0 else m
    r = dh - m
    c1 = dh @ d_unit
    # PC1 of the residuals through the n x n Gram matrix. With n << 4096 this
    # is the same top component as a full SVD at a fraction of the cost, and it
    # never forms the 4096-column right-singular matrix.
    G = r @ r.T
    evals, evecs = np.linalg.eigh(G)
    lam1 = float(max(evals[-1], 0.0))
    u1 = evecs[:, -1]
    s1 = math.sqrt(lam1)
    pc1 = u1 * s1
    tot = float(np.sum(np.maximum(evals, 0.0)))
    out = dict(
        norm_m=nm,
        c1=c1,
        c2_raw=pc1,
        c2_evr=float(lam1 / tot) if tot > 0 else 0.0,
        carrier_energy=float(nm ** 2),
        mean_residual_energy=float(np.mean(np.diag(G))),
        mean_displacement_norm=float(np.mean(np.linalg.norm(dh, axis=1))),
    )
    if keep_directions:
        out["d"] = d_unit
        loading = (r.T @ u1) / s1 if s1 > 0 else np.zeros(r.shape[1])
        n_l = float(np.linalg.norm(loading))
        out["c2_loading"] = loading / n_l if n_l > 0 else loading
    return out


def orient_by_z(dec, z):
    """The reported orientation: flip c2 so its Spearman with z_strict is >= 0."""
    rho = float(spearmanr(dec["c2_raw"], z)[0])
    if not np.isfinite(rho):
        rho = 0.0
    sign = -1.0 if rho < 0 else 1.0
    return sign, abs(rho)


def two_sided(a):
    """max(AUC, 1 - AUC) with the direction that achieved it."""
    if a is None:
        return None, 0
    return (a, +1) if a >= 0.5 else (1.0 - a, -1)


# -------------------------------------------------------------------- main --
def main():
    layers_all = list(range(EXPECTED_ROWS))
    res = {
        "protocol": "docs/duplex/PREREG_spec_displacement_pilot.md",
        "compute": "CPU only; hidden states read from disk, nothing rescored",
        "frozen_constants": {
            "seed": SEED,
            "primary_layer": PRIMARY_LAYER,
            "layers_reported": LAYERS_REPORTED,
            "c1a_floor": C1A_FLOOR, "c1b_floor": C1B_FLOOR,
            "c2_margin": C2_MARGIN, "c3_ceiling": C3_CEIL,
            "c4_union_floor": C4_UNION_FLOOR, "c4_strict_floor": C4_STRICT_FLOOR,
            "probe_l2_lambda": PROBE_LAMBDA,
            "space": "raw float32 hidden-state space, no standardisation",
        },
    }

    corpora = {}
    for slug, expect in (("mhclip_en", 161), ("hateclipseg", 394)):
        z_strict = load_z(STRICT_SCORES[slug])
        ids = sorted(os.path.splitext(os.path.basename(p))[0]
                     for p in glob.glob(os.path.join(STRICT_HIDDEN[slug], "*.npy")))
        if len(ids) != expect or set(ids) != set(z_strict):
            raise SystemExit(f"ABORT: {slug} strict arm {len(ids)}/{len(z_strict)}")
        arms = {"strict": load_hidden(STRICT_HIDDEN[slug], ids, layers_all)}
        zs = {"strict": np.array([z_strict[v] for v in ids])}
        for arm in ("union", "spam"):
            hd = os.path.join(NEW_ROOT, slug, arm, "hidden")
            zp = os.path.join(NEW_ROOT, slug, arm, "scores.jsonl")
            za = load_z(zp)
            have = {os.path.splitext(os.path.basename(p))[0]
                    for p in glob.glob(os.path.join(hd, "*.npy"))}
            if set(ids) != have or set(ids) != set(za):
                raise SystemExit(f"ABORT: {slug}/{arm} coverage mismatch "
                                 f"(hidden {len(have)}, z {len(za)}, want {len(ids)})")
            arms[arm] = load_hidden(hd, ids, layers_all)
            zs[arm] = np.array([za[v] for v in ids])
            if arm == "union":
                ntok = {}
                with open(zp) as f:
                    for line in f:
                        if line.strip():
                            r = json.loads(line)
                            ntok[r["video_id"]] = r.get("n_tokens_total")
                n_tokens = np.array([ntok[v] for v in ids], dtype=float)
        corpora[slug] = dict(ids=ids, arms=arms, z=zs, n_tokens=n_tokens,
                             perm=derangement(len(ids), SEED))
        print(f"loaded {slug}: {len(ids)} videos x 3 arms")

    # ---- integrity check against the dual-axis union z --------------------
    da = load_z(DUAL_AXIS_Z)
    en = corpora["mhclip_en"]
    common = [v for v in en["ids"] if v in da]
    zz = np.array([da[v] for v in common])
    zn = en["z"]["union"][[en["ids"].index(v) for v in common]]
    res["dual_axis_reproduction"] = {
        "n_common": len(common),
        "spearman": float(spearmanr(zz, zn)[0]),
        "pearson": float(np.corrcoef(zz, zn)[0, 1]),
        "max_abs_diff": float(np.max(np.abs(zz - zn))),
        "mean_abs_diff": float(np.mean(np.abs(zz - zn))),
        "gating": False,
    }

    # ---- strata -----------------------------------------------------------
    no_pt, pt, normals = en_strata(en["ids"])
    pos_en = {v: i for i, v in enumerate(en["ids"])}
    i_no_pt = [pos_en[v] for v in no_pt]
    i_pt = [pos_en[v] for v in pt]
    i_norm = [pos_en[v] for v in normals]

    hcs = corpora["hateclipseg"]
    y_union, y_strict, y_flip = hcs_labels(hcs["ids"])
    i_flip = np.where(y_flip)[0]
    i_strictpos = np.where(y_strict)[0]

    res["data"] = {
        "mhclip_en": {"n": len(en["ids"]), "no_protected_target_positives": len(no_pt),
                      "protected_target_positives": len(pt),
                      "shipped_normals": len(normals)},
        "hateclipseg": {"n": len(hcs["ids"]), "union_positive": int(y_union.sum()),
                        "strict_positive": int(y_strict.sum()),
                        "flip_union_pos_strict_neg": int(y_flip.sum())},
        "lexicons_json": "quarantined, never read",
    }

    # ---- contrasts, all layers -------------------------------------------
    # contrast -> per corpus per layer decomposition and both readouts
    def contrast_arms(cp, kind, layer):
        A = cp["arms"]
        if kind == "real":
            return A["union"][:, layer], A["strict"][:, layer]
        if kind == "shuffled":
            return A["union"][cp["perm"], layer], A["strict"][:, layer]
        if kind == "spam":
            return A["spam"][:, layer], A["strict"][:, layer]
        raise ValueError(kind)

    store = {}   # (slug, kind, layer) -> dict
    for slug, cp in corpora.items():
        for kind in ("real", "shuffled", "spam"):
            for layer in layers_all:
                a, b = contrast_arms(cp, kind, layer)
                dec = decompose(a, b, layer in LAYERS_REPORTED)
                sign, rho = orient_by_z(dec, cp["z"]["strict"])
                dec["c2_sign_by_z"] = sign
                dec["c2_spearman_z"] = rho
                store[(slug, kind, layer)] = dec

    # ---- the flip-separation AUCs -----------------------------------------
    def flip_auc(slug, kind, layer, readout, direction=+1):
        dec = store[(slug, kind, layer)]
        s = dec["c1"] if readout == "c1" else dec["c2_raw"] * dec["c2_sign_by_z"]
        s = s * direction
        if slug == "mhclip_en":
            return auc_pos_neg(s[i_no_pt], s[i_pt])
        return auc_pos_neg(s[i_flip], s[i_strictpos])

    # c2's operative direction: chosen once on MHClip-EN layer 27, real
    # contrast, two-sided; then applied unchanged everywhere.
    a_en_c2 = flip_auc("mhclip_en", "real", PRIMARY_LAYER, "c2", +1)
    _, c2_dir = two_sided(a_en_c2)
    res["c2_orientation"] = {
        "reported_sign_rule": "flip so Spearman(c2, z_strict) >= 0",
        "operative_direction_vs_reported": int(c2_dir),
        "chosen_on": "MHClip-EN layer 27, real contrast, two-sided",
        "auc_before_direction_choice": a_en_c2,
    }

    def score_auc(slug, kind, layer, readout):
        if readout == "c1":
            return flip_auc(slug, kind, layer, "c1", +1)
        return flip_auc(slug, kind, layer, "c2", c2_dir)

    # ---- clause tables ----------------------------------------------------
    clauses = {}
    for readout in ("c1", "c2"):
        real_en = score_auc("mhclip_en", "real", PRIMARY_LAYER, readout)
        real_hcs = score_auc("hateclipseg", "real", PRIMARY_LAYER, readout)
        shuf_en = score_auc("mhclip_en", "shuffled", PRIMARY_LAYER, readout)
        shuf_hcs = score_auc("hateclipseg", "shuffled", PRIMARY_LAYER, readout)
        spam_en = score_auc("mhclip_en", "spam", PRIMARY_LAYER, readout)
        spam_hcs = score_auc("hateclipseg", "spam", PRIMARY_LAYER, readout)
        if readout == "c2":
            # the shuffled placebo is scored two-sided, which makes it harder
            # for the real contrast to clear the margin
            shuf_en = max(shuf_en, 1.0 - shuf_en)
            shuf_hcs = max(shuf_hcs, 1.0 - shuf_hcs)
            spam_en = max(spam_en, 1.0 - spam_en)
            spam_hcs = max(spam_hcs, 1.0 - spam_hcs)
        clauses[readout] = {
            "C1a_mhclip_en": {"auc": real_en, "floor": C1A_FLOOR[readout],
                              "pass": real_en >= C1A_FLOOR[readout]},
            "C1b_hateclipseg": {"auc": real_hcs, "floor": C1B_FLOOR[readout],
                                "pass": real_hcs >= C1B_FLOOR[readout]},
            "C2_shuffled": {
                "mhclip_en": {"real": real_en, "shuffled": shuf_en,
                              "margin": real_en - shuf_en},
                "hateclipseg": {"real": real_hcs, "shuffled": shuf_hcs,
                                "margin": real_hcs - shuf_hcs},
                "pass": (real_en - shuf_en >= C2_MARGIN
                         and real_hcs - shuf_hcs >= C2_MARGIN)},
            "C3_off_construct": {
                "mhclip_en": spam_en, "hateclipseg": spam_hcs,
                "ceiling": C3_CEIL,
                "pass": spam_en <= C3_CEIL and spam_hcs <= C3_CEIL},
        }

    # ---- C4: the frozen composition rule on HateClipSeg -------------------
    z_hcs_strict = hcs["z"]["strict"]
    v = kde_valley(z_hcs_strict)
    t_z = v["value"]
    selfcheck = {
        "valley_threshold": t_z,
        "reference": REF_VALLEY_Z,
        "matches_reference": t_z is not None and abs(t_z - REF_VALLEY_Z) < 1e-6,
    }
    base = z_hcs_strict >= t_z
    selfcheck["valley_macro_f1_union"] = macro_f1(y_union, base)["macro_f1"]
    selfcheck["valley_macro_f1_strict"] = macro_f1(y_strict, base)["macro_f1"]
    selfcheck["reproduces_committed_valley_f1"] = (
        abs(selfcheck["valley_macro_f1_union"] - REF_VALLEY_UNION_F1) < 2e-3
        and abs(selfcheck["valley_macro_f1_strict"] - REF_VALLEY_STRICT_F1) < 2e-3)
    res["c4_selfcheck"] = selfcheck

    def c_threshold(c):
        """Frozen recipe: KDE valley first, free two-component mixture on
        failure, taking the posterior-0.5 crossing between the component
        means."""
        kv = kde_valley(c)
        if kv.get("value") is not None:
            return float(kv["value"]), "kde_valley", kv
        fit = _fit_gmm(np.asarray(c, dtype=float), SEED)
        bs = decision_boundaries(fit)
        lo, hi = sorted([fit["mu_neg"], fit["mu_pos"]])
        inside = [b for b in bs if lo <= b <= hi]
        if inside:
            t = float(inside[0])
        elif bs:
            t = float(min(bs, key=lambda x: abs(x - float(np.median(c)))))
        else:
            return None, "gmm_no_boundary", fit
        return t, "gmm_fallback", fit

    for readout in ("c1", "c2"):
        dec = store[("hateclipseg", "real", PRIMARY_LAYER)]
        c = (dec["c1"] if readout == "c1"
             else dec["c2_raw"] * dec["c2_sign_by_z"] * c2_dir)
        t_c, recipe, detail = c_threshold(c)
        if t_c is None:
            clauses[readout]["C4_composition"] = {
                "pass": False, "reason": "no c threshold could be found",
                "recipe": recipe}
            continue
        flip_set = c >= t_c
        dec_union = base
        dec_strict = base & ~flip_set
        m_union_on_union = macro_f1(y_union, dec_union)
        m_strict_on_strict = macro_f1(y_strict, dec_strict)
        m_strict_on_union = macro_f1(y_union, dec_strict)
        m_union_on_strict = macro_f1(y_strict, dec_union)
        crossover = (m_strict_on_strict["macro_f1"] > m_union_on_strict["macro_f1"]
                     and m_union_on_union["macro_f1"] > m_strict_on_union["macro_f1"])
        clauses[readout]["C4_composition"] = {
            "c_threshold": t_c,
            "threshold_recipe": recipe,
            "kde_modes": (detail.get("n_grid_local_maxima")
                          if isinstance(detail, dict) else None),
            "n_base_set": int(base.sum()),
            "n_flip_set": int(flip_set.sum()),
            "n_strict_decision_positive": int(dec_strict.sum()),
            "union_collapse": {"macro_f1": m_union_on_union["macro_f1"],
                               "floor": C4_UNION_FLOOR,
                               "detail": m_union_on_union},
            "strict_collapse": {"macro_f1": m_strict_on_strict["macro_f1"],
                                "floor": C4_STRICT_FLOOR,
                                "detail": m_strict_on_strict},
            "crossover_control": {
                "strict_decision_on_strict": m_strict_on_strict["macro_f1"],
                "union_decision_on_strict": m_union_on_strict["macro_f1"],
                "union_decision_on_union": m_union_on_union["macro_f1"],
                "strict_decision_on_union": m_strict_on_union["macro_f1"],
                "pass": bool(crossover)},
            "pass": bool(m_union_on_union["macro_f1"] >= C4_UNION_FLOOR
                         and m_strict_on_strict["macro_f1"] >= C4_STRICT_FLOOR
                         and crossover),
        }

    for readout in ("c1", "c2"):
        cl = clauses[readout]
        cl["verdict"] = ("SURVIVES" if all(
            cl[k]["pass"] for k in ("C1a_mhclip_en", "C1b_hateclipseg",
                                    "C2_shuffled", "C3_off_construct",
                                    "C4_composition")) else "DEAD")
    res["clauses"] = clauses
    res["verdict"] = ("SURVIVES" if any(clauses[r]["verdict"] == "SURVIVES"
                                        for r in ("c1", "c2")) else "DEAD")

    # ---- descriptive: energy split, correlations, layer sweep ------------
    energy = {}
    for slug in corpora:
        energy[slug] = {}
        for kind in ("real", "shuffled", "spam"):
            rows = {}
            for layer in layers_all:
                d = store[(slug, kind, layer)]
                ec, er = d["carrier_energy"], d["mean_residual_energy"]
                rows[str(layer)] = {
                    "carrier_energy": ec, "mean_residual_energy": er,
                    "carrier_share": ec / (ec + er) if ec + er > 0 else None,
                    "residual_pc1_variance_share": d["c2_evr"],
                    "mean_displacement_norm": d["mean_displacement_norm"],
                }
            energy[slug][kind] = rows
    res["energy_split"] = energy

    corr = {}
    for slug, cp in corpora.items():
        d = store[(slug, "real", PRIMARY_LAYER)]
        c2 = d["c2_raw"] * d["c2_sign_by_z"] * c2_dir
        corr[slug] = {
            "spearman_c1_z_strict": float(spearmanr(d["c1"], cp["z"]["strict"])[0]),
            "spearman_c1_z_union": float(spearmanr(d["c1"], cp["z"]["union"])[0]),
            "spearman_c2_z_strict": float(spearmanr(c2, cp["z"]["strict"])[0]),
            "spearman_c2_z_union": float(spearmanr(c2, cp["z"]["union"])[0]),
            "spearman_c1_c2": float(spearmanr(d["c1"], c2)[0]),
            "spearman_z_strict_z_union":
                float(spearmanr(cp["z"]["strict"], cp["z"]["union"])[0]),
            # nuisance check: is the displacement just prompt length?
            "spearman_c1_n_prompt_tokens":
                float(spearmanr(d["c1"], cp["n_tokens"])[0]),
            "spearman_c2_n_prompt_tokens":
                float(spearmanr(c2, cp["n_tokens"])[0]),
        }
    res["correlations"] = corr

    # The hypothesis is middle-peaked: c large on the flip stratum, small on
    # both protected-target hate and benign content. These profiles show it
    # directly and are descriptive, not clause-bearing.
    def profile(s, idx):
        v = np.asarray(s)[idx]
        return {"n": int(len(v)), "mean": float(v.mean()),
                "sd": float(v.std(ddof=1)) if len(v) > 1 else None,
                "median": float(np.median(v))}

    d_en = store[("mhclip_en", "real", PRIMARY_LAYER)]
    d_hcs = store[("hateclipseg", "real", PRIMARY_LAYER)]
    i_hcs_clean = np.where(~y_union)[0]
    res["stratum_profiles"] = {
        "mhclip_en": {
            r: {"no_protected_target_positive": profile(sc, i_no_pt),
                "protected_target_positive": profile(sc, i_pt),
                "shipped_normal": profile(sc, i_norm)}
            for r, sc in (("c1", d_en["c1"]),
                          ("c2", d_en["c2_raw"] * d_en["c2_sign_by_z"] * c2_dir))},
        "hateclipseg": {
            r: {"flip_union_pos_strict_neg": profile(sc, i_flip),
                "strict_positive": profile(sc, i_strictpos),
                "clean_normal": profile(sc, i_hcs_clean)}
            for r, sc in (("c1", d_hcs["c1"]),
                          ("c2", d_hcs["c2_raw"] * d_hcs["c2_sign_by_z"] * c2_dir))},
    }

    sweep = {}
    for slug in corpora:
        sweep[slug] = {}
        for readout in ("c1", "c2"):
            sweep[slug][readout] = {
                str(layer): {
                    "real": score_auc(slug, "real", layer, readout),
                    "shuffled": score_auc(slug, "shuffled", layer, readout),
                    "spam": score_auc(slug, "spam", layer, readout),
                } for layer in layers_all}
    res["layer_sweep_flip_auc"] = sweep

    # ---- post-hoc diagnostic, added after the clauses were computed -------
    # The shuffled placebo replaces h_i^union by h_pi(i)^union and leaves
    # h_i^strict alone, so c1 under the placebo still contains the per-video
    # term -<h_i^strict, d>. If that single term already separates the strata,
    # then c1 is a re-reading of the strict arm rather than an interaction
    # between the two specs. This split is diagnostic, not clause-bearing, and
    # was written after the verdict was fixed.
    posthoc = {}
    for slug, cp in corpora.items():
        d_unit = store[(slug, "real", PRIMARY_LAYER)]["d"]
        s_strict = -(cp["arms"]["strict"][:, PRIMARY_LAYER] @ d_unit)
        s_union = cp["arms"]["union"][:, PRIMARY_LAYER] @ d_unit
        z_only = cp["z"]["strict"]
        if slug == "mhclip_en":
            f = lambda s: auc_pos_neg(np.asarray(s)[i_no_pt], np.asarray(s)[i_pt])
        else:
            f = lambda s: auc_pos_neg(np.asarray(s)[i_flip], np.asarray(s)[i_strictpos])
        posthoc[slug] = {
            "auc_strict_term_only": f(s_strict),
            "auc_union_term_only": f(s_union),
            "auc_c1_full": f(store[(slug, "real", PRIMARY_LAYER)]["c1"]),
            "auc_z_strict_only": f(z_only),
            "spearman_c1_vs_strict_term":
                float(spearmanr(store[(slug, "real", PRIMARY_LAYER)]["c1"],
                                s_strict)[0]),
        }
    res["post_hoc_diagnostics"] = {
        "note": ("computed after the clause verdicts were fixed; descriptive "
                 "only, no clause depends on it"),
        "carrier_projection_split": posthoc,
    }

    # ---- descriptive: cosine with the supervised probe direction ---------
    Hp = en["arms"]["strict"][:, PRIMARY_LAYER]
    Xp = np.concatenate([Hp[i_no_pt], Hp[i_pt]], axis=0)
    yp = np.array([1] * len(i_no_pt) + [0] * len(i_pt))
    mu, sd = Xp.mean(axis=0), Xp.std(axis=0)
    sd = np.where(sd <= 0, 1.0, sd)
    w, _, ok = logistic_fit((Xp - mu) / sd, yp, PROBE_LAMBDA)
    w_raw = w / sd
    w_raw = w_raw / np.linalg.norm(w_raw)
    den = store[("mhclip_en", "real", PRIMARY_LAYER)]
    res["probe_direction"] = {
        "fit_converged": ok,
        "note": ("measurement instrument only; refit on the same 49 videos as "
                 "the readout-bottleneck test, mapped back to raw space"),
        "cos_pc1_residual_vs_probe":
            float(abs(np.dot(den["c2_loading"], w_raw))),
        "cos_carrier_vs_probe": float(abs(np.dot(den["d"], w_raw))),
    }

    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"verdict": res["verdict"],
                      "c1": {k: clauses["c1"][k].get("pass")
                             for k in clauses["c1"] if k != "verdict"},
                      "c2": {k: clauses["c2"][k].get("pass")
                             for k in clauses["c2"] if k != "verdict"}}, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
