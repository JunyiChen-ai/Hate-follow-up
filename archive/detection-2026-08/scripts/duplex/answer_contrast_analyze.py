"""Stage B of the answer-contrast pilot: aborts, controls and the five clauses.

Pre-registration: docs/duplex/PREREG_answer_contrast_pilot.md.

CPU only. Reads the two-arm hidden states written by
src/duplex/extract_answer_contrast.py and the frozen judge's scores.jsonl
files; no model call happens here.

Writes docs/duplex/reports/answer_contrast_pilot.json. No video id reaches the
output.

Usage:
  python scripts/duplex/answer_contrast_analyze.py
"""

import glob
import json
import math
import os
import sys
import time

import numpy as np
from scipy.stats import spearmanr

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(ROOT, "src", "our_method"))
os.environ.setdefault("HVD_DATA_ROOT", "/home/jehc223/data")

from anchored_operating_point import macro_f1, valley_report  # noqa: E402
from crossbench_analyze import load_z  # noqa: E402
from readout_bottleneck_killtest import auc, auc_pos_neg, en_strata  # noqa: E402
from selftrained_readout_killtest import load_corpus  # noqa: E402

# ---------------------------------------------------------------- constants --
SEED = 20260808
LAYER_PRIMARY = 27
LAYERS_SECONDARY = [18, 36]
N_ROWS, DIM = 37, 4096
BAND_POS = 13.0

A1_COS_FLOOR = 0.99            # degeneracy abort
A2_RHO_FLOOR = 0.90            # z-renaming abort
C1_FLOOR = 0.70
C2_FLOOR = 0.72
C3_FLOOR = 0.65
C4_MARGIN = 0.05
C5_FLOOR = 0.852
NORM_CONTROL_SLACK = 0.03      # DEAD if norm AUC >= score AUC - slack

PREFLIGHT_RHO = 0.999
PREFLIGHT_MEDIAN_ABS = 0.05

# ANSWER_CONTRAST_ROOT exists so that the pipeline can be dry-run on synthetic
# arm arrays before the real extraction finishes. Every reported result comes
# from the default path.
AC = os.environ.get("ANSWER_CONTRAST_ROOT",
                    os.path.join(ROOT, "results", "answer_contrast"))
OUT_OVERRIDE = os.environ.get("ANSWER_CONTRAST_OUT")
# The preregistered fallback (fit on HateMM train only) is selected by setting
# ANSWER_CONTRAST_FIT=hatemm, never by editing this list after a result.
# Preregistered fit set: ImpliHateVid train plus HateMM train. HateMM train is
# dropped because the frozen judge's input for that split does not exist -- the
# fresh Whisper transcripts the judge reads were only ever produced for the
# test splits and for the ImpliHateVid full corpus, so HateMM train cannot be
# extracted byte-identically. Recorded at pre-flight, before any clause was
# computed; the fit set is narrowed by data availability, never by a result.
FIT_SETS = [("ImpliHateVid", "train")]
EVAL_SETS = [("HateMM", "test"), ("HateClipSeg", "test"),
             ("MHClip_EN", "test"), ("ImpliHateVid", "test")]

FROZEN_SCORES = {
    "ImpliHateVid_train": os.path.join(ROOT, "results", "c2_fullcorpus",
                                       "judge_8b", "scores.jsonl"),
    "HateMM_test": os.path.join(ROOT, "results", "testruns", "hatemm",
                                "judge_8b", "scores.jsonl"),
    "HateClipSeg_test": os.path.join(ROOT, "results", "hateclipseg",
                                     "judge_8b", "scores.jsonl"),
    "MHClip_EN_test": os.path.join(ROOT, "results", "testruns", "mhclip_en",
                                   "judge_8b", "scores.jsonl"),
    "ImpliHateVid_test": os.path.join(ROOT, "results", "testruns",
                                      "implihatevid", "judge_8b",
                                      "scores.jsonl"),
}
EXPECTED_N = {"ImpliHateVid_train": 1283,
              "HateMM_test": 215, "HateClipSeg_test": 394,
              "MHClip_EN_test": 161, "ImpliHateVid_test": 400}

OUT = OUT_OVERRIDE or os.path.join(ROOT, "docs", "duplex", "reports",
                                   "answer_contrast_pilot.json")


def tag(dataset, split):
    return f"{dataset}_{split}"


# --------------------------------------------------------------------- io ---
def load_run(dataset, split):
    """Ids in a stable order, the recomputed z, and the arms directory."""
    d = os.path.join(AC, tag(dataset, split))
    z = load_z(os.path.join(d, "scores.jsonl"))
    arms_dir = os.path.join(d, "arms")
    have = {os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(arms_dir, "*.npy"))}
    ids = sorted(set(z) & have)
    return ids, np.array([z[v] for v in ids], dtype=float), arms_dir


def load_arm_layer(arms_dir, ids, layer):
    """(n, 2, DIM) float32 at one layer, in the given id order."""
    out = np.empty((len(ids), 2, DIM), dtype=np.float32)
    for k, v in enumerate(ids):
        a = np.load(os.path.join(arms_dir, v + ".npy"), mmap_mode="r")
        if a.shape != (2, N_ROWS, DIM):
            raise SystemExit(f"ABORT: arm array for one video has shape "
                             f"{a.shape}, expected {(2, N_ROWS, DIM)}")
        out[k] = np.asarray(a[:, layer, :], dtype=np.float32)
    return out


def derangement(n, rng):
    """A permutation with no fixed point."""
    for _ in range(1000):
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p
    p = np.roll(np.arange(n), 1)
    return p


# ------------------------------------------------------------------ readout --
def fit_axis(fit_deltas, fit_z, permute=None, rng=None):
    """Return (mean-free scaler, PC1, sign) from the pooled fit residuals.

    fit_deltas: list of (n_c, 2, DIM) arrays, one per fit corpus.
    permute: None, or a list of derangements applied to each corpus's No arm.
    """
    residuals = []
    for c, A in enumerate(fit_deltas):
        no = A[:, 1, :]
        if permute is not None:
            no = no[permute[c]]
        D = A[:, 0, :] - no
        residuals.append(D - D.mean(axis=0))
    R = np.concatenate(residuals, axis=0)
    sd = R.std(axis=0)
    sd = np.where(sd <= 0, 1.0, sd)
    Rs = R / sd
    Rs = Rs - Rs.mean(axis=0)
    _, S, Vt = np.linalg.svd(Rs, full_matrices=False)
    pc1 = Vt[0]
    evr = float((S[0] ** 2) / (S ** 2).sum())
    proj = Rs @ pc1
    rho = float(spearmanr(proj, fit_z)[0])
    sign = -1.0 if (np.isfinite(rho) and rho < 0) else 1.0
    return {"sd": sd, "pc1": pc1 * sign, "evr": evr,
            "fit_spearman_with_z": rho, "sign": sign,
            "mean_delta": R.mean(axis=0)}


def score_corpus(A, axis, permute=None):
    """Score one corpus: its own mean centring, the fit set's scale, PC1."""
    no = A[:, 1, :]
    if permute is not None:
        no = no[permute]
    D = A[:, 0, :] - no
    r = (D - D.mean(axis=0)) / axis["sd"]
    return r @ axis["pc1"], np.linalg.norm(r, axis=1), D


def cos_to_mean(D):
    m = D.mean(axis=0)
    nm = np.linalg.norm(m)
    if nm <= 0:
        return np.zeros(D.shape[0])
    return (D @ m) / (np.linalg.norm(D, axis=1) * nm + 1e-12)


# ------------------------------------------------------------------ arenas ---
def build_arenas():
    """Every labelled stratum, built from the frozen label loaders."""
    ar = {}

    ids_h, z_h, y_h, _ = load_corpus("HateClipSeg")
    band = np.flatnonzero(z_h >= BAND_POS)
    ar["C1"] = {"corpus": "HateClipSeg_test",
                "ids": [ids_h[i] for i in band],
                "y": y_h[band],
                "n_pos": int(y_h[band].sum()),
                "n_neg": int((1 - y_h[band]).sum())}
    if (ar["C1"]["n_pos"], ar["C1"]["n_neg"]) != (126, 57):
        raise SystemExit(f"ABORT: C1 arena is {ar['C1']['n_pos']}/"
                         f"{ar['C1']['n_neg']}, expected 126/57")

    ids_e, z_e, _, _ = load_corpus("MHClip_EN")
    no_pt, pt, _ = en_strata(ids_e)
    ar["C2"] = {"corpus": "MHClip_EN_test", "no_target": no_pt,
                "protected": pt}

    ids_m, z_m, y_m, _ = load_corpus("HateMM")
    v = valley_report(z_m, y_m)
    pred = z_m >= v["threshold"]
    gold = y_m.astype(bool)
    fp = [ids_m[i] for i in np.flatnonzero(~gold & pred)]
    tp = [ids_m[i] for i in np.flatnonzero(gold & pred)]
    if (len(fp), len(tp)) != (70, 83):
        raise SystemExit(f"ABORT: C3 arena is {len(fp)} FP / {len(tp)} TP, "
                         "expected 70/83")
    ar["C3"] = {"corpus": "HateMM_test", "fp": fp, "tp": tp,
                "valley_threshold": float(v["threshold"]),
                "z_by_id": {ids_m[i]: float(z_m[i]) for i in range(len(ids_m))}}

    ids_i, z_i, y_i, _ = load_corpus("ImpliHateVid")
    ar["C5"] = {"corpus": "ImpliHateVid_test", "ids": ids_i, "y": y_i,
                "z": z_i,
                "valley_macro_f1": valley_report(z_i, y_i)["macro_f1"]}
    return ar


def zmatched_pairs(fp, tp, z_by_id):
    """Greedy 1:1 nearest-z matching of true positives to false positives."""
    pool = sorted(tp, key=lambda v: z_by_id[v])
    used, pairs = set(), []
    for f in sorted(fp, key=lambda v: z_by_id[v]):
        best, bestd = None, None
        for t in pool:
            if t in used:
                continue
            d = abs(z_by_id[t] - z_by_id[f])
            if bestd is None or d < bestd:
                best, bestd = t, d
        if best is None:
            break
        used.add(best)
        pairs.append((f, best))
    return [p[0] for p in pairs], [p[1] for p in pairs]


# -------------------------------------------------------------------- main ---
def main():
    t0 = time.time()
    res = {
        "title": "Answer-contrast readout pilot, Stage B",
        "preregistration": "docs/duplex/PREREG_answer_contrast_pilot.md",
        "status": "preregistered kill test, run once; analysis is CPU only",
        "frozen_constants": {
            "layer_primary": LAYER_PRIMARY,
            "layers_secondary": LAYERS_SECONDARY,
            "n_components": 1,
            "seed": SEED,
            "abort_a1_cos_floor": A1_COS_FLOOR,
            "abort_a2_rho_floor": A2_RHO_FLOOR,
            "clause_bars": {"C1": C1_FLOOR, "C2": C2_FLOOR, "C3": C3_FLOOR,
                            "C4_margin": C4_MARGIN, "C5": C5_FLOOR},
            "norm_control_slack": NORM_CONTROL_SLACK,
        },
    }

    # ---- load every run, pre-flight the prompt identity -------------------
    runs, preflight = {}, {}
    for ds, sp in FIT_SETS + EVAL_SETS:
        t = tag(ds, sp)
        ids, z, arms_dir = load_run(ds, sp)
        if len(ids) < EXPECTED_N[t]:
            raise SystemExit(f"ABORT: {t} has {len(ids)} complete videos, "
                             f"expected at least {EXPECTED_N[t]}")
        if len(ids) != EXPECTED_N[t]:
            res.setdefault("coverage_surplus", {})[t] = {
                "n_extracted": len(ids), "n_expected": EXPECTED_N[t]}
        runs[t] = {"ids": ids, "z": z, "arms_dir": arms_dir}
        if t in FROZEN_SCORES:
            frozen = load_z(FROZEN_SCORES[t])
            common = [v for v in ids if v in frozen]
            zf = np.array([frozen[v] for v in common])
            zn = np.array([runs[t]["z"][ids.index(v)] for v in common])
            rho = float(spearmanr(zn, zf)[0])
            med = float(np.median(np.abs(zn - zf)))
            preflight[t] = {"n_common": len(common), "spearman": rho,
                            "median_abs_diff": med,
                            "max_abs_diff": float(np.max(np.abs(zn - zf))),
                            "passes": bool(rho >= PREFLIGHT_RHO
                                           and med <= PREFLIGHT_MEDIAN_ABS)}
            if not preflight[t]["passes"]:
                raise SystemExit(f"ABORT: {t} pre-flight failed: "
                                 f"spearman {rho}, median abs diff {med}")
        print(f"[load] {t}: {len(ids)} videos", flush=True)
    res["preflight_z_reproduction"] = preflight

    arenas = build_arenas()
    res["arenas"] = {
        "C1": {"corpus": "HateClipSeg in-band", "n_strict": arenas["C1"]["n_pos"],
               "n_non_strict": arenas["C1"]["n_neg"]},
        "C2": {"corpus": "MHClip_EN", "n_no_target": len(arenas["C2"]["no_target"]),
               "n_protected": len(arenas["C2"]["protected"])},
        "C3": {"corpus": "HateMM", "n_valley_fp": len(arenas["C3"]["fp"]),
               "n_valley_tp": len(arenas["C3"]["tp"]),
               "valley_threshold": arenas["C3"]["valley_threshold"]},
        "C5": {"corpus": "ImpliHateVid", "n": len(arenas["C5"]["ids"]),
               "incumbent_valley_macro_f1": arenas["C5"]["valley_macro_f1"]},
    }

    rng = np.random.default_rng(SEED)
    derangements = {t: derangement(len(runs[t]["ids"]), rng)
                    for t in sorted(runs)}

    def run_layer(layer, permuted):
        perm_fit = ([derangements[tag(ds, sp)] for ds, sp in FIT_SETS]
                    if permuted else None)
        fit_arrays, fit_z = [], []
        for ds, sp in FIT_SETS:
            t = tag(ds, sp)
            fit_arrays.append(load_arm_layer(runs[t]["arms_dir"],
                                             runs[t]["ids"], layer))
            fit_z.append(runs[t]["z"])
        # the fit-set z used for the sign is the pooled z of the fit corpora
        axis = fit_axis(fit_arrays, np.concatenate(fit_z), permute=perm_fit)

        # degeneracy on the fit set, unpermuted arms only
        cosines = []
        for c, A in enumerate(fit_arrays):
            D = A[:, 0, :] - A[:, 1, :]
            cosines.append(cos_to_mean(D))
        axis["median_cos_to_mean"] = float(np.median(np.concatenate(cosines)))

        out = {"axis": {k: axis[k] for k in
                        ("evr", "fit_spearman_with_z", "sign",
                         "median_cos_to_mean")}}
        per_corpus = {}
        for ds, sp in EVAL_SETS:
            t = tag(ds, sp)
            A = load_arm_layer(runs[t]["arms_dir"], runs[t]["ids"], layer)
            p = derangements[t] if permuted else None
            s, rn, D = score_corpus(A, axis, permute=p)
            per_corpus[t] = {
                "ids": runs[t]["ids"], "score": s, "rnorm": rn,
                "z": runs[t]["z"],
                "spearman_score_z": float(spearmanr(s, runs[t]["z"])[0]),
                "median_cos_to_mean": float(np.median(cos_to_mean(D))),
                "mean_delta_norm": float(np.linalg.norm(D, axis=1).mean()),
            }
            del A, D
        out["per_corpus"] = per_corpus
        del fit_arrays
        return out

    def index_of(t, wanted):
        pos = {v: i for i, v in enumerate(runs[t]["ids"])}
        missing = [v for v in wanted if v not in pos]
        if missing:
            raise SystemExit(f"ABORT: {len(missing)} arena videos missing "
                             f"from {t}")
        return np.array([pos[v] for v in wanted], dtype=int)

    def clause_values(state):
        pc = state["per_corpus"]
        v = {}
        # C1
        t = "HateClipSeg_test"
        idx = index_of(t, arenas["C1"]["ids"])
        s = pc[t]["score"][idx]
        y = arenas["C1"]["y"]
        v["C1_auc"] = auc(s, y)
        v["C1_norm_auc"] = auc(pc[t]["rnorm"][idx], y)
        v["C1_z_auc"] = auc(pc[t]["z"][idx], y)
        # C2
        t = "MHClip_EN_test"
        i_no = index_of(t, arenas["C2"]["no_target"])
        i_pt = index_of(t, arenas["C2"]["protected"])
        a = auc_pos_neg(pc[t]["score"][i_no], pc[t]["score"][i_pt])
        v["C2_auc_directional"] = a
        v["C2_auc_signfree"] = max(a, 1.0 - a)
        # C3
        t = "HateMM_test"
        i_fp = index_of(t, arenas["C3"]["fp"])
        i_tp = index_of(t, arenas["C3"]["tp"])
        v["C3_auc"] = auc_pos_neg(pc[t]["score"][i_tp], pc[t]["score"][i_fp])
        v["C3_z_auc"] = auc_pos_neg(pc[t]["z"][i_tp], pc[t]["z"][i_fp])
        mfp, mtp = zmatched_pairs(arenas["C3"]["fp"], arenas["C3"]["tp"],
                                  arenas["C3"]["z_by_id"])
        v["C3_zmatched_auc"] = auc_pos_neg(
            pc[t]["score"][index_of(t, mtp)], pc[t]["score"][index_of(t, mfp)])
        v["C3_zmatched_n_pairs"] = len(mfp)
        # C5
        t = "ImpliHateVid_test"
        idx = index_of(t, arenas["C5"]["ids"])
        s = pc[t]["score"][idx]
        y = arenas["C5"]["y"]
        rep = valley_report(s, y)
        # A score distribution with no KDE valley cannot supply an operating
        # point at all, which is a C5 failure rather than a missing number.
        v["C5_macro_f1"] = rep["macro_f1"]
        v["C5_threshold"] = rep["threshold"]
        v["C5_valley_found"] = rep["threshold"] is not None
        v["C5_auc"] = auc(s, y)
        v["max_abs_spearman_score_z"] = max(
            abs(pc[k]["spearman_score_z"]) for k in pc)
        v["spearman_score_z_per_corpus"] = {
            k: pc[k]["spearman_score_z"] for k in pc}
        v["median_cos_to_mean_per_corpus"] = {
            k: pc[k]["median_cos_to_mean"] for k in pc}
        return v

    # ---- primary layer, real and placebo ---------------------------------
    print(f"[layer {LAYER_PRIMARY}] real", flush=True)
    real = run_layer(LAYER_PRIMARY, permuted=False)
    real_v = clause_values(real)
    print(f"  C1 {real_v['C1_auc']:.4f} C2 {real_v['C2_auc_signfree']:.4f} "
          f"C3 {real_v['C3_auc']:.4f} C5 {real_v['C5_macro_f1']}", flush=True)

    print(f"[layer {LAYER_PRIMARY}] pairing placebo", flush=True)
    plac = run_layer(LAYER_PRIMARY, permuted=True)
    plac_v = clause_values(plac)
    print(f"  placebo C1 {plac_v['C1_auc']:.4f}", flush=True)

    # ---- aborts ----------------------------------------------------------
    a1 = real["axis"]["median_cos_to_mean"] >= A1_COS_FLOOR
    a2 = real_v["max_abs_spearman_score_z"] >= A2_RHO_FLOOR
    res["aborts"] = {
        "A1_degeneracy": {
            "rule": f"median cos(delta h, mean delta h) on the fit set >= "
                    f"{A1_COS_FLOOR} -> DEAD",
            "value": real["axis"]["median_cos_to_mean"],
            "fires": bool(a1)},
        "A2_z_renaming": {
            "rule": f"|Spearman(score, z)| >= {A2_RHO_FLOOR} on any eval "
                    f"corpus -> DEAD",
            "value": real_v["max_abs_spearman_score_z"],
            "per_corpus": real_v["spearman_score_z_per_corpus"],
            "fires": bool(a2)},
    }

    # ---- controls and clauses --------------------------------------------
    c1 = real_v["C1_auc"] >= C1_FLOOR
    # The norm control asks whether a passing score is really commitment
    # magnitude. It can only take a verdict away, never add one, so it is
    # evaluated but only becomes the cause of death when C1 has passed.
    # Clarified during the synthetic dry run, before any real arm existed.
    norm_fires = real_v["C1_norm_auc"] >= real_v["C1_auc"] - NORM_CONTROL_SLACK
    norm_dead = bool(norm_fires and c1)
    c2 = real_v["C2_auc_signfree"] >= C2_FLOOR
    c3 = real_v["C3_auc"] >= C3_FLOOR
    c4 = real_v["C1_auc"] >= plac_v["C1_auc"] + C4_MARGIN
    c5 = (real_v["C5_macro_f1"] is not None
          and real_v["C5_macro_f1"] >= C5_FLOOR)

    res["controls"] = {
        "pairing_placebo": {
            "rule": f"real C1 AUC >= placebo C1 AUC + {C4_MARGIN}",
            "real_c1_auc": real_v["C1_auc"],
            "placebo_c1_auc": plac_v["C1_auc"],
            "margin": real_v["C1_auc"] - plac_v["C1_auc"],
            "passes": bool(c4)},
        "norm_control": {
            "rule": f"DEAD if ||r|| AUC >= score AUC - {NORM_CONTROL_SLACK} "
                    "on the C1 arena",
            "score_c1_auc": real_v["C1_auc"],
            "norm_c1_auc": real_v["C1_norm_auc"],
            "condition_met": bool(norm_fires),
            "fires": bool(norm_dead),
            "note": ("the control is decisive only when C1 passes; with C1 "
                     "failed there is no passing score for it to explain "
                     "away")},
    }
    res["clauses"] = {
        "C1_primary_hateclipseg_band": {
            "bar": C1_FLOOR, "value": real_v["C1_auc"],
            "reference_in_band_z": real_v["C1_z_auc"],
            "reference_band_conditioned_pca": 0.6348,
            "result": "PASS" if c1 else "FAIL"},
        "C2_mhclip_en_construct": {
            "bar": C2_FLOOR, "value_signfree": real_v["C2_auc_signfree"],
            "value_directional": real_v["C2_auc_directional"],
            "reference_z_signfree": 0.679, "supervised_ceiling": 0.837,
            "result": "PASS" if c2 else "FAIL"},
        "C3_hatemm_midband": {
            "bar": C3_FLOOR, "value": real_v["C3_auc"],
            "reference_z_auc": real_v["C3_z_auc"],
            "descriptive_zmatched_auc": real_v["C3_zmatched_auc"],
            "descriptive_zmatched_pairs": real_v["C3_zmatched_n_pairs"],
            "result": "PASS" if c3 else "FAIL"},
        "C4_pairing_placebo": {
            "bar": C4_MARGIN,
            "value": real_v["C1_auc"] - plac_v["C1_auc"],
            "result": "PASS" if c4 else "FAIL"},
        "C5_implihatevid_regression_guard": {
            "bar": C5_FLOOR, "value": real_v["C5_macro_f1"],
            "valley_found": real_v["C5_valley_found"],
            "incumbent_valley_macro_f1": arenas["C5"]["valley_macro_f1"],
            "score_auc": real_v["C5_auc"],
            "result": "PASS" if c5 else "FAIL"},
    }

    verdict = "DEAD"
    if a1:
        reason = "abort A1 fired: the manufactured contrast is degenerate"
    elif a2:
        reason = "abort A2 fired: the readout is z under a new name"
    elif norm_dead:
        reason = "norm control fired: the readout is commitment magnitude"
    elif c1 and c2 and c3 and c4 and c5:
        verdict, reason = "SURVIVES", "no abort fired and all five clauses hold"
    else:
        failed = [n for n, ok in [("C1", c1), ("C2", c2), ("C3", c3),
                                  ("C4", c4), ("C5", c5)] if not ok]
        reason = "clauses " + ", ".join(failed) + " failed"
    res["verdict"] = verdict
    res["verdict_reason"] = reason

    res["axis_facts"] = {
        "primary_layer": real["axis"],
        "explained_variance_ratio_pc1": real["axis"]["evr"],
        "median_cos_to_mean_per_eval_corpus":
            real_v["median_cos_to_mean_per_corpus"],
    }

    # ---- secondary layers, descriptive ------------------------------------
    secondary = {}
    for layer in LAYERS_SECONDARY:
        print(f"[layer {layer}] secondary", flush=True)
        st = run_layer(layer, permuted=False)
        sv = clause_values(st)
        secondary[str(layer)] = {
            "C1_auc": sv["C1_auc"], "C2_auc_signfree": sv["C2_auc_signfree"],
            "C3_auc": sv["C3_auc"], "C5_macro_f1": sv["C5_macro_f1"],
            "max_abs_spearman_score_z": sv["max_abs_spearman_score_z"],
            "pc1_explained_variance_ratio": st["axis"]["evr"],
            "median_cos_to_mean": st["axis"]["median_cos_to_mean"],
        }
        print(f"  C1 {sv['C1_auc']:.4f} C2 {sv['C2_auc_signfree']:.4f} "
              f"C3 {sv['C3_auc']:.4f} C5 {sv['C5_macro_f1']}", flush=True)
    res["secondary_layers_descriptive"] = secondary

    res["wall_clock_seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    print(json.dumps({"verdict": verdict, "reason": reason,
                      "clauses": res["clauses"], "aborts": res["aborts"],
                      "controls": res["controls"]}, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
