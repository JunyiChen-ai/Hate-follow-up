"""Stage A of the answer-contrast pilot: the band-conditioned PCA control.

Frozen before the run, and frozen before any Stage-B GPU work.

Motivation. The readout-bottleneck kill test found that unsupervised principal
components of the judge's late-layer hidden states are the scalar readout z
wearing a different name: corpus-wide PC1 correlates with z at Spearman 0.992
and reproduces z's AUC to two decimals. That failure has a cheap potential
rescue that the earlier test never ran. Inside the saturation band, z has almost
no variance by construction, so a PCA fitted on band members only cannot
rediscover z. If band-conditioning alone restores unsupervised access to the
construct, the manufactured answer-side contrast of Stage B is unnecessary and
this pilot stops here with a reportable positive control.

Frozen protocol.
  Band: z >= +13, the frozen saturation band of the operating-point program.
  Corpora: HateClipSeg (expect 183 in band), HateMM test (81), ImpliHateVid
    test (70). All three read from disk; no model call.
  Layers: 18, 27, 36 of the 37 stored rows (row 0 is the embedding output).
  Standardisation: per dimension, using the mean and standard deviation of the
    band members only, no label read.
  Axes: top 3 principal components by SVD of the standardized, re-centred band
    matrix.
  Two fits, both label-free:
    self  - fitted on HateClipSeg's own band (183 rows). PRIMARY.
    pooled- fitted on the three per-corpus-standardized bands concatenated
            (334 rows), projected onto the HateClipSeg rows. SECONDARY,
            descriptive only.
  Arena: HateClipSeg band, strict-hate (expect 126) versus non-strict (57).
  Sign: a principal component has no intrinsic sign and the in-band z ordering
    is too degenerate to orient it, so each axis is scored sign-free as
    max(AUC, 1 - AUC). This is the generous reading, and it is generous in the
    direction that makes the control easier to pass, which is the conservative
    direction for the decision below.

Frozen decision rule.
  Let A = the best sign-free AUC over the 9 PRIMARY candidates
  (3 layers x 3 components, self-fit).
    A >= 0.70 -> CONTROL-SUCCEEDED. The free control does the job; Stage B is
                 not run and the pilot reports band-conditioning as the finding.
    A <  0.70 -> proceed to Stage B.

Descriptive, decision-irrelevant: an L2 logistic probe (lambda = 1.0, fixed)
with leave-one-out cross-validation on the same in-band arena, reporting the
supervised ceiling that any label-free in-band axis would have to approach.

Output: results/answer_contrast/stage_a.json. No video id reaches the output.

Usage:
  python scripts/duplex/answer_contrast_stagea.py
"""

import json
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

from readout_bottleneck_killtest import (  # noqa: E402
    auc, auc_pos_neg, load_hidden, logistic_fit, loo_probe, standardize,
)
from selftrained_readout_killtest import load_corpus  # noqa: E402

BAND_POS = 13.0
LAYERS = [18, 27, 36]
N_PC = 3
PROBE_LAMBDA = 1.0
CONTROL_FLOOR = 0.70

EXPECTED_BAND = {"HateClipSeg": 183, "HateMM": 81, "ImpliHateVid": 70}
EXPECTED_ARENA = (126, 57)

OUT = os.path.join(ROOT, "results", "answer_contrast", "stage_a.json")


def pca_fit(X):
    """Return (mean, components) of the standardized matrix X."""
    mu = X.mean(axis=0)
    Xc = X - mu
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = (S ** 2) / max(Xc.shape[0] - 1, 1)
    evr = (var / var.sum()).tolist()
    return mu, Vt[:N_PC], evr[:N_PC]


def main():
    t0 = time.time()
    res = {
        "title": "Answer-contrast pilot, Stage A: band-conditioned PCA control",
        "preregistration": "docs/duplex/PREREG_answer_contrast_pilot.md",
        "status": "frozen control, CPU only, no model call",
        "frozen_constants": {
            "band": f"z >= {BAND_POS}",
            "layers": LAYERS,
            "n_components": N_PC,
            "standardisation": ("per dimension over band members only, that "
                                "corpus's own hidden states, no label read"),
            "sign_rule": "sign-free scoring, max(AUC, 1 - AUC)",
            "primary_candidates": "3 layers x 3 components, HateClipSeg self-fit",
            "control_floor": CONTROL_FLOOR,
            "probe_l2_lambda": PROBE_LAMBDA,
        },
        "decision_rule": (
            f"best sign-free AUC over the 9 primary candidates >= "
            f"{CONTROL_FLOOR} -> CONTROL-SUCCEEDED, stop before Stage B; "
            f"otherwise proceed to Stage B"),
    }

    # ---- load the three corpora and cut their bands ----------------------
    bands = {}
    for name in ["HateClipSeg", "HateMM", "ImpliHateVid"]:
        ids, zs, y, hidden_dir = load_corpus(name)
        in_band = np.flatnonzero(zs >= BAND_POS)
        if in_band.size != EXPECTED_BAND[name]:
            raise SystemExit(f"ABORT: {name} band holds {in_band.size} videos, "
                             f"expected {EXPECTED_BAND[name]}")
        band_ids = [ids[i] for i in in_band]
        H = load_hidden(hidden_dir, band_ids)
        bands[name] = {"H": H, "z": zs[in_band], "y": y[in_band],
                       "n": in_band.size}
        print(f"[{name}] band n={in_band.size} "
              f"z range {zs[in_band].min():.3f}..{zs[in_band].max():.3f}",
              flush=True)

    hcs = bands["HateClipSeg"]
    y_hcs = hcs["y"]
    n_strict, n_non = int(y_hcs.sum()), int((1 - y_hcs).sum())
    if (n_strict, n_non) != EXPECTED_ARENA:
        raise SystemExit(f"ABORT: HateClipSeg in-band arena is {n_strict}/"
                         f"{n_non}, expected {EXPECTED_ARENA[0]}/"
                         f"{EXPECTED_ARENA[1]}")

    z_band = hcs["z"]
    res["band_facts"] = {
        name: {"n_in_band": bands[name]["n"],
               "z_min": float(bands[name]["z"].min()),
               "z_max": float(bands[name]["z"].max()),
               "z_sd": float(bands[name]["z"].std()),
               "z_iqr": float(np.percentile(bands[name]["z"], 75)
                              - np.percentile(bands[name]["z"], 25))}
        for name in bands}
    res["arena"] = {
        "corpus": "HateClipSeg, in-band only",
        "n_strict_hate": n_strict,
        "n_non_strict": n_non,
        "z_auc_in_band": auc(z_band, y_hcs),
        "z_auc_note": ("the incumbent scalar's ordering power inside the band; "
                       "the band was cut so that this is near chance"),
    }
    print(f"[arena] strict {n_strict} vs non-strict {n_non}, "
          f"in-band z AUC {res['arena']['z_auc_in_band']:.4f}", flush=True)

    # ---- primary: HateClipSeg self-fit ------------------------------------
    primary, secondary = [], []
    for layer in LAYERS:
        Xs, dead = standardize(hcs["H"][:, layer, :].astype(np.float32))
        Xs = Xs.astype(np.float64)
        mu, comps, evr = pca_fit(Xs)
        proj = (Xs - mu) @ comps.T
        for j in range(N_PC):
            a = proj[:, j]
            raw = auc(a, y_hcs)
            rho = float(spearmanr(a, z_band)[0])
            primary.append({
                "fit": "self", "layer": layer, "pc": j + 1,
                "explained_variance_ratio": float(evr[j]),
                "spearman_with_in_band_z": rho if np.isfinite(rho) else 0.0,
                "auc_raw": raw,
                "auc_signfree": max(raw, 1.0 - raw),
                "n_zero_variance_dims": dead,
            })

    # ---- secondary: pooled fit --------------------------------------------
    for layer in LAYERS:
        parts = []
        for name in ["ImpliHateVid", "HateMM", "HateClipSeg"]:
            Xn, _ = standardize(bands[name]["H"][:, layer, :].astype(np.float32))
            parts.append(Xn.astype(np.float64))
        pooled = np.concatenate(parts, axis=0)
        mu, comps, evr = pca_fit(pooled)
        Xh = parts[-1]
        proj = (Xh - mu) @ comps.T
        for j in range(N_PC):
            a = proj[:, j]
            raw = auc(a, y_hcs)
            rho = float(spearmanr(a, z_band)[0])
            secondary.append({
                "fit": "pooled", "layer": layer, "pc": j + 1,
                "n_pooled_rows": int(pooled.shape[0]),
                "explained_variance_ratio": float(evr[j]),
                "spearman_with_in_band_z": rho if np.isfinite(rho) else 0.0,
                "auc_raw": raw,
                "auc_signfree": max(raw, 1.0 - raw),
            })

    res["primary_axes"] = primary
    res["secondary_axes_descriptive"] = secondary

    best_primary = max(primary, key=lambda r: r["auc_signfree"])
    best_secondary = max(secondary, key=lambda r: r["auc_signfree"])
    res["best_primary"] = best_primary
    res["best_secondary_descriptive"] = best_secondary

    for r in primary + secondary:
        print(f"  {r['fit']:6s} L{r['layer']:2d} PC{r['pc']} "
              f"evr={r['explained_variance_ratio']:.3f} "
              f"rho_z={r['spearman_with_in_band_z']:+.3f} "
              f"auc={r['auc_raw']:.4f} signfree={r['auc_signfree']:.4f}",
              flush=True)

    # ---- descriptive supervised ceiling in the same arena -----------------
    print("[probe] leave-one-out supervised ceiling, layer 27 in-band",
          flush=True)
    Xs27, _ = standardize(hcs["H"][:, 27, :].astype(np.float32))
    Xs27 = Xs27.astype(np.float64)
    held, ok, nits = loo_probe(Xs27, y_hcs, PROBE_LAMBDA)
    probe_auc = auc(held, y_hcs)
    w, b, _, _ = logistic_fit(Xs27, y_hcs, PROBE_LAMBDA)
    res["descriptive_supervised_ceiling"] = {
        "note": ("decision-irrelevant; states whether the in-band arena is "
                 "linearly separable at all at layer 27"),
        "layer": 27,
        "loo_auc": probe_auc,
        "in_sample_auc": auc((Xs27 @ w + b).tolist(), y_hcs),
        "n_folds": int(len(y_hcs)),
        "all_folds_converged": bool(ok),
        "median_lbfgs_iterations": float(np.median(nits)),
    }
    print(f"[probe] LOO AUC {probe_auc:.4f}", flush=True)

    a_best = best_primary["auc_signfree"]
    verdict = "CONTROL-SUCCEEDED" if a_best >= CONTROL_FLOOR else "PROCEED"
    res["stage_a_verdict"] = {
        "best_primary_signfree_auc": a_best,
        "floor": CONTROL_FLOOR,
        "verdict": verdict,
        "meaning": ("band-conditioned PCA recovers the construct without "
                    "labels; the manufactured answer-side contrast is "
                    "unnecessary and Stage B is not run"
                    if verdict == "CONTROL-SUCCEEDED" else
                    "band-conditioned PCA does not recover the construct; "
                    "Stage B proceeds"),
    }
    res["wall_clock_seconds"] = round(time.time() - t0, 1)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    print(json.dumps(res["stage_a_verdict"], indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
