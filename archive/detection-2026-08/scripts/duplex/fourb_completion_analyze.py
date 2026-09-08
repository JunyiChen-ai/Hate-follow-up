"""4B completion sweep: one fixed 4B row against one fixed 8B row, five corpora.

Pre-registration: docs/duplex/PREREG_4b_completion.md (frozen and committed at
ba00da1, before any new score was written). Every convention below is quoted
from it or inherited from PREREG_scale_emergence.md, and none is re-chosen.

This is a fixed model swap evaluated everywhere. Both arms are scored on all
five corpora with the same code, and nothing anywhere selects a model per
corpus.

CPU only. No model call. Reads the per-video raw z and the per-video stored
hidden states written by the frozen judge, plus the unembedding rows and the
final RMSNorm weight of each checkpoint from the local Hugging Face cache.

Readouts per corpus per arm:
  - de-quantized z, recomputed in fp32 from the stored final state;
  - ranking AUC of that z against gold;
  - de-quantized KDE relative trough depth and mode count (label-free);
  - macro-F1 at that arm's own valley (label-free, transductive);
  - macro-F1 at the macro-F1-maximizing labeled threshold (diagnostic ceiling),
    with the hateful-F1-max convention of the committed reports alongside.

HateClipSeg is reported under both shipped label collapses. The label-free
threshold is identical across the two by construction.

Writes results/scale_emergence/fourb_completion.json. No video id reaches the
output.

Usage:
  python scripts/duplex/fourb_completion_analyze.py
"""

import json
import math
import os
import sys

import numpy as np
import torch

ROOT = "/home/jehc223/Hate-follow-up"
sys.path.insert(0, os.path.join(ROOT, "scripts", "duplex"))
sys.path.insert(0, os.path.join(ROOT, "src", "our_method"))
os.environ.setdefault("HVD_DATA_ROOT", "/home/jehc223/data")

# Every numeric convention is imported, not re-implemented: the KDE recipe, the
# AUC routine, the macro-F1 definition, the oracle search, the head loader and
# the final-RMSNorm rule are the ones the scale-emergence run used.
from scale_emergence_analyze import (  # noqa: E402
    auc, kde_valley, macro_f1, oracle_threshold, load_head, final_state,
)
from data_utils import load_annotations  # noqa: E402
from hateclipseg_prep import video_labels as hcs_video_labels  # noqa: E402

OUT = os.path.join(ROOT, "results", "scale_emergence", "fourb_completion.json")

ARMS = [
    ("4b", "Qwen/Qwen3-VL-4B-Instruct", "judge_4b", 36),
    ("8b", "Qwen/Qwen3-VL-8B-Instruct", "judge_8b", 36),
]

# slug, working directory, dataset, expected coverage
CORPORA = [
    ("implihatevid", "results/testruns/implihatevid", "ImpliHateVid", 400),
    ("hatemm",       "results/testruns/hatemm",       "HateMM",       215),
    ("mhclip_en",    "results/testruns/mhclip_en",    "MHClip_EN",    161),
    ("mhclip_zh",    "results/testruns/mhclip_zh",    "MHClip_ZH",    149),
    ("hateclipseg",  "results/hateclipseg",           "HateClipSeg",  394),
]

LABEL_MAP = {
    "ImpliHateVid": {"Hateful": 1, "Normal": 0},
    "HateMM": {"Hate": 1, "Non Hate": 0},
    "MHClip_EN": {"Hateful": 1, "Offensive": 1, "Normal": 0},
    "MHClip_ZH": {"Hateful": 1, "Offensive": 1, "Normal": 0},
}

# The committed 8B valley macro-F1, verified against the committed reports
# before the pre-registration was frozen. The stored-bf16 recomputation must
# reproduce each of these to within REPRO_TOL or the analysis aborts.
COMMITTED_8B_VALLEY = {
    "implihatevid|primary":       (0.8823, "docs/duplex/reports/test_c2_implihatevid_8b.json"),
    "hatemm|primary":             (0.6562, "docs/duplex/reports/test_c2_hatemm_8b.json"),
    "mhclip_en|primary":          (0.6962, "docs/duplex/reports/test_c2_mhclip_en_8b.json"),
    "mhclip_zh|primary":          (0.7256, "docs/duplex/reports/test_c2_mhclip_zh_8b.json"),
    "hateclipseg|offensive_union": (0.6522, "results/hateclipseg/gated_anchor_results.json"),
    "hateclipseg|hateful_strict":  (0.4479, "results/hateclipseg/gated_anchor_results.json"),
}
REPRO_TOL = 0.002

# Frozen aggregate references.
REF_8B_FOUR_CORPUS_VALLEY_MEAN = 0.7401   # ANCHORED_OPERATING_POINT_NOTE.md
REF_8B_FOUR_CORPUS_ORACLE_MEAN = 0.8175   # same
REF_TRIAGE_FOUR_CORPUS_MEAN = 0.808       # OPERATING_POINT_CLOSEOUT_NOTE.md


def oracle_hateful_f1max(z, y):
    """The committed reports' convention: maximise hateful-class F1."""
    z = np.asarray(z, float)
    y = np.asarray(y, int)
    best, best_t, best_macro = -1.0, None, None
    for t in np.unique(z):
        pred = (z >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        if f > best:
            best, best_t, best_macro = f, float(t), macro_f1(y, pred)
    return best_t, best, best_macro


def load_scores(judge_dir, n_layers, repo):
    """Per-video stored z, de-quantized z and the geometry, in a fixed id order."""
    spath = os.path.join(judge_dir, "scores.jsonl")
    hdir = os.path.join(judge_dir, "hidden")
    if not os.path.exists(spath):
        raise SystemExit(f"ABORT: no scores at {spath}")

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

    Wy, Wn, g, lm_key = load_head(repo)
    dvec = Wy.mean(0) - Wn.mean(0)

    zd, cosv = [], []
    for v in ids:
        p = os.path.join(hdir, v + ".npy")
        if not os.path.isfile(p):
            raise SystemExit(f"ABORT: {judge_dir}: a scored video has no hidden state")
        a = np.load(p)
        if a.shape != (n_layers + 1, g.shape[0]):
            raise SystemExit(f"ABORT: {judge_dir}: hidden shape {a.shape}")
        h = final_state(torch.from_numpy(a[-1]).float(), g)
        zd.append(float(torch.logsumexp(Wy @ h, 0) - torch.logsumexp(Wn @ h, 0)))
        cosv.append(float(torch.dot(h, dvec) / (h.norm() * dvec.norm())))

    return (ids, np.array([z_stored[v] for v in ids]), np.array(zd),
            np.array(cosv), float(dvec.norm()), lm_key)


def score_collapse(zd, zs, y, v_dq, v_bf):
    """Everything that depends on a label, for one collapse of one cell."""
    out = {
        "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
        "ranking_auc_dequantized": auc(zd, y),
        "ranking_auc_stored_bf16": auc(zs, y),
    }
    if v_dq.get("value") is None:
        out["valley_macro_f1_dequantized"] = None
        out["valley_errors"] = None
    else:
        pred = (zd >= v_dq["value"]).astype(int)
        out["valley_macro_f1_dequantized"] = macro_f1(y, pred)
        out["valley_errors"] = {"fn": int(((pred == 0) & (y == 1)).sum()),
                                "fp": int(((pred == 1) & (y == 0)).sum())}
    if v_bf.get("value") is None:
        out["valley_macro_f1_stored_bf16"] = None
    else:
        out["valley_macro_f1_stored_bf16"] = macro_f1(
            y, (zs >= v_bf["value"]).astype(int))

    o_thr, o_f1 = oracle_threshold(zd, y)
    h_thr, h_f1, h_macro = oracle_hateful_f1max(zd, y)
    out["oracle_macro_f1"] = o_f1
    out["oracle_threshold_macro_f1_max"] = o_thr
    out["oracle_hateful_f1_max"] = {"threshold": h_thr, "hateful_f1": h_f1,
                                    "macro_f1_at_that_threshold": h_macro}
    out["gap_valley_to_oracle"] = (
        None if out["valley_macro_f1_dequantized"] is None
        else float(o_f1 - out["valley_macro_f1_dequantized"]))
    return out


def main():
    cells = {}
    for slug, wd, dataset, expected in CORPORA:
        # labels, one vector per collapse
        if dataset == "HateClipSeg":
            labs = hcs_video_labels()
            collapses = {
                "offensive_union": lambda v: labs[v][0],
                "hateful_strict": lambda v: labs[v][1],
            }
        else:
            ann = load_annotations(dataset)
            lmap = LABEL_MAP[dataset]
            collapses = {"primary": lambda v, a=ann, m=lmap: m[a[v]["label"]]}

        for arm, repo, jdir, nlayers in ARMS:
            judge_dir = os.path.join(ROOT, wd, jdir)
            ids, zs, zd, cosv, dnorm, lm_key = load_scores(judge_dir, nlayers, repo)
            if len(ids) != expected:
                raise SystemExit(
                    f"ABORT: coverage {slug}/{arm}: {len(ids)} != {expected}")

            v_dq = kde_valley(zd)
            v_bf = kde_valley(zs)
            cell = {
                "arm": arm, "checkpoint": repo, "corpus": dataset,
                "n_videos": len(ids), "unembedding_key": lm_key,
                "z_distribution": {
                    "sd_dequantized": float(zd.std(ddof=1)),
                    "dynamic_range_dequantized": float(zd.max() - zd.min()),
                    "min": float(zd.min()), "max": float(zd.max()),
                    "max_abs_dequantization_shift": float(np.abs(zd - zs).max()),
                    "n_modes_dequantized": v_dq["n_grid_local_maxima"],
                    "n_modes_stored_bf16": v_bf["n_grid_local_maxima"],
                    "relative_trough_depth_dequantized":
                        v_dq["relative_trough_depth"] if v_dq.get("value") is not None else None,
                    "relative_trough_depth_stored_bf16":
                        v_bf["relative_trough_depth"] if v_bf.get("value") is not None else None,
                    "valley_dequantized": v_dq.get("value"),
                    "valley_stored_bf16": v_bf.get("value"),
                },
                "angular_commitment": {
                    "cos_sd": float(cosv.std(ddof=1)),
                    "cos_mean": float(cosv.mean()),
                    "answer_direction_norm": dnorm,
                },
                "collapses": {name: score_collapse(zd, zs, np.array(
                    [fn(v) for v in ids], dtype=int), v_dq, v_bf)
                    for name, fn in collapses.items()},
            }
            cells[f"{slug}|{arm}"] = cell
            for cname, c in cell["collapses"].items():
                print(f"{slug:13s} {arm} {cname:16s} n={cell['n_videos']:4d} "
                      f"auc={c['ranking_auc_dequantized']:.4f} "
                      f"trough={cell['z_distribution']['relative_trough_depth_dequantized']} "
                      f"valley={c['valley_macro_f1_dequantized']} "
                      f"oracle={c['oracle_macro_f1']:.4f}", flush=True)

    # ---- frozen reproduction check on the 8B arm --------------------------
    repro = []
    for key, (committed, src) in COMMITTED_8B_VALLEY.items():
        slug, cname = key.split("|")
        got = cells[f"{slug}|8b"]["collapses"][cname]["valley_macro_f1_stored_bf16"]
        delta = None if got is None else abs(got - committed)
        ok = delta is not None and delta <= REPRO_TOL
        repro.append({"cell": key, "committed": committed, "source": src,
                      "recomputed_stored_bf16": got, "abs_delta": delta,
                      "within_tolerance": bool(ok)})
        if not ok:
            raise SystemExit(
                f"ABORT: 8B reproduction failed on {key}: {got} vs {committed}")
    print("8B reproduction: all six cells within %.3f" % REPRO_TOL, flush=True)

    # ---- frozen aggregates -------------------------------------------------
    def valley(slug, arm, cname):
        return cells[f"{slug}|{arm}"]["collapses"][cname][
            "valley_macro_f1_dequantized"]

    def oracle(slug, arm, cname):
        return cells[f"{slug}|{arm}"]["collapses"][cname]["oracle_macro_f1"]

    FOUR = [("implihatevid", "primary"), ("hatemm", "primary"),
            ("mhclip_en", "primary"), ("mhclip_zh", "primary")]
    FIVE = FOUR + [("hateclipseg", "offensive_union")]

    def mean_over(pairs, arm, getter):
        vals = [getter(s, arm, c) for s, c in pairs]
        if any(v is None for v in vals):
            return None, vals
        return float(np.mean(vals)), vals

    aggregates = {}
    for name, pairs in (("four_corpus", FOUR), ("five_corpus", FIVE)):
        block = {}
        for arm, _, _, _ in ARMS:
            vm, vv = mean_over(pairs, arm, valley)
            om, ov = mean_over(pairs, arm, oracle)
            block[arm] = {"valley_mean": vm, "valley_values": vv,
                          "oracle_mean": om, "oracle_values": ov}
        block["corpora"] = [f"{s}|{c}" for s, c in pairs]
        block["valley_mean_delta_4b_minus_8b"] = (
            None if block["4b"]["valley_mean"] is None
            or block["8b"]["valley_mean"] is None
            else float(block["4b"]["valley_mean"] - block["8b"]["valley_mean"]))
        aggregates[name] = block

    aggregates["four_corpus"]["committed_8b_valley_mean_reference"] = \
        REF_8B_FOUR_CORPUS_VALLEY_MEAN
    aggregates["four_corpus"]["committed_8b_oracle_mean_reference"] = \
        REF_8B_FOUR_CORPUS_ORACLE_MEAN
    aggregates["four_corpus"]["triage_reference"] = {
        "value": REF_TRIAGE_FOUR_CORPUS_MEAN,
        "note": ("TRIAGE's reported four-corpus figure, at 1.73 MLLM calls per "
                 "video against one call per video here. Not a like-for-like "
                 "comparison and never used as a bar."),
    }

    # per-corpus 4B-minus-8B deltas, reported for every cell including strict
    deltas = {}
    for slug, _, dataset, _ in CORPORA:
        for cname in cells[f"{slug}|4b"]["collapses"]:
            v4 = valley(slug, "4b", cname)
            v8 = valley(slug, "8b", cname)
            deltas[f"{slug}|{cname}"] = {
                "valley_4b": v4, "valley_8b": v8,
                "delta": None if (v4 is None or v8 is None) else float(v4 - v8),
                "auc_4b": cells[f"{slug}|4b"]["collapses"][cname]["ranking_auc_dequantized"],
                "auc_8b": cells[f"{slug}|8b"]["collapses"][cname]["ranking_auc_dequantized"],
                "oracle_4b": oracle(slug, "4b", cname),
                "oracle_8b": oracle(slug, "8b", cname),
            }

    res = {
        "protocol": "docs/duplex/PREREG_4b_completion.md",
        "design": ("A fixed model swap evaluated everywhere. One "
                   "Qwen3-VL-4B-Instruct checkpoint and one "
                   "Qwen3-VL-8B-Instruct checkpoint, each scored on all five "
                   "corpora with the same pipeline. No per-corpus model "
                   "selection is performed or reported anywhere."),
        "status": ("Measurement, not a method claim. D3 fired at d804f51 and "
                   "killed the graded angular-rotation mechanism, so no "
                   "mechanism is attached to any scale effect reported here."),
        "compute": ("CPU only for this analysis; no model call. The 4B scoring "
                    "of the three new corpora ran on one RTX 5090 under "
                    "scripts/duplex/run_4b_completion.sh."),
        "frozen_constants": {
            "kde": {"n_grid": 4001, "grid_pad": 2.0, "bandwidth": "scott",
                    "trough": "1 - density(valley)/min(density of the two modes)"},
            "oracle": "macro-F1-maximising threshold; hateful-F1-max reported alongside",
            "label_maps": LABEL_MAP,
            "hateclipseg_collapses": {
                "offensive_union": ("primary: positive if any of hateful, "
                                    "insulting, sexual, violence or harm "
                                    "appears at video level"),
                "hateful_strict": "secondary: only the hateful category maps to 1",
            },
            "repro_tolerance": REPRO_TOL,
        },
        "cells": cells,
        "reproduction_check_8b": repro,
        "per_corpus_delta_4b_minus_8b": deltas,
        "aggregates": aggregates,
        "interpretation_boundaries": [
            "A mean is not a method. The strongest licensed statement is the "
            "mechanism-free one: on these five corpora under this pipeline the "
            "label-free operating point does or does not prefer the "
            "intermediate scale.",
            "No per-corpus model mixing. Neither row may be assembled from the "
            "better of the two arms per corpus.",
            "Ranking and thresholding are separate axes and are reported "
            "separately; a valley win alongside an AUC loss says the recipe "
            "lands better on that arm's score distribution, not that the arm "
            "is the better judge.",
            "HateMM and HateClipSeg both carry hate-adjacent surface features "
            "in their negative class, and the HateClipSeg union collapse is "
            "87.3 percent positive, where a rule that flags almost everything "
            "scores well by accident of prevalence.",
            "Unweighted means over corpora of very different sizes are a crude "
            "summary.",
        ],
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
        f.write("\n")
    print(json.dumps(aggregates, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
