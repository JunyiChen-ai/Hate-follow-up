"""Cross-benchmark measurement: one condition, three datasets, two model sizes.

Method-development measurement on each benchmark's own `train_clean` split. No
pre-registration governs it and none is claimed. The single condition is the
channel-restoration method as it stands after the ImpliHateVid full-corpus run:
the gated fresh Whisper large-v3 transcript fed uncapped to one 8B (or 2B) judge
call, and a label-free KDE-valley threshold computed on that run's own raw-z
distribution.

Every component is inherited, not chosen here. The ASR settings, the degeneracy
gate, the judge prompt and the raw-z readout come from the frozen
channel-restoration and duplex-readout modules; the threshold recipe comes from
`docs/duplex/reports/rawz_detector_train.json` and this code path reproduces
that document's ImpliHateVid C0 valley to the last digit as a self-check before
being applied to a new benchmark.

Labels enter the analysis only. The threshold is computed from the score
distribution with no labels; the oracle threshold is reported as a diagnostic
gap and is never used by the method.

Output: docs/duplex/reports/crossbench_<slug>_<arm>.json. Statistics only: no
transcript text and no individual video ids reach the output.
"""

import argparse
import json
import math
import os
import re
import sys

import numpy as np
from scipy.stats import gaussian_kde

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(ROOT, "src", "our_method"))

# The frozen regex cue family and the rank statistics are imported from the
# kill-test's analysis module so that neither is redefined or re-tuned here.
from channel_restoration_analyze import (  # noqa: E402
    PATS, auc, fisher, has_cue, median, quantile,
)
from data_utils import load_annotations, load_clean_split_ids  # noqa: E402

# Frozen threshold recipe, from docs/duplex/reports/rawz_detector_train.json.
N_GRID = 4001
GRID_PAD = 2.0
FROZEN_IHV_C0_VALLEY_8B = -2.903500000000001

# Binary collapse of each benchmark's own annotation vocabulary. MHClip is
# 3-class and `Offensive` maps to 1: the binary task is Hateful+Offensive vs
# Normal. HateMM is already binary.
LABEL_MAP = {
    "HateMM": {"Hate": 1, "Non Hate": 0},
    "MHClip_EN": {"Hateful": 1, "Offensive": 1, "Normal": 0},
    "MHClip_ZH": {"Hateful": 1, "Offensive": 1, "Normal": 0},
}

REFERENCE_METHODS = ["naive_2b", "holistic_2b", "holistic_8b", "mars_2b",
                     "boundary_rescue", "alarm_backup_7b_20260416"]
REF_DOC = os.path.join(ROOT, "docs", "results_2026_04_16_v2.md")
REF_COLUMN = {"HateMM": "HateMM", "MHClip_EN": "MHClip_EN",
              "MHClip_ZH": "MHClip_ZH"}


def load_z(path):
    d = {}
    if not os.path.exists(path):
        return d
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(r.get("z"), (int, float)) and math.isfinite(r["z"]):
                d[r["video_id"]] = r["z"]
    return d


def load_recs(path):
    d = {}
    if not os.path.exists(path):
        return d
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            d[r["video_id"]] = r
    return d


def kde_valley(z):
    """The frozen label-free threshold recipe from rawz_detector_train.json.

    Gaussian KDE with Scott bandwidth over a 4001-point grid spanning
    [min - 2, max + 2]. The two highest grid local maxima are the modes; the
    threshold is the minimum-density grid point strictly between them. Uses z
    only. Returns None-valued fields when fewer than two modes exist, which is
    itself the diagnostic that the recipe has no valley to find.
    """
    z = np.asarray(z, dtype=float)
    kde = gaussian_kde(z, bw_method="scott")
    grid = np.linspace(z.min() - GRID_PAD, z.max() + GRID_PAD, N_GRID)
    dens = kde(grid)
    loc = [i for i in range(1, N_GRID - 1)
           if dens[i] > dens[i - 1] and dens[i] > dens[i + 1]]
    out = {
        "bandwidth_scott": float(kde.factor * np.std(z, ddof=1)),
        "n_grid_local_maxima": len(loc),
        "grid_points": N_GRID,
        "grid_span": [float(grid[0]), float(grid[-1])],
    }
    if len(loc) < 2:
        out["value"] = None
        out["note"] = "fewer than two modes: the recipe finds no valley"
        return out
    a, b = sorted(sorted(loc, key=lambda i: -dens[i])[:2])
    j = a + 1 + int(np.argmin(dens[a + 1:b]))
    out.update({
        "value": float(grid[j]),
        "mode_locations": [float(grid[a]), float(grid[b])],
        "mode_densities": [float(dens[a]), float(dens[b])],
        "valley_density": float(dens[j]),
        # Trough depth: how far the valley sits below the modes that bracket it.
        # 1.0 would be a valley as high as the mode, i.e. no separation at all.
        "valley_over_lower_mode": float(dens[j] / min(dens[a], dens[b])),
        "valley_over_higher_mode": float(dens[j] / max(dens[a], dens[b])),
        "relative_trough_depth": float(1.0 - dens[j] / min(dens[a], dens[b])),
    })
    return out


def valley_bootstrap(z, n_boot=1000, seed=0):
    """Stability of the valley location under resampling of the score set."""
    rng = np.random.default_rng(seed)
    z = np.asarray(z, dtype=float)
    vals, n_fail = [], 0
    for _ in range(n_boot):
        r = kde_valley(rng.choice(z, size=len(z), replace=True))
        if r.get("value") is None:
            n_fail += 1
        else:
            vals.append(r["value"])
    out = {"n_boot": n_boot, "n_without_valley": n_fail,
           "frac_without_valley": round(n_fail / n_boot, 4)}
    if not vals:
        return out
    v = np.sort(np.array(vals))
    out.update({
        "median": float(np.median(v)),
        "q1": float(np.quantile(v, 0.25)),
        "q3": float(np.quantile(v, 0.75)),
        "ci95": [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))],
        "sd": float(np.std(v, ddof=1)),
    })
    return out


def operating_point(zs, pos, neg, thr):
    tp = sum(1 for v in pos if zs[v] >= thr)
    fn = len(pos) - tp
    fp = sum(1 for v in neg if zs[v] >= thr)
    tn = len(neg) - fp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    pn = tn / (tn + fn) if tn + fn else 0.0
    rn = tn / (tn + fp) if tn + fp else 0.0
    f1n = 2 * pn * rn / (pn + rn) if pn + rn else 0.0
    n = tp + fp + fn + tn
    return {"threshold": thr, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1_hateful": f1,
            "precision_normal": pn, "recall_normal": rn, "f1_normal": f1n,
            "accuracy": (tp + tn) / n if n else 0.0,
            "macro_f1": (f1 + f1n) / 2}


def oracle_f1max(zs, pos, neg):
    """F1-maximizing threshold against gold labels. Diagnostic upper bound."""
    cands = sorted({zs[v] for v in pos} | {zs[v] for v in neg})
    best = None
    for t in cands:
        op = operating_point(zs, pos, neg, t)
        if best is None or op["f1_hateful"] > best["f1_hateful"]:
            best = op
    return best


def auc_np(pos_scores, neg_scores):
    """Rank AUC with tie correction, on raw arrays."""
    s = np.concatenate([pos_scores, neg_scores])
    lab = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    order = np.argsort(s, kind="mergesort")
    s_sorted, lab_sorted = s[order], lab[order]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    n1 = len(pos_scores)
    return float((ranks[lab_sorted == 1].sum() - n1 * (n1 + 1) / 2)
                 / (n1 * len(neg_scores)))


def auc_boot(pos_scores, neg_scores, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    p, n = np.asarray(pos_scores), np.asarray(neg_scores)
    vals = [auc_np(p[rng.integers(0, len(p), len(p))],
                   n[rng.integers(0, len(n), len(n))]) for _ in range(n_boot)]
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def dist(xs):
    if not xs:
        return None
    return {"n": len(xs), "min": round(min(xs), 4),
            "q1": round(quantile(xs, 0.25), 4),
            "median": round(median(xs), 4),
            "q3": round(quantile(xs, 0.75), 4),
            "max": round(max(xs), 4),
            "mean": round(sum(xs) / len(xs), 4)}


def reference_points(dataset):
    """Prior-project numbers for this benchmark, read out of the repo's own
    consolidated results table. Context only: a different split, a different
    model and a different instrument, all recorded alongside each row."""
    col = REF_COLUMN.get(dataset)
    if not col or not os.path.exists(REF_DOC):
        return None
    rows, header = {}, None
    with open(REF_DOC) as f:
        for line in f:
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if header is None and cells and cells[0] == "method":
                header = cells
                continue
            if header is None or len(cells) != len(header) or col not in header:
                continue
            m, var = cells[0], cells[1]
            if m not in REFERENCE_METHODS:
                continue
            val = cells[header.index(col)]
            if val in ("--", ""):
                continue
            best = rows.get(m)
            try:
                mf1 = float(val.split("/")[1])
            except (IndexError, ValueError):
                continue
            if best is None or mf1 > best["macro_f1"]:
                rows[m] = {"variant": var, "acc_over_macro_f1": val,
                           "macro_f1": mf1}
    return {
        "source": "docs/results_2026_04_16_v2.md (this repo)",
        "caveat": "NOT directly comparable to the numbers in this report. Those "
                  "rows are TEST-split results from the prior project, produced "
                  "by a different judge (2B/7B/8B depending on the row) under the "
                  "old clipped-transcript instrument and a generative "
                  "hateful/not decode, not the raw-z readout used here. This "
                  "report measures TRAIN_CLEAN with a raw-z threshold. Read them "
                  "as an order-of-magnitude orientation only.",
        "split": "test", "instrument": "generative decode, clipped transcript",
        "best_variant_per_method": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(LABEL_MAP))
    ap.add_argument("--split", default="train")
    ap.add_argument("--judge-dir", required=True)
    ap.add_argument("--work-dir", required=True,
                    help="the dataset's crossbench working directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--selfcheck-c0-scores", default=os.path.join(
        ROOT, "results/duplex_readout/ImpliHateVid/scores.jsonl"))
    args = ap.parse_args()

    ds = args.dataset
    lmap = LABEL_MAP[ds]
    ann = load_annotations(ds)
    split_ids = load_clean_split_ids(ds, args.split)
    # Defensive de-duplication: MHClip_EN carries one duplicated id upstream.
    seen, ids = set(), []
    n_dup = 0
    for v in split_ids:
        if v in seen:
            n_dup += 1
            continue
        seen.add(v)
        ids.append(v)

    unknown = sorted({ann[v]["label"] for v in ids} - set(lmap))
    if unknown:
        raise SystemExit(f"ABORT: unmapped labels in {ds}: {unknown}")

    z = load_z(os.path.join(args.judge_dir, "scores.jsonl"))
    scored = [v for v in ids if v in z]
    pos = [v for v in scored if lmap[ann[v]["label"]] == 1]
    neg = [v for v in scored if lmap[ann[v]["label"]] == 0]
    zs = [z[v] for v in scored]

    gate = load_recs(os.path.join(args.work_dir, "gate_outcomes.jsonl"))
    asr = load_recs(os.path.join(args.work_dir, "fresh_transcripts.jsonl"))
    meta = load_recs(os.path.join(args.work_dir, "audio_meta.jsonl"))
    ovr_path = os.path.join(args.work_dir, "c2_overrides.json")
    overrides = json.load(open(ovr_path)) if os.path.exists(ovr_path) else {}

    raw_counts = {}
    for v in ids:
        raw_counts[ann[v]["label"]] = raw_counts.get(ann[v]["label"], 0) + 1

    out = {
        "title": f"Cross-benchmark measurement on {ds}, {args.model_name}",
        "status": "method-development measurement on train_clean; NOT a "
                  "kill-test; no pre-registration governs it and none is claimed",
        "dataset": f"{ds} {args.split}_clean, {len(ids)} videos "
                   f"(test split untouched)",
        "condition": "the method as it stands: gated fresh Whisper large-v3 "
                     "transcript fed uncapped to a single judge call, with the "
                     "dataset transcript as the gate's fallback; label-free "
                     "KDE-valley threshold on this run's own raw-z distribution",
        "label_mapping": {"map": lmap,
                          "note": "MHClip is 3-class and Offensive maps to 1: "
                                  "the binary task is Hateful+Offensive vs Normal",
                          "raw_label_counts_train_clean": raw_counts,
                          "n_duplicate_ids_dropped": n_dup},
        "config_provenance": {
            "audio": "scripts/duplex/crossbench_audio.py, which imports ffprobe, "
                     "extract_wav and the 1800 s cap from "
                     "scripts/duplex/channel_restoration_audio.py unmodified; "
                     "ffmpeg 16 kHz mono, silero-VAD speech fraction",
            "asr": "scripts/duplex/crossbench_asr.py, which imports the model id "
                   "and the text statistics from "
                   "scripts/duplex/channel_restoration_asr.py unmodified: "
                   "openai/whisper-large-v3, transformers ASR pipeline, fp16 on "
                   "cuda (asserted), language auto (so MHClip_ZH resolves to zh), "
                   "chunk_length_s=30 long-form, batch_size=8",
            "gate": "scripts/duplex/crossbench_gate.py, which imports "
                    "collapse_repeats and both bounds from "
                    "scripts/duplex/channel_restoration_gate.py unmodified: "
                    "clause-level repetition collapse (max_ngram 8), reject iff "
                    "vad_speech_frac < 0.05 AND gzip ratio of raw fresh text > 7",
            "judge": "src/duplex/extract_duplex_readout.py, unmodified: frozen "
                     "prag reader, 16 frames from frames_16, max_pixels 100352, "
                     "single forward pass, raw unclipped "
                     "z = logsumexp(Yes ids) - logsumexp(No ids), "
                     "transcript_limit 0 (uncapped)",
            "threshold_recipe": "docs/duplex/reports/rawz_detector_train.json, "
                                "estimator b_kde_valley, reimplemented here and "
                                "verified against that document's C0 value",
            "model": args.model_name,
        },
        "mllm_calls_per_video": 1,
        "coverage": {
            "n_train_clean": len(ids),
            "n_scored": len(scored),
            "n_missing": len(ids) - len(scored),
            "n_hateful": len(pos), "n_normal": len(neg),
            "prevalence_hateful": round(len(pos) / len(scored), 4) if scored else None,
        },
    }

    # ---------- restoration coverage: did the ASR arm actually run? ----------
    n_mp4 = sum(1 for v in ids if meta.get(v, {}).get("mp4_exists"))
    n_wav = sum(1 for v in ids if meta.get(v, {}).get("wav_ok"))
    g_counts = {"accepted": 0, "rejected": 0, "no_fresh_pass": 0}
    for v in ids:
        o = gate.get(v, {}).get("outcome")
        if o in g_counts:
            g_counts[o] += 1
    out["restoration_coverage"] = {
        "role": "how much of the judge input the channel-restoration stage "
                "actually replaced. Zero here means the method degenerated to "
                "its own fallback branch and the ASR route was never exercised.",
        "n_with_source_media": n_mp4,
        "n_with_usable_audio": n_wav,
        "n_fresh_transcripts": sum(1 for v in ids if v in asr),
        "gate_outcomes": g_counts,
        "transcript_source_in_judge_input": {
            "override_gated_fresh": sum(1 for v in scored if v in overrides),
            "fallback_dataset_transcript": sum(1 for v in scored
                                               if v not in overrides)},
        "restoration_fraction": round(
            sum(1 for v in scored if v in overrides) / len(scored), 4)
        if scored else None,
    }

    # ---------- self-check: reproduce the frozen ImpliHateVid C0 valley -------
    z_ihv = load_z(args.selfcheck_c0_scores)
    if z_ihv:
        v_ihv = kde_valley(list(z_ihv.values()))
        out["threshold_recipe_selfcheck"] = {
            "note": "this code path applied to the ImpliHateVid C0 scores must "
                    "reproduce the valley frozen in rawz_detector_train.json",
            "n": len(z_ihv),
            "recomputed_valley": v_ihv["value"],
            "frozen_valley": FROZEN_IHV_C0_VALLEY_8B,
            "matches": bool(v_ihv["value"] is not None
                            and abs(v_ihv["value"] - FROZEN_IHV_C0_VALLEY_8B) < 1e-9),
        }
    else:
        out["threshold_recipe_selfcheck"] = {
            "note": "reference C0 scores not on disk; self-check skipped",
            "path": args.selfcheck_c0_scores}

    if not scored:
        out["error"] = "no scored videos; analysis stops here"
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
            f.write("\n")
        print(json.dumps(out, indent=1))
        return

    # ---------- AUC ----------
    zp = np.array([z[v] for v in pos])
    zn = np.array([z[v] for v in neg])
    out["auc"] = {
        "hateful_vs_normal": round(auc(z, pos, neg), 6),
        "boot95": [round(x, 6) for x in auc_boot(zp, zn)],
        "n_pos": len(pos), "n_neg": len(neg),
    }
    if ds in ("MHClip_EN", "MHClip_ZH"):
        hate_only = [v for v in scored if ann[v]["label"] == "Hateful"]
        off_only = [v for v in scored if ann[v]["label"] == "Offensive"]
        out["auc"]["by_3class_subgroup"] = {
            "Hateful_vs_Normal": round(auc(z, hate_only, neg), 6) if hate_only else None,
            "Offensive_vs_Normal": round(auc(z, off_only, neg), 6) if off_only else None,
            "n_Hateful": len(hate_only), "n_Offensive": len(off_only),
        }

    # ---------- label-free threshold ----------
    v = kde_valley(zs)
    out["threshold_label_free"] = {
        "definition": "Gaussian KDE of this run's raw-z values, Scott bandwidth, "
                      "4001-point grid over [min-2, max+2]; the two highest grid "
                      "local maxima are the modes and the threshold is the "
                      "minimum-density grid point strictly between them. Labels "
                      "are not used.",
        "computed_on": f"the {ds} {args.split}_clean score distribution",
        **v,
        "stability_bootstrap": valley_bootstrap(zs),
        "z_distribution": dist(zs),
        "z_distribution_hateful": dist([z[x] for x in pos]),
        "z_distribution_normal": dist([z[x] for x in neg]),
    }

    # ---------- operating points ----------
    ops = {}
    if v["value"] is not None:
        ops["METHOD_self_computed_valley"] = {
            "label_free": True,
            "role": "the method number: threshold computed from this run's own "
                    "score distribution, no labels",
            **operating_point(z, pos, neg, v["value"])}
    ops["diagnostic_zero"] = {
        "label_free": True,
        "role": "diagnostic: the model's own Yes/No boundary",
        **operating_point(z, pos, neg, 0.0)}
    ops["diagnostic_frozen_ImpliHateVid_C0_valley"] = {
        "label_free": True,
        "role": "diagnostic only: the 8B valley frozen on ImpliHateVid C0, "
                "transplanted unchanged, to show how far a threshold travels "
                "across benchmarks",
        **operating_point(z, pos, neg, FROZEN_IHV_C0_VALLEY_8B)}
    orc = oracle_f1max(z, pos, neg)
    ops["diagnostic_oracle_f1max_NOT_label_free"] = {
        "label_free": False,
        "role": "diagnostic upper bound: chosen against gold labels. The method "
                "does not use it.",
        **orc}
    out["operating_points"] = ops
    if v["value"] is not None:
        out["oracle_gap"] = {
            "macro_f1_method": ops["METHOD_self_computed_valley"]["macro_f1"],
            "macro_f1_oracle": orc["macro_f1"],
            "gap": round(orc["macro_f1"]
                         - ops["METHOD_self_computed_valley"]["macro_f1"], 6),
            "threshold_method": v["value"],
            "threshold_oracle": orc["threshold"],
        }

    # ---------- error anatomy at the label-free threshold ----------
    def seen_text(x):
        return overrides.get(x, ann[x]["transcript"] or "")

    if v["value"] is not None:
        thr = v["value"]
        fn_ids = [x for x in pos if z[x] < thr]
        fp_ids = [x for x in neg if z[x] >= thr]
        tn_ids = [x for x in neg if z[x] < thr]
        fn_cue = sum(1 for x in fn_ids if has_cue(seen_text(x)))
        fp_cue = sum(1 for x in fp_ids if has_cue(seen_text(x)))
        tn_cue = sum(1 for x in tn_ids if has_cue(seen_text(x)))
        anat = {
            "threshold": thr,
            "false_negatives": {
                "n": len(fn_ids),
                "z": dist([z[x] for x in fn_ids]),
                "seen_transcript_chars": dist([len(seen_text(x)) for x in fn_ids]),
                "n_with_empty_seen_transcript": sum(
                    1 for x in fn_ids if not seen_text(x).strip()),
                "gate_outcome": {
                    "fell_back_to_dataset_transcript": sum(
                        1 for x in fn_ids if x not in overrides),
                    "used_gated_fresh_transcript": sum(
                        1 for x in fn_ids if x in overrides)},
                "surface_cue_in_seen_transcript": {
                    "n": fn_cue,
                    "frac": round(fn_cue / len(fn_ids), 4) if fn_ids else None},
            },
            "false_positives": {
                "n": len(fp_ids),
                "z": dist([z[x] for x in fp_ids]),
                "seen_transcript_chars": dist([len(seen_text(x)) for x in fp_ids]),
                "surface_cue_in_seen_transcript": {
                    "note": "the frozen regex cue family from the kill-test "
                            "analysis, applied to the transcript the judge read. "
                            "A text-locating device only: never an input to any "
                            "model or threshold. The patterns are English, so on "
                            "MHClip_ZH they under-fire and the rates there are "
                            "not interpretable.",
                    "n": fp_cue,
                    "frac": round(fp_cue / len(fp_ids), 4) if fp_ids else None,
                    "true_negative_baseline_n": tn_cue,
                    "true_negative_baseline_frac": round(
                        tn_cue / len(tn_ids), 4) if tn_ids else None,
                    "fisher_p": fisher(fp_cue, len(fp_ids) - fp_cue,
                                       tn_cue, len(tn_ids) - tn_cue)},
                "by_marker_family": {
                    k: sum(1 for x in fp_ids if p.search(seen_text(x)))
                    for k, p in PATS.items()},
                "gate_outcome": {
                    "fell_back_to_dataset_transcript": sum(
                        1 for x in fp_ids if x not in overrides),
                    "used_gated_fresh_transcript": sum(
                        1 for x in fp_ids if x in overrides)},
            },
        }
        if ds in ("MHClip_EN", "MHClip_ZH"):
            anat["false_negatives"]["by_3class_subgroup"] = {
                "Hateful": sum(1 for x in fn_ids if ann[x]["label"] == "Hateful"),
                "Offensive": sum(1 for x in fn_ids if ann[x]["label"] == "Offensive"),
            }
        out["error_anatomy_at_method_threshold"] = anat

    # ---------- starvation diagnostics ----------
    old_chars = {x: len(ann[x]["transcript"] or "") for x in ids}
    diag = {
        "role": "how much text the judge had, and how much of it the ASR route "
                "replaced. This is the starvation axis the channel-restoration "
                "story runs on.",
        "dataset_transcript_chars": {
            "all": dist([old_chars[x] for x in ids]),
            "hateful": dist([old_chars[x] for x in ids
                             if lmap[ann[x]["label"]] == 1]),
            "normal": dist([old_chars[x] for x in ids
                            if lmap[ann[x]["label"]] == 0]),
            "n_empty": sum(1 for x in ids if old_chars[x] == 0),
            "n_over_300_chars": sum(1 for x in ids if old_chars[x] > 300),
            "frac_over_300_chars": round(
                sum(1 for x in ids if old_chars[x] > 300) / len(ids), 4),
            "note": "300 characters was the old clipped instrument's visible "
                    "window; the fraction above it bounds how much text that "
                    "instrument could never see on this benchmark",
        },
    }
    fresh_ids = [x for x in ids if x in asr and not asr[x].get("error")]
    if fresh_ids:
        diag["fresh_transcript_chars"] = {
            "all": dist([asr[x]["fresh_chars"] for x in fresh_ids]),
            "hateful": dist([asr[x]["fresh_chars"] for x in fresh_ids
                             if lmap[ann[x]["label"]] == 1]),
            "normal": dist([asr[x]["fresh_chars"] for x in fresh_ids
                            if lmap[ann[x]["label"]] == 0]),
        }
        uncapped = [x for x in fresh_ids if not asr[x].get("hit_duration_cap")]
        diag["edit_norm_fresh_vs_dataset"] = {
            "note": "normalized Levenshtein over case-folded, "
                    "punctuation-stripped text; 0 identical, 1 disjoint. Clips "
                    "that hit the 30-minute cap are excluded.",
            "all": dist([asr[x]["edit_norm_vs_dataset"] for x in uncapped]),
            "hateful": dist([asr[x]["edit_norm_vs_dataset"] for x in uncapped
                             if lmap[ann[x]["label"]] == 1]),
            "normal": dist([asr[x]["edit_norm_vs_dataset"] for x in uncapped
                            if lmap[ann[x]["label"]] == 0]),
        }
        diag["gate_rejection_by_label_half"] = {
            half: {
                "n": sum(1 for x in ids if lmap[ann[x]["label"]] == bit),
                "rejected": sum(1 for x in ids
                                if lmap[ann[x]["label"]] == bit
                                and gate.get(x, {}).get("outcome") == "rejected"),
                "no_fresh_pass": sum(1 for x in ids
                                     if lmap[ann[x]["label"]] == bit
                                     and gate.get(x, {}).get("outcome")
                                     == "no_fresh_pass"),
            } for half, bit in (("hateful", 1), ("normal", 0))}
        langs = {}
        for x in fresh_ids:
            langs[asr[x].get("top_language")] = \
                langs.get(asr[x].get("top_language"), 0) + 1
        diag["detected_language_counts"] = langs
        diag["n_hit_30min_cap"] = sum(1 for x in fresh_ids
                                      if asr[x].get("hit_duration_cap"))
        diag["n_asr_error"] = sum(1 for x in ids
                                  if x in asr and asr[x].get("error"))
    else:
        diag["fresh_transcript_chars"] = None
        diag["edit_norm_fresh_vs_dataset"] = None
        diag["unavailable_reason"] = (
            "no fresh transcript exists for this benchmark: the source media "
            "was not reachable, so stage A produced no usable audio and stage B "
            "had nothing to transcribe. Every video therefore took the gate's "
            "fallback branch and the judge read the dataset transcript.")
    out["starvation_diagnostics"] = diag

    out["reference_points_prior_project"] = reference_points(ds)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
