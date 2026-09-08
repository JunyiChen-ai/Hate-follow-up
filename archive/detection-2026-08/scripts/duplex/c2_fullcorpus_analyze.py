"""Full-corpus C2 measurement: AUC, label-free threshold, error anatomy.

This is method-development measurement on ImpliHateVid `train_clean`, not a
kill-test. No pre-registration governs it and none is claimed. The channel
restoration kill-test scored C2 on the 662 dismissed videos only; this run
extends the same pipeline to all 1283 so that the methodized detector --- fresh
full-length Whisper transcript, one 8B judge call, label-free KDE-valley
threshold --- can be read end to end on the whole corpus.

Every configuration is inherited, not chosen here. The ASR settings, the
degeneracy gate, the judge prompt and the raw-z readout come from the frozen
channel-restoration stages; the KDE recipe comes from
docs/duplex/reports/rawz_detector_train.json and reproduces that document's
C0 valley to the last digit as a self-check.

Labels enter the analysis only. The threshold is computed from the score
distribution with no labels; gold EX/IM/NH prefixes are used to evaluate it and
to break down the errors. The oracle threshold is reported as a diagnostic gap
and is never used by the method.

Output: docs/duplex/reports/c2_fullcorpus_<model>.json. Statistics only: no
transcript text and no individual video ids reach the output.
"""

import json
import math
import os
import sys

import numpy as np
from scipy.stats import gaussian_kde

ROOT = "/home/jehc223/Hate-follow-up"
sys.path.insert(0, os.path.join(ROOT, "scripts", "duplex"))
sys.path.insert(0, os.path.join(ROOT, "src", "our_method"))

# The frozen regex cue family and the Step-0 subgroups are imported from the
# kill-test's analysis module so that neither is redefined or re-tuned here.
from channel_restoration_analyze import (  # noqa: E402
    CELL_III_NOSPEECH, FRAGILE, PATS, auc, fisher, has_cue, median, quantile,
)

LIMIT = 300
# Frozen C0 valley for the 8B, from rawz_detector_train.json. Diagnostic
# comparison only; --frozen-valley overrides it for the 2B contrast arm.
DEFAULT_FROZEN_VALLEY = -2.903500000000001
KDE_C0_8B = DEFAULT_FROZEN_VALLEY
N_GRID = 4001
GRID_PAD = 2.0


def load_z(path):
    d = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
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
            if line:
                r = json.loads(line)
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
    if not vals:
        return {"n_boot": n_boot, "n_without_valley": n_fail}
    v = np.sort(np.array(vals))
    return {
        "n_boot": n_boot,
        "n_without_valley": n_fail,
        "median": float(np.median(v)),
        "q1": float(np.quantile(v, 0.25)),
        "q3": float(np.quantile(v, 0.75)),
        "ci95": [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))],
        "sd": float(np.std(v, ddof=1)),
    }


def operating_point(zs, hateful, nh, thr):
    tp = sum(1 for v in hateful if zs[v] >= thr)
    fn = len(hateful) - tp
    fp = sum(1 for v in nh if zs[v] >= thr)
    tn = len(nh) - fp
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
            "accuracy": (tp + tn) / n, "macro_f1": (f1 + f1n) / 2}


def oracle_f1max(zs, hateful, nh):
    """F1-maximizing threshold against gold labels. Diagnostic upper bound."""
    cands = sorted({zs[v] for v in hateful} | {zs[v] for v in nh})
    best = None
    for t in cands:
        op = operating_point(zs, hateful, nh, t)
        if best is None or op["f1_hateful"] > best["f1_hateful"]:
            best = op
    return best


def auc_boot(zs, pos, neg, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    p = np.array([zs[v] for v in pos])
    n = np.array([zs[v] for v in neg])
    vals = []
    for _ in range(n_boot):
        pi = rng.integers(0, len(p), len(p))
        ni = rng.integers(0, len(n), len(n))
        zp = {f"p{i}": float(p[k]) for i, k in enumerate(pi)}
        zn = {f"n{i}": float(n[k]) for i, k in enumerate(ni)}
        zz = dict(zp)
        zz.update(zn)
        vals.append(auc(zz, list(zp), list(zn)))
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def zstats(vals):
    if not vals:
        return None
    return {"n": len(vals), "min": round(min(vals), 4),
            "q1": round(quantile(vals, 0.25), 4),
            "median": round(median(vals), 4),
            "q3": round(quantile(vals, 0.75), 4),
            "max": round(max(vals), 4)}


def main():
    global KDE_C0_8B
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--cr-dir", default=os.path.join(ROOT, "results", "c2_fullcorpus"))
    ap.add_argument("--c0-scores", default=os.path.join(
        ROOT, "results/duplex_readout/ImpliHateVid/scores.jsonl"))
    ap.add_argument("--frozen-valley", type=float, default=DEFAULT_FROZEN_VALLEY,
                    help="the C0 valley frozen for this model, diagnostic only")
    ap.add_argument("--killtest-scores", default=os.path.join(
        ROOT, "results/channel_restoration/c2_8b/scores.jsonl"),
        help="prior C2 run to check determinism against; empty string to skip")
    args = ap.parse_args()

    KDE_C0_8B = args.frozen_valley
    judge_dir, dest, model_name = args.judge_dir, args.out, args.model_name
    cr_dir, c0_path = args.cr_dir, args.c0_scores

    from data_utils import load_annotations, load_clean_split_ids
    ann = load_annotations("ImpliHateVid")
    all1283 = load_clean_split_ids("ImpliHateVid", "train")

    z0 = load_z(c0_path)
    z2 = load_z(os.path.join(judge_dir, "scores.jsonl"))
    z2_kill = load_z(args.killtest_scores) if args.killtest_scores else {}
    gate = load_recs(os.path.join(cr_dir, "gate_outcomes.jsonl"))
    asr = load_recs(os.path.join(cr_dir, "fresh_transcripts.jsonl"))
    with open(os.path.join(cr_dir, "c2_overrides.json")) as f:
        overrides = json.load(f)
    reused = set(load_recs(os.path.join(
        ROOT, "results/channel_restoration/fresh_transcripts.jsonl")))

    scored = [v for v in all1283 if v in z2]
    hateful = [v for v in scored if not v.startswith("NH_")]
    nh = [v for v in scored if v.startswith("NH_")]
    ex = [v for v in scored if v.startswith("EX_")]
    im = [v for v in scored if v.startswith("IM_")]
    zs = [z2[v] for v in scored]

    out = {
        "title": f"Full-corpus C2 measurement, {model_name}",
        "status": "method-development measurement on train_clean; NOT a "
                  "kill-test; no pre-registration governs it and none is claimed",
        "dataset": "ImpliHateVid train_clean, 1283 videos (test split untouched)",
        "condition": "C2 = original mp4 audio re-transcribed with Whisper "
                     "large-v3, gated, fed uncapped to the judge",
        "config_provenance": {
            "asr": "scripts/duplex/channel_restoration_asr.py, unmodified: "
                   "openai/whisper-large-v3, transformers ASR pipeline, fp16 on "
                   "cuda (asserted), language auto, chunk_length_s=30 long-form, "
                   "batch_size=8, 30-minute audio cap applied at wav extraction",
            "audio": "scripts/duplex/channel_restoration_audio.py, unmodified: "
                     "ffmpeg 16 kHz mono, 1800 s cap, silero-VAD speech fraction",
            "gate": "scripts/duplex/channel_restoration_gate.py, unmodified: "
                    "clause-level repetition collapse (max_ngram 8), reject iff "
                    "vad_speech_frac < 0.05 AND gzip ratio of raw fresh text > 7",
            "judge": "src/duplex/extract_duplex_readout.py, unmodified: frozen "
                     "prag reader, 16 frames from frames_16, max_pixels 100352, "
                     "single forward pass, raw unclipped "
                     "z = logsumexp(Yes ids) - logsumexp(No ids)",
            "threshold_recipe": "docs/duplex/reports/rawz_detector_train.json, "
                                "estimator b_kde_valley, reimplemented here and "
                                "verified against that document's C0 value",
            "c0_baseline": c0_path,
        },
        "mllm_calls_per_video": 1,
        "coverage": {
            "n_train_clean": len(all1283),
            "n_scored_C2": len(scored),
            "n_hateful": len(hateful), "n_NH": len(nh),
            "n_EX": len(ex), "n_IM": len(im),
            "n_missing": len(all1283) - len(scored),
        },
    }

    # ---------- self-check: reproduce the frozen C0 valley ----------
    c0_ids = [v for v in all1283 if v in z0]
    c0_valley = kde_valley([z0[v] for v in c0_ids])
    out["c0_recipe_selfcheck"] = {
        "note": "the same code path applied to the C0 scores must reproduce the "
                "frozen valley in rawz_detector_train.json",
        "recomputed_valley": c0_valley["value"],
        "frozen_valley": KDE_C0_8B,
        "matches": bool(c0_valley["value"] is not None
                        and abs(c0_valley["value"] - KDE_C0_8B) < 1e-9),
        "recomputed_bandwidth": c0_valley["bandwidth_scott"],
        "c0_trough_depth": c0_valley.get("relative_trough_depth"),
        "c0_stability_bootstrap": valley_bootstrap([z0[v] for v in c0_ids]),
        "why": "the C0 valley's own sampling spread is the reference against "
               "which the C2 valley's stability is read; neither is a bar",
    }

    # ---------- gate statistics ----------
    def gate_block(vids):
        g = [gate[v] for v in vids if v in gate]
        a = [r for r in asr.values() if r["video_id"] in set(vids)]
        acc = [v for v in vids if v in overrides]
        shrink = [r["collapse_shrink"] for r in g
                  if r.get("collapse_shrink") is not None]
        return {
            "n": len(vids),
            "accepted": sum(1 for r in g if r["outcome"] == "accepted"),
            "rejected": sum(1 for r in g if r["outcome"] == "rejected"),
            "no_fresh_pass": sum(1 for r in g if r["outcome"] == "no_fresh_pass"),
            "reject_rate": round(sum(1 for r in g if r["outcome"] == "rejected")
                                 / len(vids), 6) if vids else None,
            "hit_30min_cap": sum(1 for r in a if r.get("hit_duration_cap")),
            "asr_error": sum(1 for r in a if r.get("error")),
            "median_old_chars": median([r["old_chars"] for r in a]),
            "median_fresh_raw_chars": median([r["fresh_chars"] for r in a]),
            "median_gated_chars": median([len(overrides[v]) for v in acc]),
            "median_edit_norm_vs_dataset": round(median(
                [r["edit_norm_vs_dataset"] for r in a
                 if not r.get("hit_duration_cap")]), 4) if a else None,
            "median_collapse_shrink": round(median(shrink), 4) if shrink else None,
        }

    newly = [v for v in all1283 if v not in reused]
    out["gate_stats"] = {
        "full_corpus": gate_block(all1283),
        "by_label": {"hateful": gate_block([v for v in all1283
                                            if not v.startswith("NH_")]),
                     "NH": gate_block([v for v in all1283 if v.startswith("NH_")])},
        "by_provenance": {
            "reused_from_killtest": gate_block(sorted(reused)),
            "newly_transcribed": gate_block(newly),
        },
        "transcript_source_in_judge_input": {
            "override_gated_fresh": sum(1 for v in scored if v in overrides),
            "fallback_dataset_transcript": sum(1 for v in scored
                                               if v not in overrides),
        },
    }

    # ---------- AUC ----------
    def auc_block(zz):
        ids = [v for v in all1283 if v in zz]
        h = [v for v in ids if not v.startswith("NH_")]
        n_ = [v for v in ids if v.startswith("NH_")]
        e = [v for v in ids if v.startswith("EX_")]
        i_ = [v for v in ids if v.startswith("IM_")]
        return {"hateful_vs_NH": round(auc(zz, h, n_), 6),
                "IM_vs_NH": round(auc(zz, i_, n_), 6),
                "EX_vs_NH": round(auc(zz, e, n_), 6)}

    a2, a0 = auc_block(z2), auc_block(z0)
    out["auc"] = {
        "C2_full_corpus": a2,
        "C2_hateful_vs_NH_boot95": [round(x, 6)
                                    for x in auc_boot(z2, hateful, nh)],
        "C0_baseline": a0,
        "delta_C2_minus_C0": {k: round(a2[k] - a0[k], 6) for k in a2},
    }

    # ---------- label-free threshold ----------
    v2 = kde_valley(zs)
    out["threshold_label_free"] = {
        "definition": "Gaussian KDE of the C2 raw-z values, Scott bandwidth, "
                      "4001-point grid over [min-2, max+2]; the two highest grid "
                      "local maxima are the modes and the threshold is the "
                      "minimum-density grid point strictly between them. Labels "
                      "are not used.",
        "computed_on": "the real full-corpus C2 distribution",
        **v2,
        "stability_bootstrap": valley_bootstrap(zs),
        "distance_from_frozen_C0_valley": (
            round(v2["value"] - KDE_C0_8B, 4) if v2["value"] is not None else None),
    }

    # ---------- operating points ----------
    ops = {}
    if v2["value"] is not None:
        ops["METHOD_self_computed_valley"] = {
            "label_free": True,
            "role": "the method number: threshold computed from this run's own "
                    "score distribution, no labels",
            **operating_point(z2, hateful, nh, v2["value"])}
    ops["diagnostic_frozen_C0_valley"] = {
        "label_free": True,
        "role": "diagnostic only: the valley frozen from the C0 distribution, "
                "carried over unchanged",
        **operating_point(z2, hateful, nh, KDE_C0_8B)}
    orc = oracle_f1max(z2, hateful, nh)
    ops["diagnostic_oracle_f1max_NOT_label_free"] = {
        "label_free": False,
        "role": "diagnostic upper bound: chosen against gold labels. The method "
                "does not use it.",
        **orc}
    ops["diagnostic_zero"] = {
        "label_free": True,
        "role": "diagnostic: the model's own Yes/No boundary",
        **operating_point(z2, hateful, nh, 0.0)}
    out["operating_points"] = ops
    if v2["value"] is not None:
        out["oracle_gap"] = {
            "macro_f1_method": ops["METHOD_self_computed_valley"]["macro_f1"],
            "macro_f1_oracle": orc["macro_f1"],
            "gap": round(orc["macro_f1"]
                         - ops["METHOD_self_computed_valley"]["macro_f1"], 6),
            "threshold_method": v2["value"],
            "threshold_oracle": orc["threshold"],
        }

    # C0 baseline at its own frozen valley, for the headline comparison
    c0_h = [v for v in c0_ids if not v.startswith("NH_")]
    c0_n = [v for v in c0_ids if v.startswith("NH_")]
    out["c0_baseline_operating_point"] = {
        "role": "the published C0 label-free number this run is measured against",
        **operating_point(z0, c0_h, c0_n, KDE_C0_8B)}

    # ---------- error anatomy at the self-computed valley ----------
    if v2["value"] is not None:
        thr = v2["value"]
        fn_ids = [v for v in hateful if z2[v] < thr]
        fp_ids = [v for v in nh if z2[v] >= thr]
        tn_ids = [v for v in nh if z2[v] < thr]

        def seen_text(v):
            """The transcript the judge actually read under C2."""
            return overrides.get(v, ann[v]["transcript"] or "")

        fn_by_marker = sum(1 for v in fn_ids if has_cue(seen_text(v)))
        fp_by_marker = sum(1 for v in fp_ids if has_cue(seen_text(v)))
        tn_by_marker = sum(1 for v in tn_ids if has_cue(seen_text(v)))
        step0_res = set(CELL_III_NOSPEECH)
        frag = set(FRAGILE)
        out["error_anatomy_at_method_threshold"] = {
            "threshold": thr,
            "false_negatives": {
                "n": len(fn_ids),
                "by_group": {"EX": sum(1 for v in fn_ids if v.startswith("EX_")),
                             "IM": sum(1 for v in fn_ids if v.startswith("IM_"))},
                "share_IM": round(sum(1 for v in fn_ids if v.startswith("IM_"))
                                  / len(fn_ids), 4) if fn_ids else None,
                "recall_loss_within_group": {
                    "EX": round(sum(1 for v in fn_ids if v.startswith("EX_"))
                                / len(ex), 4),
                    "IM": round(sum(1 for v in fn_ids if v.startswith("IM_"))
                                / len(im), 4)},
                "z": zstats([z2[v] for v in fn_ids]),
                "step0_no_meaningful_speech_residuals": {
                    "definition": "the 8 videos Step 0 established have no "
                                  "transcribable speech, so C2 has nothing to "
                                  "restore for them",
                    "n_subgroup": len(step0_res),
                    "n_still_FN": sum(1 for v in fn_ids if v in step0_res)},
                "step0_fragile_language": {
                    "n_subgroup": len(frag),
                    "n_still_FN": sum(1 for v in fn_ids if v in frag)},
                "gate_outcome": {
                    "fell_back_to_dataset_transcript": sum(
                        1 for v in fn_ids if v not in overrides),
                    "used_gated_fresh_transcript": sum(
                        1 for v in fn_ids if v in overrides)},
                "surface_cue_in_seen_transcript": {
                    "n": fn_by_marker,
                    "frac": round(fn_by_marker / len(fn_ids), 4) if fn_ids else None},
                "was_FN_under_C0": sum(1 for v in fn_ids
                                       if v in z0 and z0[v] < KDE_C0_8B),
            },
            "false_positives": {
                "n": len(fp_ids),
                "all_NH_by_construction": True,
                "z": zstats([z2[v] for v in fp_ids]),
                "surface_cue_in_seen_transcript": {
                    "note": "the frozen regex cue family from the kill-test "
                            "analysis, applied to the C2 transcript the judge "
                            "read. A text-locating device only: never an input "
                            "to any model or threshold.",
                    "n": fp_by_marker,
                    "frac": round(fp_by_marker / len(fp_ids), 4) if fp_ids else None,
                    "true_negative_baseline_n": tn_by_marker,
                    "true_negative_baseline_frac": round(
                        tn_by_marker / len(tn_ids), 4) if tn_ids else None,
                    "fisher_p": fisher(fp_by_marker, len(fp_ids) - fp_by_marker,
                                       tn_by_marker, len(tn_ids) - tn_by_marker),
                },
                "by_marker_family": {
                    k: sum(1 for v in fp_ids if p.search(seen_text(v)))
                    for k, p in PATS.items()},
                "gate_outcome": {
                    "fell_back_to_dataset_transcript": sum(
                        1 for v in fp_ids if v not in overrides),
                    "used_gated_fresh_transcript": sum(
                        1 for v in fp_ids if v in overrides)},
                "was_FP_under_C0": sum(1 for v in fp_ids
                                       if v in z0 and z0[v] >= KDE_C0_8B),
            },
        }

    # ---------- determinism against the kill-test C2 run ----------
    common = [v for v in scored if v in z2_kill]
    deltas = [abs(z2[v] - z2_kill[v]) for v in common]
    out["determinism_vs_killtest_c2_8b"] = {
        "note": "these videos received a byte-identical judge input in both "
                "runs, so their z must be exactly equal",
        "reference_run": args.killtest_scores or "skipped: no prior run",
        "n_common": len(common),
        "n_exactly_equal": sum(1 for d in deltas if d == 0.0),
        "max_abs_delta": round(max(deltas), 8) if deltas else None,
        "all_equal": bool(deltas and max(deltas) == 0.0),
    }

    with open(dest, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
