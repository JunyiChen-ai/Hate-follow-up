"""Occupancy-gated anchored operating point, confirmed on HateClipSeg.

Preregistration: docs/duplex/PREREG_gated_anchor_hateclipseg.md, frozen
2026-08-08 before HateClipSeg was scored by any judge in this project.

The gate is one label-free number: the fraction of the corpus whose raw judge
score sits in the model's positive saturation band, z >= +13. If that fraction
reaches 10 percent the rule anchors the positive mixture component at the
model-owned saturation location +15.0; otherwise it falls back to a fully
corpus-fit two-component mixture. Every comparator is fitted either way, so the
gate's choice can be checked against the arm that actually did better.

All fitting, thresholding and resampling code is imported from
`scripts/duplex/anchored_operating_point.py`, the predecessor experiment, so the
EM protocol, the posterior-0.5 decision, the KDE-valley recipe and the
prevalence resampling are the same implementations that produced the four-corpus
result. Nothing is re-implemented here.

CPU only, deterministic given seed 20260808. Labels are used to evaluate, to
build the oracle and to construct the prevalence resamples; no fitted component
reads a label.

Output: results/hateclipseg/gated_anchor_results.json. Statistics only: no video
id, no transcript text and no recognised string reaches the output.
"""

import json
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(ROOT, "src", "our_method"))
os.environ.setdefault("HVD_DATA_ROOT", "/home/jehc223/data")

from anchored_operating_point import (  # noqa: E402
    ANCHOR_PLACEBO, ANCHOR_PRIMARY, N_RESAMPLES, RESAMPLE_RATES, SEED,
    N_RESTARTS, SIGMA_FLOOR, _fit_gmm, gmm_report, macro_f1, oracle_f1_hateful,
    oracle_macro_f1, prevalence_stress, valley_report,
)
from channel_restoration_analyze import auc  # noqa: E402
from crossbench_analyze import load_z  # noqa: E402
from data_utils import load_clean_split_ids  # noqa: E402
from hateclipseg_prep import video_labels  # noqa: E402

# ---------------------------------------------------------------- constants --
# Frozen by docs/duplex/PREREG_gated_anchor_hateclipseg.md.
GATE_BAND = 13.0          # saturation band lower edge, from the anchor pilot
GATE_FIRE_FRACTION = 0.10  # the gate fires iff at least this share sits in band

SCORES = os.path.join(ROOT, "results", "hateclipseg", "judge_8b", "scores.jsonl")
OUT_DIR = os.path.join(ROOT, "results", "hateclipseg")

# Resampling seed index. The predecessor keyed its resample streams by corpus
# position 0..3; HateClipSeg continues that numbering, one stream per collapse.
COLLAPSE_INDEX = {"offensive_union": 4, "hateful_strict": 5}


# --------------------------------------------------------------------- data --
def load_corpus():
    """(video ids in split order, z array, {collapse: label array}, counts)."""
    ids = []
    seen = set()
    for v in load_clean_split_ids("HateClipSeg", "test"):
        if v not in seen:
            seen.add(v)
            ids.append(v)
    z = load_z(SCORES)
    labels = video_labels()
    scored = [v for v in ids if v in z]
    zs = np.array([z[v] for v in scored], dtype=float)
    ys = {
        "offensive_union": np.array([labels[v][0] for v in scored], dtype=int),
        "hateful_strict": np.array([labels[v][1] for v in scored], dtype=int),
    }
    counts = {"n_annotated": len(labels), "n_split_clean": len(ids),
              "n_scored": len(scored)}
    return scored, zs, ys, counts


def attrition():
    """Where the annotated videos went, counted the way the testruns pipeline
    counts them."""
    from hateclipseg_prep import DST, PILOT_VIDEOS
    labels = video_labels()
    ids = set(load_clean_split_ids("HateClipSeg", "test"))
    # A media file exists for a video if the source directory holds one; it is
    # usable if the prep stage's decodability prune left it linked.
    shipped = {os.path.splitext(f)[0] for f in os.listdir(PILOT_VIDEOS)}
    have_media = {v for v in labels if v in shipped}
    decodable = {v for v in labels
                 if os.path.exists(os.path.join(DST, "video", v + ".mp4"))}
    meta_path = os.path.join(OUT_DIR, "audio_meta.jsonl")
    n_audio_ok = n_no_audio_track = 0
    if os.path.exists(meta_path):
        for line in open(meta_path):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("wav_ok"):
                n_audio_ok += 1
            elif r.get("mp4_exists") and not r.get("has_audio"):
                n_no_audio_track += 1
    return {
        "n_annotated_videos": len(labels),
        "n_with_media_file": len(have_media),
        "n_no_media_anywhere": len(labels) - len(have_media),
        "n_media_undecodable_pruned": len(have_media) - len(decodable),
        "n_decodable_media": len(decodable),
        "n_in_test_clean": len(ids),
        "n_with_usable_audio": n_audio_ok,
        "n_media_present_but_no_audio_track": n_no_audio_track,
    }


def gate_statistics():
    """Transcript-restoration gate outcomes, reported descriptively."""
    path = os.path.join(OUT_DIR, "gate_outcomes.jsonl")
    if not os.path.exists(path):
        return {}
    counts, vad, shrink = {}, [], []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
        if r.get("vad_speech_frac") is not None:
            vad.append(r["vad_speech_frac"])
        if r.get("collapse_shrink") is not None:
            shrink.append(r["collapse_shrink"])
    out = {"outcomes": counts,
           "n_override_ids": len(json.load(open(
               os.path.join(OUT_DIR, "c2_overrides.json"))))}
    if vad:
        out["vad_speech_frac"] = {
            "median": float(np.median(vad)),
            "p10": float(np.percentile(vad, 10)),
            "p90": float(np.percentile(vad, 90))}
    if shrink:
        out["repetition_collapse_shrink"] = {
            "median": float(np.median(shrink)),
            "frac_above_0.5": float(np.mean([s > 0.5 for s in shrink]))}
    return out


def z_summary(zs):
    q = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    edges = list(range(-25, 26, 5))
    hist, _ = np.histogram(zs, bins=[-np.inf] + edges + [np.inf])
    return {
        "n": int(zs.size),
        "min": float(zs.min()), "max": float(zs.max()),
        "mean": float(zs.mean()), "std": float(zs.std(ddof=1)),
        "percentiles": {f"p{p}": float(np.percentile(zs, p)) for p in q},
        "histogram_bin_edges": ["-inf"] + [str(e) for e in edges] + ["+inf"],
        "histogram_counts": [int(c) for c in hist],
        "frac_at_or_above_13": float(np.mean(zs >= GATE_BAND)),
        "frac_at_or_above_15": float(np.mean(zs >= 15.0)),
        "frac_at_or_below_minus_13": float(np.mean(zs <= -GATE_BAND)),
    }


# ------------------------------------------------------------------ fitting --
def fit_all(zs, ys, collapse_index):
    """Every comparator, fitted regardless of what the gate decided."""
    med = float(np.median(zs))
    methods = {}
    methods["valley"] = valley_report(zs, ys)
    fit = _fit_gmm(zs, [SEED, collapse_index, 100], anchor=ANCHOR_PRIMARY)
    methods["anchored_15.0"] = gmm_report(fit, zs, ys, med)
    for a in ANCHOR_PLACEBO:
        f = _fit_gmm(zs, [SEED, collapse_index, int(a)], anchor=a)
        methods[f"placebo_{a}"] = gmm_report(f, zs, ys, med)
    ffit = _fit_gmm(zs, [SEED, collapse_index, 200], anchor=None)
    methods["free_gmm"] = gmm_report(ffit, zs, ys, med)
    t_or, m_or = oracle_macro_f1(zs, ys)
    methods["oracle_macro_f1_max"] = dict(m_or, threshold=t_or, label_free=False)
    t_or2, m_or2 = oracle_f1_hateful(zs, ys)
    methods["oracle_f1_hateful_max"] = dict(m_or2, threshold=t_or2,
                                            label_free=False)
    return methods


def evaluate_clauses(methods, stress, gate_fires):
    """The frozen decision rule, clause by clause."""
    a = methods["anchored_15.0"]["macro_f1"]
    fr = methods["free_gmm"]["macro_f1"]
    # A valley of None means the incumbent KDE recipe found fewer than two
    # modes, so it has no operating point at all. The clauses that compare
    # against it then hold trivially, and the report says so rather than
    # crashing on the missing number.
    va = methods["valley"]["macro_f1"]
    valley_exists = va is not None
    if not valley_exists:
        va = float("-inf")
    p10 = methods["placebo_10.0"]["macro_f1"]
    p20 = methods["placebo_20.0"]["macro_f1"]
    selected = "anchored_15.0" if gate_fires else "free_gmm"
    sel_f1 = methods[selected]["macro_f1"]
    better = max(a, fr)

    c1_ok = sel_f1 >= better - 0.02
    clauses = {
        "clause_1_correct_selection": {
            "rule": "the arm the gate selects is within 0.02 macro-F1 of the "
                    "better of {anchored, free 2-GMM}",
            "selected_arm": selected,
            "selected_macro_f1": sel_f1,
            "anchored_macro_f1": a, "free_gmm_macro_f1": fr,
            "better_arm_macro_f1": better,
            "shortfall": better - sel_f1,
            "result": "PASS" if c1_ok else "FAIL"},
    }
    applicable = [c1_ok]

    if gate_fires:
        drift = {}
        d_ok = True
        for rate in RESAMPLE_RATES:
            s = stress[f"rate_{rate}"]["methods"]
            da = s["anchored"]["median_abs_threshold_drift"]
            df = s["free_gmm"]["median_abs_threshold_drift"]
            ok = (da is not None and df is not None and da <= 0.5 * df)
            drift[f"rate_{rate}"] = {
                "anchored_median_abs_drift": da,
                "free_gmm_median_abs_drift": df,
                "ratio": (da / df if da is not None and df else None),
                "at_or_below_half": ok}
            d_ok = d_ok and ok
        sub = {
            "anchored_at_least_valley_plus_0.03": a >= va + 0.03,
            "anchored_at_least_free_minus_0.005": a >= fr - 0.005,
            "anchored_at_least_placebo_10_minus_0.01": a >= p10 - 0.01,
            "anchored_at_least_placebo_20_minus_0.01": a >= p20 - 0.01,
            "drift_at_or_below_half_free_at_both_rates": d_ok,
        }
        c2_ok = all(sub.values())
        clauses["clause_2_gate_fired"] = {
            "rule": "anchored >= valley + 0.03; anchored >= free - 0.005; "
                    "anchored >= each placebo - 0.01; anchored median threshold "
                    "drift at most half the free-2-GMM drift at both rates",
            "anchored_macro_f1": a,
            "valley_macro_f1": methods["valley"]["macro_f1"],
            "valley_exists": valley_exists,
            "free_gmm_macro_f1": fr,
            "placebo_10_macro_f1": p10, "placebo_20_macro_f1": p20,
            "sub_conditions": sub, "drift": drift,
            "result": "PASS" if c2_ok else "FAIL"}
        applicable.append(c2_ok)
        clauses["clause_3_gate_did_not_fire"] = {
            "applicable": False,
            "reason": "the gate fired, so clause 3 does not apply"}
    else:
        sub = {
            "free_at_least_valley_minus_0.005": fr >= va - 0.005,
            "forced_anchored_underperforms_free_by_more_than_0.02": (fr - a) > 0.02,
        }
        c3_ok = all(sub.values())
        clauses["clause_2_gate_fired"] = {
            "applicable": False,
            "reason": "the gate did not fire, so clause 2 does not apply"}
        clauses["clause_3_gate_did_not_fire"] = {
            "rule": "free 2-GMM >= valley - 0.005, and the forced anchored fit "
                    "underperforms free 2-GMM by more than 0.02",
            "free_gmm_macro_f1": fr,
            "valley_macro_f1": methods["valley"]["macro_f1"],
            "valley_exists": valley_exists,
            "anchored_macro_f1": a,
            "free_minus_anchored": fr - a,
            "sub_conditions": sub,
            "result": "PASS" if c3_ok else "FAIL"}
        applicable.append(c3_ok)

    clauses["overall"] = "PASS" if all(applicable) else "FAIL"
    return clauses


# -------------------------------------------------------------------- main --
def main():
    ids, zs, ys, counts = load_corpus()
    if counts["n_scored"] != counts["n_split_clean"]:
        raise SystemExit(
            f"ABORT: {counts['n_scored']} scored but {counts['n_split_clean']} "
            f"ids in test_clean; the judge stage did not cover the corpus")
    del ids

    gate_frac = float(np.mean(zs >= GATE_BAND))
    gate_fires = gate_frac >= GATE_FIRE_FRACTION
    gate = {
        "statistic": "fraction of the corpus with raw z at or above +13",
        "band_lower_edge": GATE_BAND,
        "fire_threshold": GATE_FIRE_FRACTION,
        "n_in_band": int(np.sum(zs >= GATE_BAND)),
        "n_total": int(zs.size),
        "fraction_in_band": gate_frac,
        "fires": bool(gate_fires),
        "selected_arm": "anchored_15.0" if gate_fires else "free_gmm",
        "uses_labels": False,
    }
    print(json.dumps(gate, indent=1), flush=True)

    per_collapse = {}
    for collapse, y in ys.items():
        ci = COLLAPSE_INDEX[collapse]
        methods = fit_all(zs, y, ci)
        full_thr = {"valley": methods["valley"]["threshold"],
                    "free_gmm": methods["free_gmm"]["threshold"],
                    "anchored": methods["anchored_15.0"]["threshold"]}
        print(f"[{collapse}] fitted; running prevalence stress", flush=True)
        stress = prevalence_stress(zs, y, ci, full_thr)
        pos = np.flatnonzero(y == 1)
        neg = np.flatnonzero(y == 0)
        zdict = {int(i): float(zs[i]) for i in range(zs.size)}
        per_collapse[collapse] = {
            "n_hateful": int(y.sum()), "n_normal": int((1 - y).sum()),
            "prevalence_hateful": round(float(y.mean()), 4),
            "judge_roc_auc": (auc(zdict, [int(i) for i in pos],
                                  [int(i) for i in neg])
                              if pos.size and neg.size else None),
            "methods": methods,
            "prevalence_stress": stress,
            "decision_rule_verdict": evaluate_clauses(methods, stress, gate_fires),
        }
        print(f"[{collapse}] verdict "
              f"{per_collapse[collapse]['decision_rule_verdict']['overall']}",
              flush=True)

    out = {
        "title": "Occupancy-gated anchored operating point on HateClipSeg, "
                 "Qwen3-VL-8B judge, held-out corpus",
        "preregistration": "docs/duplex/PREREG_gated_anchor_hateclipseg.md",
        "status": "preregistered confirmatory experiment, run once; the judge "
                  "pass is the frozen c2 test pipeline; all fitting is CPU only",
        "frozen_constants": {
            "gate_band_lower_edge": GATE_BAND,
            "gate_fire_fraction": GATE_FIRE_FRACTION,
            "anchor_primary": ANCHOR_PRIMARY,
            "anchor_placebo": list(ANCHOR_PLACEBO),
            "seed": SEED, "em_restarts": N_RESTARTS,
            "resample_rates": list(RESAMPLE_RATES),
            "n_resamples_per_rate": N_RESAMPLES,
            "sigma_floor": SIGMA_FLOOR,
            "decision": "predict hateful iff posterior of the positive "
                        "component is at least 0.5"},
        "inputs": "results/hateclipseg/judge_8b/scores.jsonl (8B only)",
        "label_collapse": {
            "primary": "offensive_union: any of hateful, insulting, sexual, "
                       "violence, harm at video level maps to 1",
            "secondary": "hateful_strict: only the hateful category maps to 1",
            "source": "idea-stage/pilots/b1_coverage_audit/data/"
                      "video_level_annotation.csv",
            "lexicons_json": "quarantined, never read"},
        "corpus_counts": counts,
        "attrition": attrition(),
        "transcript_restoration_gate": gate_statistics(),
        "z_distribution": z_summary(zs),
        "occupancy_gate": gate,
        "summary_macro_f1": {
            c: {k: per_collapse[c]["methods"][k]["macro_f1"]
                for k in ("anchored_15.0", "placebo_10.0", "placebo_20.0",
                          "free_gmm", "valley", "oracle_macro_f1_max",
                          "oracle_f1_hateful_max")}
            for c in per_collapse},
        "summary_threshold": {
            c: {k: per_collapse[c]["methods"][k]["threshold"]
                for k in ("anchored_15.0", "placebo_10.0", "placebo_20.0",
                          "free_gmm", "valley", "oracle_macro_f1_max")}
            for c in per_collapse},
        "judge_roc_auc": {c: per_collapse[c]["judge_roc_auc"]
                          for c in per_collapse},
        "verdict_primary_collapse":
            per_collapse["offensive_union"]["decision_rule_verdict"]["overall"],
        "per_collapse": per_collapse,
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "gated_anchor_results.json"), "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(json.dumps({
        "verdict_primary": out["verdict_primary_collapse"],
        "gate": {"fraction_in_band": gate["fraction_in_band"],
                 "fires": gate["fires"], "selected": gate["selected_arm"]},
        "summary_macro_f1": out["summary_macro_f1"],
        "judge_roc_auc": out["judge_roc_auc"]}, indent=1))


if __name__ == "__main__":
    main()
