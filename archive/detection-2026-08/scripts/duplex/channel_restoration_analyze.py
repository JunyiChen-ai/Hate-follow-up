"""Channel restoration: P1-P5 analysis and verdict.

Reads the C0 baseline scores already on disk, the C1 and C2 rerun scores, the
2B C1 scores, and the stage-A/B/C audio records, and evaluates the
pre-registered predictions against their frozen bars. Emits statistics only: no
transcript text reaches the output.

Pre-registration: docs/duplex/PREREG_channel_restoration.md.
"""

import json
import math
import os
import re
import sys

ROOT = "/home/jehc223/Hate-follow-up"
CR = os.path.join(ROOT, "results", "channel_restoration")
LIMIT = 300
Z_DISMISS = -2.944
KDE_8B = -2.9035000000000001
KDE_2B = 0.374625

# The frozen regex cue family. A text-locating device for the analysis only:
# never an input to any model or threshold (prereg, label-use statement).
MARKERS = {
    "identity": r"\b(black|white|jew|jewish|muslim|islam|arab|asian|chinese|indian|mexican|latino|hispanic|african|immigrant|migrant|refugee|gay|lesbian|trans|transgender|queer|women|woman|female|men|man|male|christian|catholic|hindu|race|racial|ethnic|nationality)\w*\b",
    "slur_or_profanity": r"\b(nigg\w*|f[a4]gg?\w*|k[iy]ke|spic|chink|gook|wetback|tranny|retard\w*|bitch\w*|whore|slut|cunt|fuck\w*|shit\w*|bastard\w*)\b",
    "violence": r"\b(kill|killed|killing|shoot|shot|stab|beat|beating|attack\w*|murder\w*|rape|raped|lynch\w*|die|death|destroy|hang)\b",
    "meta_hate": r"\b(racist|racism|hate|hateful|bigot\w*|nazi|hitler|holocaust|supremac\w*|antisemit\w*|xenophob\w*|homophob\w*|misogyn\w*)\b",
}
PATS = {k: re.compile(v, re.I) for k, v in MARKERS.items()}

# Step-0 subgroups, named in the pre-registration before this run.
CELL_II_ASR = ["EX_222", "EX_408", "EX_494", "IM_139", "IM_218", "IM_86"]
CELL_III_NOSPEECH = ["EX_231", "EX_439", "EX_52", "IM_227", "IM_276",
                     "IM_320", "IM_328", "IM_436"]
FRAGILE = ["IM_122", "IM_189", "IM_367"]


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
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                d[r["video_id"]] = r
    return d


def fisher(a, b, c, d):
    """Two-sided Fisher exact on [[a,b],[c,d]]."""
    lf = lambda x: math.lgamma(x + 1)
    n = a + b + c + d
    def p(x):
        b_, c_, d_ = a + b - x, a + c - x, d - (x - a)
        if min(x, b_, c_, d_) < 0:
            return 0.0
        return math.exp(lf(a + b) + lf(c + d) + lf(a + c) + lf(b + d) - lf(n)
                        - lf(x) - lf(b_) - lf(c_) - lf(d_))
    p0, tot = p(a), 0.0
    for x in range(max(0, a - d), min(a + b, a + c) + 1):
        px = p(x)
        if px <= p0 * (1 + 1e-9):
            tot += px
    return min(1.0, tot)


def median(xs):
    xs = sorted(xs)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] if lo == hi else xs[lo] + (pos - lo) * (xs[hi] - xs[lo])


def auc(scores, pos, neg):
    rows = sorted([(scores[v], 1) for v in pos] + [(scores[v], 0) for v in neg])
    r, i = [0.0] * len(rows), 0
    while i < len(rows):
        j = i
        while j + 1 < len(rows) and rows[j + 1][0] == rows[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[k] = avg
        i = j + 1
    n1 = len(pos)
    s = sum(r[k] for k in range(len(rows)) if rows[k][1] == 1)
    return (s - n1 * (n1 + 1) / 2) / (n1 * len(neg))


def has_cue(text):
    return any(p.search(text) for p in PATS.values())


def rate_block(name, flipped, total):
    return {"group": name, "n": total, "flipped": flipped,
            "flip_rate": round(flipped / total, 6) if total else None}


def main():
    ids = json.load(open(sys.argv[1]))
    dismissed = ids["dismissed_all"]
    hateful = ids["dismissed_hateful"]
    nh = ids["dismissed_nh"]
    nh150 = ids["nh_sample150_seed0"]

    sys.path.insert(0, os.path.join(ROOT, "src", "our_method"))
    from data_utils import load_annotations, load_clean_split_ids
    ann = load_annotations("ImpliHateVid")
    all1283 = load_clean_split_ids("ImpliHateVid", "train")

    z0 = load_z(os.path.join(ROOT, "results/duplex_readout/ImpliHateVid/scores.jsonl"))
    z1 = load_z(os.path.join(CR, "c1_8b/scores.jsonl"))
    z2 = load_z(os.path.join(CR, "c2_8b/scores.jsonl"))
    z2b0 = load_z(os.path.join(
        ROOT, "results/duplex_readout/ImpliHateVid_Qwen3-VL-2B-Instruct/scores.jsonl"))
    z2b1 = load_z(os.path.join(CR, "c1_2b/scores.jsonl"))

    asr = load_recs(os.path.join(CR, "fresh_transcripts.jsonl"))
    gate = load_recs(os.path.join(CR, "gate_outcomes.jsonl"))
    overrides = json.load(open(os.path.join(CR, "c2_overrides.json")))

    out = {
        "title": "Channel restoration kill-test",
        "prereg": "docs/duplex/PREREG_channel_restoration.md",
        "dataset": "ImpliHateVid train_clean (test split untouched)",
        "flip_definition": "z > 0 under the intervention, for a video that started below -2.944",
        "coverage": {
            "n_dismissed": len(dismissed), "n_hateful": len(hateful), "n_nh": len(nh),
            "c1_8b_scored": sum(1 for v in dismissed if v in z1),
            "c2_8b_scored": sum(1 for v in dismissed if v in z2),
            "c1_2b_scored": sum(1 for v in all1283 if v in z2b1),
            "n_all_1283": len(all1283),
        },
    }

    def flips(zs, vids):
        return [v for v in vids if v in zs and zs[v] > 0]

    # ---------- z distributions per condition per group ----------
    dist = {}
    for cname, zs, groups in [
        ("C0", z0, {"dismissed_hateful": hateful, "dismissed_nh": nh}),
        ("C1", z1, {"dismissed_hateful": hateful, "dismissed_nh": nh}),
        ("C2", z2, {"dismissed_hateful": hateful, "dismissed_nh": nh}),
    ]:
        dist[cname] = {}
        for g, vids in groups.items():
            vals = [zs[v] for v in vids if v in zs]
            dist[cname][g] = {
                "n": len(vals), "median_z": round(median(vals), 4) if vals else None,
                "q1": round(quantile(vals, 0.25), 4) if vals else None,
                "q3": round(quantile(vals, 0.75), 4) if vals else None,
                "max_z": round(max(vals), 4) if vals else None,
            }
    out["z_distribution_by_condition"] = dist

    # ---------- P1 ----------
    fh2, fn2 = flips(z2, hateful), flips(z2, nh)
    fh1, fn1 = flips(z1, hateful), flips(z1, nh)
    rh2, rn2 = len(fh2) / len(hateful), len(fn2) / len(nh)
    rh1, rn1 = len(fh1) / len(hateful), len(fn1) / len(nh)
    asym2 = rh2 - rn2
    p2f = fisher(len(fh2), len(hateful) - len(fh2), len(fn2), len(nh) - len(fn2))
    p1f = fisher(len(fh1), len(hateful) - len(fh1), len(fn1), len(nh) - len(fn1))
    if asym2 >= 0.15 and p2f < 0.01:
        p1v = "PASS"
    elif asym2 < 0.05:
        p1v = "FAIL-KILL"
    else:
        p1v = "FAIL"
    out["P1_flip_asymmetry"] = {
        "bar": "C2 flip-rate(hateful) - flip-rate(NH) >= 0.15 and Fisher p < 0.01; "
               "asymmetry < 0.05 or comparable NH rate = kill",
        "C2": {"hateful": rate_block("dismissed_hateful", len(fh2), len(hateful)),
               "nh": rate_block("dismissed_nh", len(fn2), len(nh)),
               "asymmetry": round(asym2, 6), "fisher_p": p2f},
        "C1_reported_alongside": {
            "hateful": rate_block("dismissed_hateful", len(fh1), len(hateful)),
            "nh": rate_block("dismissed_nh", len(fn1), len(nh)),
            "asymmetry": round(rh1 - rn1, 6), "fisher_p": p1f},
        "verdict": p1v,
    }

    # descriptive: flip at the frozen density valley instead of 0
    def flips_kde(zs, vids):
        return sum(1 for v in vids if v in zs and zs[v] > KDE_8B)
    out["P1_descriptive_at_kde_valley"] = {
        "valley": KDE_8B,
        "C1": {"hateful": flips_kde(z1, hateful), "nh": flips_kde(z1, nh)},
        "C2": {"hateful": flips_kde(z2, hateful), "nh": flips_kde(z2, nh)},
    }

    # ---------- P2 ----------
    truncated_h = [v for v in hateful if len((ann[v]["transcript"] or "")) > LIMIT]
    cell_i = [v for v in truncated_h
              if not has_cue((ann[v]["transcript"] or "")[:LIMIT])
              and has_cue((ann[v]["transcript"] or "")[LIMIT:])]

    def cell_stats(name, vids, predicted):
        c1f = [v for v in vids if v in z1 and z1[v] > 0]
        c2f = [v for v in vids if v in z2 and z2[v] > 0]
        return {"cell": name, "n": len(vids), "predicted": predicted,
                "flips_C1": len(c1f), "flips_C2": len(c2f),
                "flips_C2_not_C1": len([v for v in c2f if v not in c1f]),
                "median_z_C0": round(median([z0[v] for v in vids if v in z0]), 4) if vids else None,
                "median_z_C1": round(median([z1[v] for v in vids if v in z1]), 4) if vids else None,
                "median_z_C2": round(median([z2[v] for v in vids if v in z2]), 4) if vids else None}

    ci = cell_stats("(i) truncation route: cue in hidden tail only", cell_i, "flip under C1")
    cii = cell_stats("(ii) ASR route: Step-0 recoverable speech", CELL_II_ASR,
                     "flip under C2 and not under C1")
    ciii = cell_stats("(iii) residual: Step-0 no meaningful speech", CELL_III_NOSPEECH,
                      "flip under neither")
    frag = cell_stats("excluded: Step-0 fragile language", FRAGILE,
                      "excluded from pass/fail")

    # Directional pattern: (i) moves under C1, (ii) moves under C2 beyond C1,
    # (iii) moves under neither.
    base_h1 = rh1
    dir_i = (ci["flips_C1"] / ci["n"] if ci["n"] else 0) > base_h1
    dir_ii = cii["flips_C2"] > cii["flips_C1"]
    dir_iii = (ciii["flips_C1"] == 0 and ciii["flips_C2"] == 0)
    out["P2_three_way_dissociation"] = {
        "bar": "directional pattern holds across all three cells; "
               "failure weakens interpretation, does not kill",
        "cells": [ci, cii, ciii],
        "fragile_excluded": frag,
        "cell_i_ids_n": len(cell_i),
        "directional": {"cell_i_above_C1_base_rate": bool(dir_i),
                        "cell_ii_C2_exceeds_C1": bool(dir_ii),
                        "cell_iii_no_flips": bool(dir_iii)},
        "verdict": "PASS" if (dir_i and dir_ii and dir_iii) else "FAIL",
    }

    # ---------- P3 ----------
    shifts = [z2[v] - z0[v] for v in nh150 if v in z2 and v in z0]
    med_shift = median(shifts)
    p3_ok = rn2 <= 0.08 and med_shift is not None and abs(med_shift) <= 1.0
    out["P3_no_collateral_damage"] = {
        "bar": "C2 flip-rate among dismissed-NH <= 0.08 and |median z shift| <= 1.0 "
               "on the 150-video seed-0 subsample",
        "c2_nh_flip_rate": round(rn2, 6),
        "nh150_n": len(shifts),
        "nh150_median_z_shift_C2_minus_C0": round(med_shift, 4) if med_shift is not None else None,
        "nh150_q1_shift": round(quantile(shifts, 0.25), 4) if shifts else None,
        "nh150_q3_shift": round(quantile(shifts, 0.75), 4) if shifts else None,
        "all_nh_median_z_shift_C2_minus_C0": round(
            median([z2[v] - z0[v] for v in nh if v in z2 and v in z0]), 4),
        "flip_rate_bar_met": bool(rn2 <= 0.08),
        "median_shift_bar_met": bool(med_shift is not None and abs(med_shift) <= 1.0),
        "nh150_frac_shift_positive": round(
            sum(1 for s in shifts if s > 0) / len(shifts), 4) if shifts else None,
        "shift_direction": ("more negative: dismissed NH become more confidently "
                            "normal under C2" if (med_shift or 0) < 0 else
                            "more positive: dismissed NH move toward Yes under C2"),
        "verdict": "PASS" if p3_ok else "FAIL",
    }

    # ---------- P4 ----------
    def half(vids):
        return [v for v in vids if v in asr]
    stats = {}
    for hname, vids in [("dismissed_hateful", hateful), ("dismissed_nh", nh)]:
        rs = [asr[v] for v in half(vids)]
        rej = [v for v in vids if gate.get(v, {}).get("outcome") == "rejected"]
        nofresh = [v for v in vids if gate.get(v, {}).get("outcome") == "no_fresh_pass"]
        capped = [r["video_id"] for r in rs if r.get("hit_duration_cap")]
        uncapped = [r for r in rs if not r.get("hit_duration_cap")]
        # cue recovered: cue in the gated fresh transcript, absent from the old
        # visible 300-char window
        rec = 0
        for v in vids:
            if v not in overrides:
                continue
            old_vis = (ann[v]["transcript"] or "")[:LIMIT]
            if has_cue(overrides[v]) and not has_cue(old_vis):
                rec += 1
        stats[hname] = {
            "n": len(vids), "n_with_fresh_pass": len(rs),
            "median_old_chars": median([r["old_chars"] for r in rs]),
            "median_fresh_raw_chars": median([r["fresh_chars"] for r in rs]),
            "median_gated_chars": median([len(overrides[v]) for v in vids if v in overrides]),
            "median_edit_norm_vs_dataset": round(
                median([r["edit_norm_vs_dataset"] for r in uncapped]), 4) if uncapped else None,
            "frac_edit_norm_gt_0.5": round(
                sum(1 for r in uncapped if r["edit_norm_vs_dataset"] > 0.5) / len(uncapped), 4)
                if uncapped else None,
            "gate_rejected": len(rej),
            "no_fresh_pass": len(nofresh),
            "gate_accepted": len([v for v in vids if v in overrides]),
            "hit_30min_cap": len(capped),
            "cue_recovered_absent_from_old_visible_window": rec,
            "frac_cue_recovered": round(rec / len(vids), 4),
        }
    out["P4_transcription_changed"] = {
        "bar": "descriptive; a null change would make P1 uninterpretable",
        "by_label_half": stats,
        "total_gate_rejected": stats["dismissed_hateful"]["gate_rejected"]
                               + stats["dismissed_nh"]["gate_rejected"],
        "total_no_fresh_pass": stats["dismissed_hateful"]["no_fresh_pass"]
                               + stats["dismissed_nh"]["no_fresh_pass"],
        "total_hit_30min_cap": stats["dismissed_hateful"]["hit_30min_cap"]
                               + stats["dismissed_nh"]["hit_30min_cap"],
        "collapse_shrink_median": round(median(
            [g["collapse_shrink"] for g in gate.values()
             if g.get("collapse_shrink") is not None]), 4),
    }

    # ---------- determinism check ----------
    fully_visible = [v for v in dismissed if len(ann[v]["transcript"] or "") <= LIMIT]
    ident = [v for v in fully_visible if v in z1 and v in z0]
    out["determinism_check_C1_vs_C0"] = {
        "note": "for these the C1 input is byte-identical to C0; any flip is decoder "
                "nondeterminism, not evidence",
        "n_fully_visible_dismissed": len(fully_visible),
        "n_hateful": sum(1 for v in fully_visible if not v.startswith("NH_")),
        "n_compared": len(ident),
        "n_z_exactly_equal": sum(1 for v in ident if z1[v] == z0[v]),
        "max_abs_z_delta": round(max((abs(z1[v] - z0[v]) for v in ident), default=0.0), 6),
        "n_flipped": sum(1 for v in ident if z1[v] > 0),
    }

    # ---------- P5, 2B arm ----------
    hat_all = [v for v in all1283 if v.split("_")[0] in ("EX", "IM")]
    im_all = [v for v in all1283 if v.startswith("IM_")]
    nh_all = [v for v in all1283 if v.startswith("NH_")]
    have = [v for v in all1283 if v in z2b1 and v in z2b0]
    im_h = [v for v in im_all if v in z2b1]
    nh_h = [v for v in nh_all if v in z2b1]
    auc0 = auc(z2b0, [v for v in im_all if v in z2b0], [v for v in nh_all if v in z2b0])
    auc1 = auc(z2b1, im_h, nh_h)
    fn0 = [v for v in hat_all if v in z2b0 and z2b0[v] < KDE_2B]
    fn1 = [v for v in hat_all if v in z2b1 and z2b1[v] < KDE_2B]
    fp0 = [v for v in nh_all if v in z2b0 and z2b0[v] >= KDE_2B]
    fp1 = [v for v in nh_all if v in z2b1 and z2b1[v] >= KDE_2B]
    fn_red = len(fn0) - len(fn1)
    fp_growth = len(fp1) - len(fp0)
    p5_auc_ok = (auc1 - auc0) >= 0.02
    p5_fn_ok = len(fn0) > 0 and (fn_red / len(fn0)) >= 0.15
    p5_fp_ok = fp_growth < 0.5 * fn_red
    p5v = "PASS" if (p5_auc_ok and p5_fn_ok and p5_fp_ok) else "FAIL-KILL"
    out["P5_2B_arm"] = {
        "bar": "AUC(IM vs NH, raw z) improves by >= 0.02 under C1 AND the FN set below "
               "the frozen valley 0.374625 shrinks by >= 15% while FP growth is less "
               "than half the FN reduction",
        "n_scored_C1": len(z2b1), "n_expected": len(all1283),
        "valley": KDE_2B,
        "auc_C0": round(auc0, 4), "auc_C1": round(auc1, 4),
        "auc_delta": round(auc1 - auc0, 4), "auc_bar_met": bool(p5_auc_ok),
        "false_negatives_C0": len(fn0), "false_negatives_C1": len(fn1),
        "fn_reduction": fn_red,
        "fn_reduction_frac": round(fn_red / len(fn0), 4) if fn0 else None,
        "fn_bar_met": bool(p5_fn_ok),
        "false_positives_C0": len(fp0), "false_positives_C1": len(fp1),
        "fp_growth": fp_growth, "fp_bar_met": bool(p5_fp_ok),
        "median_z_C0_IM": round(median([z2b0[v] for v in im_all if v in z2b0]), 4),
        "median_z_C1_IM": round(median([z2b1[v] for v in im_h]), 4),
        "median_z_C0_NH": round(median([z2b0[v] for v in nh_all if v in z2b0]), 4),
        "median_z_C1_NH": round(median([z2b1[v] for v in nh_h]), 4),
        "verdict": p5v,
    }

    out["kill_verdict"] = {
        "8B_arm": ("channel starvation KILLED for the 8B" if p1v.startswith("FAIL")
                   else "channel starvation SURVIVES for the 8B"),
        "8B_basis": f"P1 {p1v}",
        "2B_arm": ("channel starvation KILLED for the 2B" if p5v.startswith("FAIL")
                   else "channel starvation SURVIVES for the 2B"),
        "2B_basis": f"P5 {p5v}",
        "note": "P2, P3 and P4 qualify the interpretation of a P1 pass and do not "
                "by themselves kill the mechanism",
    }

    dest = sys.argv[2]
    with open(dest, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
