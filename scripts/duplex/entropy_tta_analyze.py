"""Analysis for the entropy-adaptation kill test.

Every statistic is computed with the machinery already frozen for the test_c2
measurement: `kde_valley`, `operating_point`, `oracle_f1max`, `auc_np` and
`auc_boot` are imported from `scripts/duplex/crossbench_analyze.py`, not
reimplemented. Labels enter here and nowhere else.

Output: results/entropy_tta/results.json plus a printed clause table.
No video ids and no transcript text reach the output.
"""

import argparse
import json
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(ROOT, "src", "our_method"))

from crossbench_analyze import (  # noqa: E402
    LABEL_MAP, auc_boot, auc_np, dist, kde_valley, load_z, operating_point,
    oracle_f1max, valley_bootstrap,
)
from data_utils import load_annotations, load_clean_split_ids  # noqa: E402

DATASET = "HateMM"
SPLIT = "test"
BASELINE_SCORES = os.path.join(ROOT, "results/testruns/hatemm/judge_8b/scores.jsonl")
OUT_ROOT = os.path.join(ROOT, "results", "entropy_tta")

CLAUSE1_MIN_VALLEY_F1 = 0.75
CLAUSE2_MIN_AUC = 0.90
CLAUSE3_MAX_PLACEBO_SHARE = 0.50


def arm_stats(z, ids, ann, lmap):
    scored = [v for v in ids if v in z]
    pos = [v for v in scored if lmap[ann[v]["label"]] == 1]
    neg = [v for v in scored if lmap[ann[v]["label"]] == 0]
    zs = [z[v] for v in scored]
    valley = kde_valley(zs)
    thr = valley.get("value")
    out = {
        "n_scored": len(scored), "n_hateful": len(pos), "n_normal": len(neg),
        "auc": round(auc_np(np.array([z[v] for v in pos]),
                            np.array([z[v] for v in neg])), 6),
        "auc_boot95": [round(x, 6) for x in auc_boot(
            np.array([z[v] for v in pos]), np.array([z[v] for v in neg]))],
        "z_distribution_all": dist(zs),
        "z_distribution_hateful": dist([z[v] for v in pos]),
        "z_distribution_normal": dist([z[v] for v in neg]),
        "valley": valley,
        "valley_stability_bootstrap": valley_bootstrap(zs),
    }
    out["operating_point_valley"] = (
        operating_point(z, pos, neg, thr) if thr is not None else None)
    out["operating_point_zero"] = operating_point(z, pos, neg, 0.0)
    out["operating_point_oracle_f1max"] = oracle_f1max(z, pos, neg)
    return out, pos, neg


def movement(base_z, new_z, cohort, thr_base, thr_new):
    """How a cohort of videos moved, in z and across each arm's own threshold."""
    d = [new_z[v] - base_z[v] for v in cohort]
    crossed = sum(1 for v in cohort if (base_z[v] >= thr_base) != (new_z[v] >= thr_new))
    return {
        "n": len(cohort),
        "delta_z": dist(d),
        "n_changed_side_of_own_threshold": crossed,
        "frac_changed_side": round(crossed / len(cohort), 4) if cohort else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir", default=os.path.join(OUT_ROOT, "real"))
    ap.add_argument("--placebo-dir", default=os.path.join(OUT_ROOT, "placebo"))
    ap.add_argument("--selfcheck-dir", default=os.path.join(OUT_ROOT, "frozen_selfcheck"))
    ap.add_argument("--extra-arm", action="append", default=[],
                    help="name=path/to/scores.jsonl, reported descriptively")
    ap.add_argument("--out", default=os.path.join(OUT_ROOT, "results.json"))
    args = ap.parse_args()

    ann = load_annotations(DATASET)
    lmap = LABEL_MAP[DATASET]
    seen, ids = set(), []
    for v in load_clean_split_ids(DATASET, SPLIT):
        if v not in seen:
            seen.add(v)
            ids.append(v)

    arms = {"baseline_frozen_vllm_free_reference": BASELINE_SCORES}
    for name, d in (("frozen_selfcheck", args.selfcheck_dir),
                    ("real", args.real_dir), ("placebo", args.placebo_dir)):
        p = os.path.join(d, "scores.jsonl")
        if os.path.exists(p):
            arms[name] = p
    for spec in args.extra_arm:
        name, _, p = spec.partition("=")
        if os.path.exists(p):
            arms[name] = p

    zs, stats, cohorts = {}, {}, {}
    for name, p in arms.items():
        z = load_z(p)
        if len(z) != len(ids):
            print(f"WARNING: {name} has {len(z)}/{len(ids)} scored videos")
        zs[name] = z
        stats[name], pos, neg = arm_stats(z, ids, ann, lmap)
        cohorts[name] = (pos, neg)

    base = "baseline_frozen_vllm_free_reference"
    base_thr = stats[base]["valley"]["value"]
    base_f1 = stats[base]["operating_point_valley"]["macro_f1"]
    pos, neg = cohorts[base]
    base_fp = [v for v in neg if zs[base][v] >= base_thr]
    base_fn = [v for v in pos if zs[base][v] < base_thr]

    # Self-check: this run's own frozen-model rescoring must reproduce the
    # committed baseline. A mismatch means the adaptation code path is not the
    # same instrument and the whole comparison is void.
    selfcheck = None
    if "frozen_selfcheck" in stats:
        s = stats["frozen_selfcheck"]
        d_auc = abs(s["auc"] - stats[base]["auc"])
        d_f1 = abs(s["operating_point_valley"]["macro_f1"] - base_f1)
        d_z_max = max(abs(zs["frozen_selfcheck"][v] - zs[base][v]) for v in ids)
        selfcheck = {
            "role": "the adaptation script's own frozen-model rescoring, "
                    "compared against the committed test_c2 baseline",
            "auc_selfcheck": s["auc"], "auc_baseline": stats[base]["auc"],
            "abs_delta_auc": round(d_auc, 6),
            "valley_macro_f1_selfcheck": s["operating_point_valley"]["macro_f1"],
            "valley_macro_f1_baseline": base_f1,
            "abs_delta_valley_macro_f1": round(d_f1, 6),
            "max_abs_delta_z_per_video": round(d_z_max, 6),
            "tolerance": 0.001,
            "passes": bool(d_auc <= 0.001 and d_f1 <= 0.001),
        }

    for name in stats:
        if name == base:
            continue
        thr = stats[name]["valley"]["value"]
        f1 = (stats[name]["operating_point_valley"]["macro_f1"]
              if thr is not None else None)
        stats[name]["gain_over_baseline"] = {
            "valley_macro_f1": round(f1 - base_f1, 6) if f1 is not None else None,
            "auc": round(stats[name]["auc"] - stats[base]["auc"], 6),
            "relative_trough_depth": round(
                stats[name]["valley"].get("relative_trough_depth", float("nan"))
                - stats[base]["valley"]["relative_trough_depth"], 6),
        }
        if thr is not None:
            stats[name]["cohort_movement"] = {
                "baseline_false_positives": movement(
                    zs[base], zs[name], base_fp, base_thr, thr),
                "baseline_false_negatives": movement(
                    zs[base], zs[name], base_fn, base_thr, thr),
                "all_hateful": movement(zs[base], zs[name], pos, base_thr, thr),
                "all_normal": movement(zs[base], zs[name], neg, base_thr, thr),
            }

    real = stats.get("real")
    plac = stats.get("placebo")
    verdicts = {}
    if real is not None:
        r_f1 = real["operating_point_valley"]["macro_f1"] \
            if real["operating_point_valley"] else None
        verdicts["clause1_operating_point_recovers"] = {
            "rule": f"adapted valley macro-F1 >= {CLAUSE1_MIN_VALLEY_F1}",
            "value": r_f1, "baseline": base_f1,
            "passes": bool(r_f1 is not None and r_f1 >= CLAUSE1_MIN_VALLEY_F1),
        }
        verdicts["clause2_ranking_preserved"] = {
            "rule": f"adapted AUC >= {CLAUSE2_MIN_AUC}",
            "value": real["auc"], "baseline": stats[base]["auc"],
            "passes": bool(real["auc"] >= CLAUSE2_MIN_AUC),
        }
        if plac is not None:
            r_gain = (r_f1 - base_f1) if r_f1 is not None else None
            p_f1 = plac["operating_point_valley"]["macro_f1"] \
                if plac["operating_point_valley"] else None
            p_gain = (p_f1 - base_f1) if p_f1 is not None else None
            share = (p_gain / r_gain) if (r_gain and r_gain > 0) else None
            verdicts["clause3_evidence_driven"] = {
                "rule": f"placebo valley-F1 gain < {CLAUSE3_MAX_PLACEBO_SHARE} "
                        f"x real arm's gain",
                "real_gain": round(r_gain, 6) if r_gain is not None else None,
                "placebo_gain": round(p_gain, 6) if p_gain is not None else None,
                "placebo_share_of_real_gain": round(share, 4) if share is not None else None,
                "note": None if (r_gain and r_gain > 0) else
                        "the real arm did not gain, so the ratio is undefined; "
                        "clause 3 cannot be satisfied by a failed real arm",
                "passes": bool(share is not None and share < CLAUSE3_MAX_PLACEBO_SHARE),
            }

    verdict = None
    if len(verdicts) == 3:
        verdict = "SURVIVES" if all(v["passes"] for v in verdicts.values()) else "DEAD"

    traj = {}
    for name, d in (("real", args.real_dir), ("placebo", args.placebo_dir)):
        p = os.path.join(d, "entropy_trajectory.jsonl")
        if not os.path.exists(p):
            continue
        h = [json.loads(l)["entropy_nats"] for l in open(p) if l.strip()]
        n = len(h)
        traj[name] = {
            "n_steps": n,
            "mean_entropy_nats": round(float(np.mean(h)), 6),
            "mean_first_quarter": round(float(np.mean(h[:n // 4])), 6),
            "mean_last_quarter": round(float(np.mean(h[-(n // 4):])), 6),
            "median": round(float(np.median(h)), 6),
            "max": round(float(np.max(h)), 6),
            "n_steps_above_0.1_nats": int(sum(1 for x in h if x > 0.1)),
        }

    doc = {
        "title": "Entropy-adaptation kill test on HateMM test (215 videos)",
        "prereg": "docs/duplex/PREREG_entropy_tta_killtest.md",
        "verdict": verdict,
        "clause_verdicts": verdicts,
        "frozen_model_selfcheck": selfcheck,
        "entropy_trajectory": traj,
        "arms": stats,
        "notes": {
            "labels": "labels enter this analysis only; no training step saw one",
            "machinery": "kde_valley, operating_point, oracle_f1max, auc_np and "
                         "auc_boot are imported unmodified from "
                         "scripts/duplex/crossbench_analyze.py",
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=2)
    print(json.dumps({"verdict": verdict, "clauses": verdicts,
                      "selfcheck": selfcheck}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
