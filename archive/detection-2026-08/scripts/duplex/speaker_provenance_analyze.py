#!/usr/bin/env python3
"""Evaluate the frozen speaker-provenance P/S/R probe."""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SEED = 20260808
N_BOOT = 2000


def read_jsonl(path):
    with open(path) as f:
        return {row["video_id"]: float(row["z"]) for row in map(json.loads, f)}


def auc(pos, neg, scores):
    """Probability that a positive outranks a negative, with half-credit ties."""
    p = np.asarray([scores[i] for i in pos], dtype=float)
    n = np.asarray([scores[i] for i in neg], dtype=float)
    comparisons = p[:, None] - n[None, :]
    return float((np.sum(comparisons > 0) + .5 * np.sum(comparisons == 0)) /
                 comparisons.size)


def ci(values):
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def main():
    cohort = json.load(open(ROOT / "results/temporal_attribution/cohorts.json"))
    cells = cohort["cohorts"]
    scores = {"F": {k: float(v) for k, v in cohort["joint_z"].items()}}
    for arm in "PSR":
        scores[arm] = read_jsonl(
            ROOT / f"results/speaker_provenance/judge_{arm}/scores.jsonl")

    fp, tp, tn = (cells[k] for k in ("fp", "tp", "tn"))
    required = set(fp + tp + tn)
    missing = {a: sorted(required - set(s)) for a, s in scores.items()}
    if any(missing.values()):
        raise RuntimeError(f"Missing scores: {missing}")

    valley = float(json.load(open(ROOT / "results/stance_gate/cohorts.json"))
                   ["arms"]["ihv_train"]["valley"])
    auc_tf = {a: auc(tp, fp, s) for a, s in scores.items()}
    auc_nf = {a: auc(tn, fp, s) for a, s in scores.items()}
    delta_fp = np.array([scores["P"][i] - scores["F"][i] for i in fp])
    delta_tp = np.array([scores["P"][i] - scores["F"][i] for i in tp])
    flip_fp = np.array([scores["F"][i] >= valley and scores["P"][i] < valley for i in fp])
    flip_tp = np.array([scores["F"][i] >= valley and scores["P"][i] < valley for i in tp])
    toward_indifference = abs(auc_nf["F"] - .5) - abs(auc_nf["P"] - .5)

    rng = np.random.default_rng(SEED)
    boot = {k: [] for k in ["fp_median_delta", "tp_median_delta", "P_minus_F_auc",
                             "P_minus_S_auc", "P_minus_R_auc", "fp_flip_rate",
                             "tp_flip_rate", "tn_fp_toward_indifference"]}
    for _ in range(N_BOOT):
        bfp = rng.choice(fp, len(fp), replace=True).tolist()
        btp = rng.choice(tp, len(tp), replace=True).tolist()
        btn = rng.choice(tn, len(tn), replace=True).tolist()
        b_auc_tf = {a: auc(btp, bfp, s) for a, s in scores.items()}
        b_auc_nf = {a: auc(btn, bfp, s) for a, s in scores.items()}
        boot["fp_median_delta"].append(np.median([scores["P"][i]-scores["F"][i] for i in bfp]))
        boot["tp_median_delta"].append(np.median([scores["P"][i]-scores["F"][i] for i in btp]))
        for other in "FSR":
            boot[f"P_minus_{other}_auc"].append(b_auc_tf["P"] - b_auc_tf[other])
        boot["fp_flip_rate"].append(np.mean([scores["F"][i] >= valley and scores["P"][i] < valley for i in bfp]))
        boot["tp_flip_rate"].append(np.mean([scores["F"][i] >= valley and scores["P"][i] < valley for i in btp]))
        boot["tn_fp_toward_indifference"].append(
            abs(b_auc_nf["F"]-.5) - abs(b_auc_nf["P"]-.5))

    point = {
        "fp_median_delta": float(np.median(delta_fp)),
        "tp_median_delta": float(np.median(delta_tp)),
        "P_minus_F_auc": auc_tf["P"] - auc_tf["F"],
        "P_minus_S_auc": auc_tf["P"] - auc_tf["S"],
        "P_minus_R_auc": auc_tf["P"] - auc_tf["R"],
        "fp_flip_rate": float(flip_fp.mean()),
        "tp_flip_rate": float(flip_tp.mean()),
        "tn_fp_toward_indifference": float(toward_indifference),
    }
    clauses = {
        "1_fp_median_delta_le_-1": point["fp_median_delta"] <= -1,
        "2_tp_median_delta_ge_-0.25": point["tp_median_delta"] >= -.25,
        "3_P_minus_F_auc_ge_.05": point["P_minus_F_auc"] >= .05,
        "4_P_minus_S_auc_ge_.03": point["P_minus_S_auc"] >= .03,
        "5_P_minus_R_auc_ge_.03": point["P_minus_R_auc"] >= .03,
        "6_fp_flip_ge_.30_and_tp_flip_le_.10": point["fp_flip_rate"] >= .30 and point["tp_flip_rate"] <= .10,
        "7_tn_fp_toward_indifference_ge_.10": point["tn_fp_toward_indifference"] >= .10,
    }
    out = {
        "seed": SEED, "n_bootstrap": N_BOOT, "frozen_valley": valley,
        "n": {k: len(v) for k, v in cells.items()},
        "auc_tp_vs_fp": auc_tf, "auc_tn_vs_fp": auc_nf,
        "point": point, "bootstrap_95ci": {k: ci(v) for k, v in boot.items()},
        "clauses": clauses, "pass": all(clauses.values()),
    }
    out_path = ROOT / "results/speaker_provenance/results.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
