"""Analyze temporal-attribution pilot against its frozen bars."""

import json
import os
import random
import statistics

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUN = os.path.join(ROOT, "results", "temporal_attribution")
ARMS = ("unaligned", "aligned", "shuffled")
SEED = 20260808


def auc(scores, pos, neg):
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if scores[p] > scores[n] else 0.5 if scores[p] == scores[n] else 0.0
    return wins / (len(pos) * len(neg))


def q(xs, p):
    ys = sorted(xs)
    return ys[min(len(ys)-1, int(p * len(ys)))]


def boot_delta(a, b, pos, neg, n=1000):
    rng = random.Random(SEED)
    vals = []
    for _ in range(n):
        pp = [rng.choice(pos) for _ in pos]
        nn = [rng.choice(neg) for _ in neg]
        vals.append(auc(a, pp, nn) - auc(b, pp, nn))
    return [q(vals, .025), q(vals, .975)]


def cell_stats(scores, ids):
    xs = [scores[v] for v in ids]
    return {"n": len(xs), "mean": sum(xs)/len(xs), "median": statistics.median(xs),
            "q25": q(xs, .25), "q75": q(xs, .75)}


def main():
    cohort = json.load(open(os.path.join(RUN, "cohorts.json")))
    by = {a: {} for a in ARMS}
    for line in open(os.path.join(RUN, "probe_scores.jsonl")):
        r = json.loads(line); by[r["arm"]][r["video_id"]] = float(r["z"])
    allids = [v for vs in cohort["cohorts"].values() for v in vs]
    for arm in ARMS:
        missing = [v for v in allids if v not in by[arm]]
        if missing: raise SystemExit(f"{arm}: {len(missing)} missing")
    fp, tp, tn = (cohort["cohorts"][x] for x in ("fp", "tp", "tn"))
    aucs = {a: auc(by[a], tp, fp) for a in ARMS}
    delta_fp = {v: by["aligned"][v] - by["unaligned"][v] for v in fp}
    delta_tp = {v: by["aligned"][v] - by["unaligned"][v] for v in tp}
    clauses = {
        "auc_aligned_minus_unaligned_ge_0.05": {
            "value": aucs["aligned"] - aucs["unaligned"], "bar": .05,
            "pass": aucs["aligned"] - aucs["unaligned"] >= .05,
            "boot95": boot_delta(by["aligned"], by["unaligned"], tp, fp)},
        "auc_aligned_minus_shuffled_ge_0.05": {
            "value": aucs["aligned"] - aucs["shuffled"], "bar": .05,
            "pass": aucs["aligned"] - aucs["shuffled"] >= .05,
            "boot95": boot_delta(by["aligned"], by["shuffled"], tp, fp)},
        "fp_median_delta_le_minus_0.50": {
            "value": statistics.median(delta_fp.values()), "bar": -.5,
            "pass": statistics.median(delta_fp.values()) <= -.5},
        "tp_median_delta_ge_minus_0.25": {
            "value": statistics.median(delta_tp.values()), "bar": -.25,
            "pass": statistics.median(delta_tp.values()) >= -.25},
    }
    marked = cohort["marker_positive"]
    subgroup = {}
    for name, flag in (("marker_positive", True), ("marker_negative", False)):
        pp = [v for v in tp if marked[v] is flag]
        ff = [v for v in fp if marked[v] is flag]
        subgroup[name] = {
            "n_tp": len(pp), "n_fp": len(ff),
            "auc_unaligned": auc(by["unaligned"], pp, ff),
            "auc_aligned": auc(by["aligned"], pp, ff),
            "delta_auc": auc(by["aligned"], pp, ff) - auc(by["unaligned"], pp, ff),
            "fp_median_delta": statistics.median(
                by["aligned"][v] - by["unaligned"][v] for v in ff),
        }
    direction = (subgroup["marker_positive"]["delta_auc"] > 0 and
                 subgroup["marker_positive"]["fp_median_delta"] <
                 subgroup["marker_negative"]["fp_median_delta"])
    report = {
        "test": "temporal attribution pilot", "date": "2026-08-08",
        "model": "Qwen3-VL-8B-Instruct", "n": len(allids),
        "preregistration": "docs/duplex/PREREG_temporal_attribution_pilot.md",
        "verdict": "PASS" if all(c["pass"] for c in clauses.values()) and direction else "FAIL",
        "headline": {"auc_tp_vs_fp": aucs,
                     "aligned_minus_unaligned": aucs["aligned"]-aucs["unaligned"],
                     "aligned_minus_shuffled": aucs["aligned"]-aucs["shuffled"]},
        "clauses": clauses, "marker_directionally_consistent": direction,
        "subgroups": subgroup,
        "z_by_arm_cell": {a: {c: cell_stats(by[a], cohort["cohorts"][c])
                                    for c in ("fp", "tp", "tn")} for a in ARMS},
        "content_policy": "statistics only; no video ids or transcript text",
    }
    path = os.path.join(ROOT, "docs/duplex/reports/temporal_attribution_pilot.json")
    with open(path, "w") as f: json.dump(report, f, indent=2)
    print("VERDICT", report["verdict"])
    print(json.dumps(report["headline"], indent=2))
    print(json.dumps(clauses, indent=2))
    print("marker directional", direction)
    print(json.dumps(subgroup, indent=2))
    print("Wrote", path)


if __name__ == "__main__":
    main()
