#!/usr/bin/env python3
"""Check canonical metric agreement and summarize the rank-preserving control."""
import csv
import json
from pathlib import Path

import numpy as np

from analyze import ROOT, OUT, DATASETS, FINALS, read, auc, bootstrap, dump


def metrics(p):
    return {r["dataset"]: r for r in json.load((ROOT / p / "metrics.json").open())["per_dataset"]}


def main():
    out = OUT / "analysis"
    summary = json.load((out / "summary.json").open())
    rows = list(csv.DictReader((out / "per_video.csv").open()))
    metric_names = ("frame_ROC_AUC", "frame_PR_AUC", "within_video_macro_ROC_AUC")
    comparisons = {}
    for ds in DATASETS:
        fam = "DeHate" if ds == "DeHate" else "main"
        ma, mb = [metrics(p)[ds] for p in FINALS[fam]]
        observed = summary[ds]["all"]["final_delta"]["mean"]
        assert abs(observed - (ma[metric_names[2]] - mb[metric_names[2]])) < 1e-12
        comparisons[ds] = {"with": {k: ma[k] for k in metric_names}, "without": {k: mb[k] for k in metric_names},
            "difference": {k: ma[k] - mb[k] for k in metric_names},
            "within_difference_bootstrap": summary[ds]["all"]["final_delta"],
            "metric_files": [p + "/metrics.json" for p in FINALS[fam]]}
    current = read(FINALS["main"][0])
    previous = read("runs/20260926_twolevel/robust/q3vl-8b_r6")
    assert current.keys() == previous.keys()
    max_error = max(float(np.max(np.abs(np.asarray(current[k]["score_curve"]) - previous[k]["score_curve"]))) for k in current)
    assert max_error < 1e-10, max_error
    shifted = read("runs/20261002_verdict_analysis/main_shift_only")
    shmet = metrics("runs/20261002_verdict_analysis/main_shift_only")
    for ds in DATASETS[:2]:
        g = np.load(ROOT / f"data/gt_4fps/{ds}.npz", allow_pickle=True)
        gt = {str(v): np.asarray(g["y4"][i], np.int8) for i, v in enumerate(g["video_ids"]) if str(g["split"][i]) == "test"}
        no = read(FINALS["main"][1])
        dsrows = [r for r in rows if r["dataset"] == ds and r["final_delta"]]
        shift_deltas, residuals = [], []
        for r in dsrows:
            vid = r["video_id"]; k = ds, vid
            shauc = auc(gt[vid], shifted[k]["score_curve"])
            shift_deltas.append(shauc - float(r["final_no"]))
            residuals.append(float(r["final_full"]) - shauc)
        comparisons[ds]["common_shift_control"] = {
            "metrics": {k: shmet[ds][k] for k in metric_names},
            "shift_minus_without": bootstrap(shift_deltas),
            "with_minus_shift": bootstrap(residuals),
            "raw_modality_and_max_rankings_preserved": True,
            "caution": "Paired-output reconstruction; not deployable and not a mediation estimate.",
            "metric_file": "runs/20261002_verdict_analysis/main_shift_only/metrics.json"}
    dump(out / "comparisons.json", comparisons)
    dump(out / "verification.json", {"canonical_macro_metrics_match_per_video_means": True,
        "current_full_matches_existing_r6_on_identical_reader_cache": True,
        "max_absolute_score_difference": max_error, "n_full_scores_checked": len(current),
        "rank_control_construction": "shift_control.py verifies ordering and ties before any GT access",
        "scope": "no inference code numerical changes; GT used only for diagnostics"})
    print(json.dumps(comparisons, indent=2))
    print("VERIFICATION_DONE")


if __name__ == "__main__":
    main()
