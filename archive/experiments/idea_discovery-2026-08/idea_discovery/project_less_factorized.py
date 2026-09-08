#!/usr/bin/env python3
"""Factorized LESS readout: semantic posterior plus consensus geometry.

Dense hateful-state ranking and event geometry are distinct prediction objects.
This projector keeps the frozen LESS posterior for frame localization and the
frozen CCA consensus/closure state for interval production.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.label_free_adapt.schema import Interval, Prediction, append_jsonl


def load(path: Path):
    output = {}
    for row in map(json.loads, path.open()):
        output[(row["method"], row["dataset"], row["video_id"])] = row
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--less", type=Path, required=True)
    parser.add_argument("--cca", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    less = load(args.less)
    cca_rows = list(map(json.loads, args.cca.open()))
    cca = {(row["dataset"], row["video_id"]): row for row in cca_rows}
    methods = sorted({key[0] for key in less})
    summary = {}
    for source_method in methods:
        method = source_method.replace("less_3v_", "fact_less_3v_").replace("_v1", "_v1")
        count = 0
        for (candidate_method, dataset, video_id), row in less.items():
            if candidate_method != source_method:
                continue
            geometry = cca[(dataset, video_id)]
            intervals = [Interval(float(value[0]), float(value[1]),
                                  float(value[2]) if len(value) > 2 else 1.0)
                         for value in geometry["intervals"]]
            count += bool(intervals)
            append_jsonl(args.out, Prediction(
                method, dataset, video_id, float(row["duration"]),
                score_curve=[float(value) for value in row["score_curve"]],
                intervals=intervals, calls=0,
                modality_evidence={"semantic_state_source": source_method,
                                   "semantic_state": row.get("modality_evidence", {}),
                                   "geometry_source": "cca_v1",
                                   "geometry": geometry.get("modality_evidence", {})},
                raw={"gt_access": False,
                     "factorization": {"frame_curve": "crossfit_multiview_latent_posterior",
                                       "event_intervals": "consensus_gated_proposal_closure"},
                     "single_threshold_coupling": False}))
        summary[method] = {"n": sum(key[0] == source_method for key in less),
                           "nonempty": count}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
