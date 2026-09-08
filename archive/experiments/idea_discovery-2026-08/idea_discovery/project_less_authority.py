#!/usr/bin/env python3
"""Authority-controlled hierarchical LESS.

A target domain accepts the multimodal posterior only when opposite-fold
truth-discovery estimates assign the visual anchor reliability above the
predeclared random boundary (0.5).  Otherwise its frozen T3AL state is retained.
No dataset name or evaluation metric appears in the decision rule.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.label_free_adapt.schema import Interval, Prediction, append_jsonl


def rows(path: Path, method=None):
    output = {}
    for row in map(json.loads, path.open()):
        if method is None or row.get("method") == method:
            output[(row["dataset"], row["video_id"])] = row
    return output


def authorities(path: Path) -> dict[str, dict]:
    audit = json.loads(path.read_text())
    values = defaultdict(list)
    for name, model in audit["models"].items():
        if not name.startswith("less_3v_full_a10_dataset_v2:"):
            continue
        dataset = name.split(":")[1]
        values[dataset].append(float(model["reliability"]["visual"]))
    return {dataset: {"fold_reliabilities": folds,
                      "mean_visual_reliability": sum(folds) / len(folds),
                      "multiview_update_authorized": sum(folds) / len(folds) > 0.5}
            for dataset, folds in values.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hierarchical", type=Path, required=True)
    parser.add_argument("--a10", type=Path, required=True)
    parser.add_argument("--cca", type=Path, required=True)
    parser.add_argument("--domain-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    hierarchical = rows(args.hierarchical, "fact_less_3v_hierarchical_v3")
    a10 = rows(args.a10)
    cca = rows(args.cca)
    authority = authorities(args.domain_audit)
    keys = sorted(set(hierarchical) & set(a10) & set(cca))
    source_counts = defaultdict(int)
    for key in keys:
        authorized = authority[key[0]]["multiview_update_authorized"]
        source = hierarchical[key] if authorized else a10[key]
        source_counts["hierarchical" if authorized else "visual_anchor"] += 1
        geometry = cca[key]
        intervals = [Interval(float(value[0]), float(value[1]),
                              float(value[2]) if len(value) > 2 else 1.0)
                     for value in geometry["intervals"]]
        append_jsonl(args.out, Prediction(
            "fact_less_3v_authority_v4", key[0], key[1],
            float(source["duration"]), score_curve=[float(x) for x in source["score_curve"]],
            intervals=intervals, calls=0,
            modality_evidence={"domain_authority": authority[key[0]],
                               "semantic_state_source": "hierarchical_less" if authorized else "t3al_anchor",
                               "geometry_source": "cca_v1"},
            raw={"gt_access": False, "authority_threshold": 0.5,
                 "threshold_meaning": "better_than_random_visual_anchor_recovery",
                 "dataset_name_in_rule": False,
                 "factorized_interval_readout": "consensus_gated_proposal_closure"}))
    print(json.dumps({"n": len(keys), "authority": authority,
                      "source_counts": source_counts}, indent=2))


if __name__ == "__main__":
    main()
