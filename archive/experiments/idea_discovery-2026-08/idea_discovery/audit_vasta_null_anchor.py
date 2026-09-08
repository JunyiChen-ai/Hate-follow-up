#!/usr/bin/env python3
"""Derive VASTA's transcript veto from a label-free null-evidence anchor."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def load(path: Path):
    return {(row["dataset"], row["video_id"]): row
            for row in map(json.loads, path.open())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = load(args.manifest)
    text = load(args.text)
    gate_rows = {}
    for row in map(json.loads, args.gates.open()):
        if row["method"] == "extent_gate_or":
            gate_rows[(row["dataset"], row["video_id"])] = row

    keys = sorted(set(manifest) & set(text) & set(gate_rows))
    null_scores = [float(text[key]["log_odds"]) for key in keys
                   if not str(manifest[key].get("transcript", "")).strip()]
    if not null_scores:
        raise RuntimeError("no empty-transcript null controls found")
    anchor = float(np.median(null_scores))

    disagreement = []
    for key in keys:
        evidence = gate_rows[key]["modality_evidence"]
        if bool(evidence["a08_nonempty"]) == bool(evidence["a12_nonempty"]):
            continue
        score = float(text[key]["log_odds"])
        disagreement.append({
            "dataset": key[0],
            "video_id": key[1],
            "text_log_odds": score,
            "veto": score < anchor,
        })

    folds = {}
    datasets = sorted({key[0] for key in keys})
    for held in datasets:
        train_null = [float(text[key]["log_odds"]) for key in keys
                      if key[0] != held and
                      not str(manifest[key].get("transcript", "")).strip()]
        fold_anchor = float(np.median(train_null))
        folds[held] = {
            "calibration_datasets": [dataset for dataset in datasets
                                     if dataset != held],
            "n_null_controls": len(train_null),
            "null_anchor": fold_anchor,
            "held_disagreement_veto_count": sum(
                row["dataset"] == held and row["text_log_odds"] < fold_anchor
                for row in disagreement),
        }

    output = {
        "protocol": "label-free null-evidence calibration",
        "definition": (
            "Text can veto a visual disagreement iff its Yes-vs-No log-odds "
            "is strictly lower than the same model's empty-transcript score."),
        "n_videos": len(keys),
        "n_null_controls": len(null_scores),
        "null_anchor": anchor,
        "null_score_min": float(min(null_scores)),
        "null_score_max": float(max(null_scores)),
        "null_score_counts": {str(key): value
                              for key, value in Counter(null_scores).items()},
        "n_visual_disagreements": len(disagreement),
        "n_vetoes": sum(row["veto"] for row in disagreement),
        "vetoes_by_dataset": dict(Counter(
            row["dataset"] for row in disagreement if row["veto"])),
        "leave_one_dataset_out_anchor": folds,
        "detail": disagreement,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: output[key] for key in (
        "n_null_controls", "null_anchor", "null_score_min", "null_score_max",
        "n_visual_disagreements", "n_vetoes", "vetoes_by_dataset")},
        indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
