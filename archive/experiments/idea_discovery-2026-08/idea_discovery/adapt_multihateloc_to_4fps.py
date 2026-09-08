#!/usr/bin/env python3
"""Protocol adapter for the local MultiHateLoc reimplementation.

This is an evaluation-only adapter.  It uses the target timeline length from
the evaluation manifest (not label values) to interpolate native score curves
onto the shared 4 fps grid.  It never invents predictions for absent IDs, so
coverage remains auditable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.label_free_adapt.evaluate import binary_intervals


MAP = {"hatemm": "HateMM", "hateclipseg": "HateClipSeg",
       "mhclip_en": "MHC", "mhclip_zh": "MHC_zh"}
SCORES = {"fused": "score_fused", "dms": "score_dms"}


def resize(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) == n:
        return x.copy()
    return np.interp(np.linspace(0.0, 1.0, n),
                     np.linspace(0.0, 1.0, len(x)), x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--gt-dir", type=Path, required=True,
                    help="Used only for test IDs and target timeline lengths")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    audit = {}
    with args.out.open("w") as output:
        for source_name, dataset in MAP.items():
            source = {r["video_id"]: r for r in
                      (json.loads(line) for line in
                       (args.source_root / source_name / "scores.jsonl").open())}
            gt = np.load(args.gt_dir / f"{dataset}.npz", allow_pickle=True)
            lengths = {str(v): len(gt["y4"][i]) for i, v in enumerate(gt["video_ids"])
                       if str(gt["split"][i]) == "test"}
            overlap = sorted(set(source) & set(lengths))
            audit[dataset] = {"source": len(source), "target": len(lengths),
                              "overlap": len(overlap),
                              "coverage": len(overlap) / len(lengths)}
            for video_id in overlap:
                row = source[video_id]
                n = lengths[video_id]
                union = resize(np.asarray(row["score_union"], float), n) >= 0.5
                intervals = [list(x) + [1.0] for x in binary_intervals(union, 4.0)]
                for suffix, field in SCORES.items():
                    scores = resize(np.asarray(row[field], float), n)
                    output.write(json.dumps({
                        "schema_version": "1.0",
                        "method": f"multihateloc_reimpl_{suffix}_4fps",
                        "dataset": dataset, "video_id": video_id,
                        "duration": n / 4.0, "native_rate": 4.0,
                        "score_curve": scores.tolist(), "intervals": intervals,
                        "modality_evidence": {},
                        "raw": {"gt_access": False,
                                "evaluation_manifest_access": "timeline_length_only",
                                "source_rate": "native_local_reimplementation",
                                "interpolation": "normalized_linear"},
                        "calls": 0, "seed": 0, "error": None,
                    }, separators=(",", ":")) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
