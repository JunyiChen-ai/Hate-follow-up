#!/usr/bin/env python3
"""Label-blind citation controls for saved MELT phase predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.idea_discovery.run_melt import dense_bins, lifecycle_decode
from scripts.label_free_adapt.schema import Interval, Prediction, append_jsonl


def controlled_anchors(mode: str, cited: list[int], video_id: str) -> list[int]:
    if mode == "real":
        return cited
    if mode == "none":
        return []
    if mode == "shift":
        return sorted({(x + 8) % 16 for x in cited})
    if mode == "random":
        seed = int(hashlib.sha256(video_id.encode()).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        return sorted(rng.choice(16, size=min(len(cited), 16), replace=False).tolist())
    raise ValueError(mode)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source-method", default="melt_adaptive")
    args = ap.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    for line in args.input.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("method") != args.source_method:
            continue
        for mode in ("real", "none", "random", "shift"):
            method = f"melt_citation_{mode}"
            if row.get("error"):
                append_jsonl(args.out, Prediction(
                    method, row["dataset"], row["video_id"], row["duration"],
                    error=row["error"], raw={"source": str(args.input)}))
                continue
            evidence = row["modality_evidence"]
            phase = np.asarray(evidence["phase_scores"], float)
            cited = [int(x) for x in evidence.get("evidence_bins", [])]
            anchors = controlled_anchors(mode, cited, row["video_id"])
            intervals, event_prob, meta = lifecycle_decode(phase, row["duration"], anchors)
            original = np.asarray(row["score_curve"], float)
            curve = (dense_bins(event_prob, len(original))
                     if evidence.get("ranking_source") == "phase_field" else original)
            append_jsonl(args.out, Prediction(
                method, row["dataset"], row["video_id"], row["duration"],
                score_curve=curve.tolist(), intervals=intervals, calls=0,
                modality_evidence={"original_citations": cited, "used_anchors": anchors,
                                   "lifecycle": meta},
                raw={"source": str(args.input), "citation_mode": mode}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
