#!/usr/bin/env python
"""Convert released LAVAD JSON artefacts to the shared scores.jsonl format."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASELINES = HERE.parent
sys.path.insert(0, str(BASELINES))
from hate_common import data as hdata  # noqa: E402


def numeric_keys(obj):
    return sorted(((int(k), v) for k, v in obj.items()), key=lambda x: x[0])


def weighted_scores(score_obj, similarity_obj, n_neighbors):
    out = []
    for idx, value in numeric_keys(score_obj):
        if isinstance(value, dict):
            vals = np.asarray([float(value[str(k)]) for k in range(n_neighbors)])
            sims = np.asarray([float(similarity_obj[str(idx)][str(k)])
                               for k in range(n_neighbors)])
            weights = np.exp(sims - sims.max())
            value = float(np.dot(vals, weights / weights.sum()))
        out.append(float(value))
    return np.asarray(out, dtype=float)


def align(scores, n_gold, video_id):
    if len(scores) != n_gold:
        raise ValueError("%s: %d LAVAD scores but %d gold frames; run with "
                         "the prepared 1 fps input and --frame_interval 1"
                         % (video_id, len(scores), n_gold))
    if not np.isfinite(scores).all():
        raise ValueError("%s: non-finite LAVAD score" % video_id)
    return scores


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=hdata.CORPORA)
    ap.add_argument("--raw-scores", required=True)
    ap.add_argument("--refined-scores")
    ap.add_argument("--similarities")
    ap.add_argument("--num-neighbors", type=int, default=10)
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    if bool(args.refined_scores) != bool(args.similarities):
        ap.error("--refined-scores and --similarities must be supplied together")
    gt = hdata.gt_arrays(args.corpus, "test")
    records = []
    for vid in sorted(gt):
        raw_path = Path(args.raw_scores) / (vid + ".json")
        if not raw_path.is_file():
            continue
        raw = json.loads(raw_path.read_text())
        rec = {"video_id": vid, "n_frames": len(gt[vid]),
               "score_raw": align(weighted_scores(raw, {}, 1), len(gt[vid]), vid).tolist()}
        if args.refined_scores:
            refined = json.loads((Path(args.refined_scores) / (vid + ".json")).read_text())
            sims = json.loads((Path(args.similarities) / (vid + ".json")).read_text())
            rec["score_refined"] = align(weighted_scores(
                refined, sims, args.num_neighbors), len(gt[vid]), vid).tolist()
        records.append(rec)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    print("wrote %d/%d videos to %s" % (len(records), len(gt), args.output))


if __name__ == "__main__":
    main()
