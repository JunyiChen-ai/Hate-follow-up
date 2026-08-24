#!/usr/bin/env python
"""Score a baseline's frame-level output against the frozen gold arrays.

One evaluator for every baseline in the reproduction study. It takes a
scores.jsonl written by vadclip/infer.py or dsanet/infer.py -- or any
{video_id: scores(T,)} mapping handed to `evaluate_scores` -- and reports the
pooled statistics through scripts/duplex/frame_eval_common.py. Nothing is
recomputed here: the frame grid, the rank ROC-AUC and the step-wise average
precision all come from that module, so a baseline number and a method number
are produced by the same code.

Reported
    pooled ROC-AUC / PR-AUC over every frame of every scored video, the
    convention frame_eval_common.evaluate uses;
    within-hate macro ROC-AUC, the per-video mean restricted to videos the
    corpus labels hateful. A video labelled normal has an all-negative gold
    array and no within-video ranking to score, so restricting the macro is
    what makes it readable.
    the positive-rate and video counts, so a pooled number can be read against
    its base rate.

CPU only.

Usage
    python eval_baseline_scores.py --corpus hatemm \
        --scores results/reproduction/baselines/vadclip/hatemm/scores.jsonl \
        --branch score_align
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "duplex")))

from hate_common import data as hdata          # noqa: E402
import frame_eval_common as fec                # noqa: E402


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_scores(scores, gt, hate_ids=None):
    """Pooled and within-hate frame-level evaluation.

    ``scores`` maps video_id -> 1-D array on the 1 fps grid; ``gt`` maps
    video_id -> uint8 gold array of the same length. Videos present in gt but
    absent from scores are reported as missing rather than silently skipped.
    """
    missing = sorted(set(gt) - set(scores))
    extra = sorted(set(scores) - set(gt))
    per_video = {}
    for vid in sorted(set(scores) & set(gt)):
        s = np.asarray(scores[vid], dtype=float)
        y = np.asarray(gt[vid])
        if s.shape != y.shape:
            raise ValueError("video %s: %d scores but %d gold frames"
                             % (vid, len(s), len(y)))
        if not np.isfinite(s).all():
            raise ValueError("video %s: non-finite scores" % vid)
        per_video[vid] = (s, y)

    macro_over = None
    if hate_ids is not None:
        macro_over = {v for v in per_video if v in hate_ids}

    out = fec.evaluate(per_video, macro_over=macro_over)
    ordered = sorted(per_video)
    video_y = np.asarray([int(np.asarray(per_video[v][1]).max() > 0)
                          for v in ordered])
    video_max = np.asarray([float(np.asarray(per_video[v][0]).max())
                            for v in ordered])
    video_mean = np.asarray([float(np.asarray(per_video[v][0]).mean())
                             for v in ordered])
    out["video_level"] = {
        "aggregation": ["max", "mean"],
        "n_videos": len(ordered),
        "n_positive": int(video_y.sum()),
        "max_roc_auc": (float(roc_auc_score(video_y, video_max))
                        if len(np.unique(video_y)) == 2 else None),
        "max_pr_auc": (float(average_precision_score(video_y, video_max))
                       if video_y.size else None),
        "mean_roc_auc": (float(roc_auc_score(video_y, video_mean))
                         if len(np.unique(video_y)) == 2 else None),
        "mean_pr_auc": (float(average_precision_score(video_y, video_mean))
                        if video_y.size else None),
    }
    out["n_videos_missing_from_scores"] = len(missing)
    out["videos_missing_from_scores"] = missing[:20]
    out["n_videos_not_in_gold"] = len(extra)
    out["videos_not_in_gold"] = extra[:20]
    return out


def format_report(res, title):
    macro = res["per_video"]
    lines = [
        title,
        "  videos scored          %d" % res["n_videos"],
        "  frames                 %d (%d positive, rate %.4f)"
        % (res["n_frames"], res["n_pos"], res["positive_rate"] or 0.0),
        "  pooled ROC-AUC         %s" % _fmt(res["roc_auc"]),
        "  pooled PR-AUC          %s" % _fmt(res["pr_auc"]),
        "  within-hate macro AUC  %s  (n=%d, sd %s, median %s)"
        % (_fmt(macro["macro_auc"]), macro["n_videos_both_classes"],
           _fmt(macro["macro_auc_sd"]), _fmt(macro["macro_auc_median"])),
        "  video max ROC / AP     %s / %s"
        % (_fmt(res["video_level"]["max_roc_auc"]),
           _fmt(res["video_level"]["max_pr_auc"])),
    ]
    if res["n_videos_missing_from_scores"]:
        lines.append("  MISSING from scores    %d, e.g. %s"
                     % (res["n_videos_missing_from_scores"],
                        ", ".join(res["videos_missing_from_scores"][:5])))
    if res["n_videos_not_in_gold"]:
        lines.append("  scored but no gold     %d, e.g. %s"
                     % (res["n_videos_not_in_gold"],
                        ", ".join(res["videos_not_in_gold"][:5])))
    return "\n".join(lines)


def _fmt(x):
    return "n/a" if x is None else "%.4f" % x


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, choices=list(hdata.CORPORA))
    ap.add_argument("--scores", required=True,
                    help="scores.jsonl from a baseline's infer script")
    ap.add_argument("--split", default="test")
    ap.add_argument("--branch", default=None,
                    help="which score_* field to evaluate; default: every "
                         "field present in the file")
    ap.add_argument("--json-out", default=None,
                    help="write the full result dict here")
    ap.add_argument("--require-full-coverage", action="store_true",
                    help="abort unless every frozen-gold video is scored and "
                         "no score lies outside the frozen cohort")
    args = ap.parse_args(argv)

    scores_sha256 = file_sha256(args.scores)
    records = hdata.load_scores_jsonl(args.scores)
    if not records:
        raise SystemExit("ABORT: no records in %s" % args.scores)
    gt = hdata.gt_arrays(args.corpus, args.split)
    labels = hdata.load_labels(args.corpus)
    hate_ids = {v for v in gt if labels.get(v) == 1}

    branches = ([args.branch] if args.branch
                else sorted(next(iter(records.values())).keys()))
    results = {}
    for branch in branches:
        scores = {v: r[branch] for v, r in records.items() if branch in r}
        if not scores:
            raise SystemExit("ABORT: branch %r absent from %s"
                             % (branch, args.scores))
        results[branch] = evaluate_scores(scores, gt, hate_ids)
        if (args.require_full_coverage and
                (results[branch]["n_videos_missing_from_scores"] or
                 results[branch]["n_videos_not_in_gold"])):
            raise SystemExit(
                f"ABORT: {branch} does not exactly cover frozen {args.split}")
        print(format_report(results[branch],
                            "%s / %s / %s" % (args.corpus, args.split, branch)))
        print("")

    if file_sha256(args.scores) != scores_sha256:
        raise SystemExit("ABORT: score file changed during evaluation: %s" %
                         args.scores)

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)),
                    exist_ok=True)
        payload = {"corpus": args.corpus, "split": args.split,
                   "scores_file": os.path.abspath(args.scores),
                   "scores_sha256": scores_sha256,
                   "n_hate_videos_in_gold": len(hate_ids),
                   "results": results}
        target = os.path.abspath(args.json_out)
        temporary = target + ".tmp"
        with open(temporary, "w") as fh:
            json.dump(payload, fh, indent=2, default=float)
        os.replace(temporary, target)
        print("wrote %s" % args.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
