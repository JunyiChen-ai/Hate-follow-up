#!/usr/bin/env python
"""Rasterise Vad-R1's predicted interval onto the 1 fps grid and score it.

Input is the generations.jsonl written by run_vadr1_inference.py. Output is a
scores.jsonl in the same shape every other baseline in this study emits, plus a
report.json with the frame-level numbers and the interval descriptives.

The mapping
-----------
Vad-R1 answers with one `<when>[start, end]</when>` per video, in normalised
units over the whole video, and one `<which>` verdict. Frame-level scoring
needs a value per second, so:

    which = Normal            ->  all-zero score vector
    which = Abnormal + <when> ->  1.0 for every frame t with
                                  start*duration <= t < end*duration, else 0.0
    anything unparseable      ->  all-zero, counted as a parse failure

Containment is half-open, the same convention the gold arrays use, and the
conversion goes through frame_eval_common.build_gt_array so the predicted
vector and the gold vector are produced by one function on one grid. Duration
comes from results/reproduction/gt/<corpus>_test.json, which is the duration
build_gt_arrays.py used, so the two vectors are the same length by
construction; the script asserts it anyway.

The Normal -> all-zero step is the one place this departs from upstream's
evaluation/1-evaluate_detection.py, which maps a Normal verdict to the interval
[0, 1] because its gold for a normal video is also [0, 1] and its mIoU is only
computed when the two verdicts agree. That convention would mark every frame of
a video the model called normal as maximally anomalous, which is the opposite
of what the model said. On a frame grid, "no anomaly" is an all-zero vector.

What the AUC means here
-----------------------
The score vector is binary. A binary score gives the ROC curve exactly one
interior operating point, so the pooled ROC-AUC reduces to a function of the
true-positive and false-positive rates at that single threshold and is not
comparable in resolution to a continuous scorer's AUC. It is reported because
the study reports it for every method through one evaluator, and it is the
honest number for a method that emits one interval. The interval descriptives
below it -- frame-level IoU against the gold frames, upstream-style interval
IoU against the gold envelope, and R@IoU -- are what actually characterise the
prediction, and they are descriptive rather than the study's primary statistic.

Usage
-----
    python rasterize_and_eval.py --corpus hatemm
    python rasterize_and_eval.py --selftest
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINES = os.path.dirname(HERE)
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASELINES))
sys.path.insert(0, BASELINES)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "duplex"))

from hate_common import data as hdata                    # noqa: E402
import frame_eval_common as fec                          # noqa: E402
from eval_baseline_scores import evaluate_scores, format_report  # noqa: E402
from vadr1.run_vadr1_inference import out_dir, parse_generation  # noqa: E402

GT_ROOT = os.path.join(PROJECT_ROOT, "results", "reproduction", "gt")


# --------------------------------------------------------------- durations
def gt_durations(corpus, split="test"):
    """video_id -> duration in seconds, from the gold sidecar.

    The sidecar is the only duration of record: it is the number
    build_gt_arrays.py rasterised the gold spans with, so reading it here is
    what guarantees the predicted vector lands on the same grid.
    """
    path = os.path.join(GT_ROOT, "%s_%s.json" % (corpus, split))
    with open(path, encoding="utf-8") as fh:
        meta = json.load(fh)
    return {vid: float(rec["duration"])
            for vid, rec in meta["per_video"].items()}


# ------------------------------------------------------------ rasterising
def rasterise(when, duration, n_frames):
    """One interval -> a 0/1 float vector of length n_frames.

    ``when`` is [start, end] in normalised units, or None. Returns
    (scores, flags) where flags names every departure from a clean interval so
    the caller can count them instead of silently absorbing them.
    """
    flags = []
    if when is None:
        return np.zeros(n_frames, dtype=float), ["no_interval"]

    start, end = float(when[0]), float(when[1])
    if not (np.isfinite(start) and np.isfinite(end)):
        return np.zeros(n_frames, dtype=float), ["non_finite"]
    if start < 0.0 or end < 0.0 or start > 1.0 or end > 1.0:
        flags.append("out_of_unit_range")
        start = min(max(start, 0.0), 1.0)
        end = min(max(end, 0.0), 1.0)
    if not (end > start):
        # Includes upstream's own degenerate case, a model that answered
        # [0.0, 0.0]. Nothing is being localised, so nothing is marked.
        flags.append("degenerate_interval")
        return np.zeros(n_frames, dtype=float), flags

    scores = fec.build_gt_array([(start * duration, end * duration)],
                                duration).astype(float)
    if len(scores) != n_frames:
        raise ValueError("rasterised %d frames but the gold has %d"
                         % (len(scores), n_frames))
    if scores.sum() == 0:
        # A short interval can fall strictly between two integer seconds.
        flags.append("covers_no_frame")
    return scores, flags


def spans_from_array(arr):
    """Contiguous runs of 1 in a 0/1 array, as (start_frame, end_frame)."""
    arr = np.asarray(arr).astype(bool)
    spans = []
    start = None
    for i, v in enumerate(arr):
        if v and start is None:
            start = i
        elif not v and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(arr)))
    return spans


def frame_iou(pred, gold):
    """Intersection over union of the two positive frame sets."""
    p = np.asarray(pred).astype(bool)
    g = np.asarray(gold).astype(bool)
    union = int((p | g).sum())
    if union == 0:
        return None
    return float((p & g).sum()) / union


def interval_iou(a, b):
    """Upstream's calculate_miou, on two [start, end] pairs."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def gold_envelope(gold, duration):
    """The gold's outer [start, end] in normalised units, or None.

    Upstream's mIoU compares two single intervals. The gold here can be several
    spans, so the envelope from the first positive frame to the last is what is
    compared, and it is only ever generous to the prediction.
    """
    idx = np.nonzero(np.asarray(gold))[0]
    if len(idx) == 0:
        return None
    return [float(idx[0]) / duration, float(idx[-1] + 1) / duration]


# ------------------------------------------------------------------- main
def load_generations(path):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[rec["video_id"]] = rec
    return out


def run(corpus, split, arm, gen_path, out_root, reparse):
    gt = hdata.gt_arrays(corpus, split)
    durations = gt_durations(corpus, split)
    labels = hdata.load_labels(corpus)
    hate_ids = {v for v in gt if labels.get(v) == 1}
    gens = load_generations(gen_path)

    scores = {}
    status_counts = collections.Counter()
    flag_counts = collections.Counter()
    per_video = {}
    for vid, rec in sorted(gens.items()):
        if vid not in gt:
            status_counts["no_gold"] += 1
            continue
        if reparse and rec.get("output"):
            rec = dict(rec, **parse_generation(rec["output"],
                                               rec.get("arm", arm)))
        status = rec.get("parse_status", "unparsed")
        status_counts[status] += 1
        when = rec.get("when")
        if status not in ("positive_interval", "unparsed_interval"):
            when = None
        gold = np.asarray(gt[vid])
        s, flags = rasterise(when, durations[vid], len(gold))
        for f in flags:
            flag_counts[f] += 1
        scores[vid] = s
        per_video[vid] = {
            "parse_status": status,
            "verdict": rec.get("verdict"),
            "when": when,
            "n_pred_frames": int(s.sum()),
            "n_gold_frames": int(gold.sum()),
            "flags": flags,
        }

    # ---- frame-level statistics, through the study's one evaluator
    frame_res = evaluate_scores(scores, gt, hate_ids)

    # ---- interval descriptives, over the videos that carry gold positives
    fious, iious, r03, r05, r07 = [], [], [], [], []
    for vid, s in scores.items():
        gold = np.asarray(gt[vid])
        if gold.sum() == 0:
            continue
        fi = frame_iou(s, gold)
        if fi is None:
            continue
        fious.append(fi)
        r03.append(1.0 if fi >= 0.3 else 0.0)
        r05.append(1.0 if fi >= 0.5 else 0.0)
        r07.append(1.0 if fi >= 0.7 else 0.0)
        env = gold_envelope(gold, durations[vid])
        pred_when = per_video[vid]["when"]
        iious.append(interval_iou(env, pred_when) if pred_when else 0.0)

    def _stats(v):
        if not v:
            return {"n": 0, "mean": None, "median": None}
        a = np.asarray(v, float)
        return {"n": int(len(a)), "mean": float(a.mean()),
                "median": float(np.median(a))}

    intervals = {
        "note": "descriptive only; computed over videos with at least one "
                "gold positive frame. A video the model called normal enters "
                "with IoU 0.",
        "frame_iou": _stats(fious),
        "interval_iou_vs_gold_envelope": _stats(iious),
        "recall_at_frame_iou_0.3": _stats(r03)["mean"],
        "recall_at_frame_iou_0.5": _stats(r05)["mean"],
        "recall_at_frame_iou_0.7": _stats(r07)["mean"],
    }

    # ---- video-level verdict against the corpus label, free of charge
    tp = fp = fn = tn = 0
    n_no_verdict = 0
    for vid in scores:
        verdict = per_video[vid]["verdict"]
        if verdict is None:
            n_no_verdict += 1
        pred = 1 if verdict == "positive" else 0
        true = int(labels.get(vid, 0))
        if pred and true:
            tp += 1
        elif pred and not true:
            fp += 1
        elif not pred and true:
            fn += 1
        else:
            tn += 1
    prec = tp / float(tp + fp) if (tp + fp) else None
    rec_ = tp / float(tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec_ / (prec + rec_)) if (prec and rec_) else None
    video_level = {
        "note": "the model's own <which> verdict against the corpus hate "
                "label. An unparsed verdict counts as a negative prediction.",
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_verdict_unparsed": n_no_verdict,
        "accuracy": (tp + tn) / float(tp + fp + fn + tn) if scores else None,
        "precision": prec, "recall": rec_, "f1": f1,
    }

    report = {
        "corpus": corpus,
        "split": split,
        "arm": arm,
        "generations": gen_path,
        "n_generations_read": len(gens),
        "n_videos_scored": len(scores),
        "parse_status_counts": dict(status_counts),
        "rasterisation_flag_counts": dict(flag_counts),
        "n_all_zero_from_parse_failure": int(
            status_counts["positive_no_interval"] + status_counts["unparsed"]),
        "score_resolution_caveat":
            "scores are binary, so the ROC curve has one interior operating "
            "point and the AUC below is coarse by construction",
        "frame_level": frame_res,
        "intervals": intervals,
        "video_level_verdict": video_level,
    }

    dest = out_root or os.path.dirname(gen_path)
    os.makedirs(dest, exist_ok=True)
    scores_path = os.path.join(dest, "scores.jsonl")
    with open(scores_path, "w", encoding="utf-8") as fh:
        for vid in sorted(scores):
            fh.write(json.dumps({
                "video_id": vid,
                "score_interval": [float(x) for x in scores[vid]],
            }) + "\n")
    report_path = os.path.join(dest, "report.json")
    slim = dict(report)
    slim["frame_level"] = {k: v for k, v in frame_res.items()
                           if k != "per_video"}
    slim["frame_level"]["per_video"] = {
        k: v for k, v in frame_res["per_video"].items() if k != "per_video_auc"}
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump({"report": slim, "per_video": per_video}, fh, indent=2,
                  ensure_ascii=False)
        fh.write("\n")

    print(format_report(frame_res, "%s / %s / vadr1 (%s) / score_interval"
                        % (corpus, split, arm)))
    print("  parse statuses         %s" % dict(status_counts))
    print("  rasterisation flags    %s" % (dict(flag_counts) or "none"))
    print("  frame IoU              mean %s, median %s (n=%d)"
          % (_fmt(intervals["frame_iou"]["mean"]),
             _fmt(intervals["frame_iou"]["median"]),
             intervals["frame_iou"]["n"]))
    print("  interval IoU vs env.   mean %s, median %s"
          % (_fmt(intervals["interval_iou_vs_gold_envelope"]["mean"]),
             _fmt(intervals["interval_iou_vs_gold_envelope"]["median"])))
    print("  R@frame-IoU 0.3/.5/.7  %s / %s / %s"
          % (_fmt(intervals["recall_at_frame_iou_0.3"]),
             _fmt(intervals["recall_at_frame_iou_0.5"]),
             _fmt(intervals["recall_at_frame_iou_0.7"])))
    print("  video-level verdict    acc %s  P %s  R %s  F1 %s"
          % (_fmt(video_level["accuracy"]), _fmt(video_level["precision"]),
             _fmt(video_level["recall"]), _fmt(video_level["f1"])))
    print("  wrote %s" % scores_path)
    print("  wrote %s" % report_path)
    return report


def _fmt(x):
    return "n/a" if x is None else "%.4f" % x


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", choices=list(hdata.CORPORA))
    ap.add_argument("--split", default="test")
    ap.add_argument("--arm", default="anomaly")
    ap.add_argument("--generations", default=None,
                    help="default: the generations.jsonl for --corpus/--arm")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--reparse", action="store_true",
                    help="re-run the parser over the stored raw text instead "
                         "of trusting the fields the inference run wrote")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return 0 if selftest() else 1
    if not args.corpus:
        ap.error("--corpus is required unless --selftest")

    gen_path = args.generations or os.path.join(
        out_dir(args.corpus, args.arm), "generations.jsonl")
    if not os.path.isfile(gen_path):
        raise SystemExit("ABORT: no generations at %s" % gen_path)
    run(args.corpus, args.split, args.arm, gen_path, args.out_root,
        args.reparse)
    return 0


# --------------------------------------------------------------- selftest
def _check(name, ok, detail=""):
    print("%-62s %s%s" % (name, "OK" if ok else "FAIL",
                          ("  " + detail) if detail else ""))
    return bool(ok)


def selftest():
    ok = True

    # A 10 s video sits on frames t = 0..9.
    n = len(fec.frame_times(10.0))
    ok &= _check("10.0 s video has 10 frames on the 1 fps grid", n == 10)

    s, f = rasterise([0.2, 0.5], 10.0, 10)
    ok &= _check("[0.2, 0.5] of 10 s -> [2.0, 5.0) -> frames 2,3,4",
                 np.array_equal(s, np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0.]))
                 and not f)

    s, f = rasterise([0.0, 1.0], 10.0, 10)
    ok &= _check("[0.0, 1.0] marks every frame", s.sum() == 10 and not f)

    s, f = rasterise(None, 10.0, 10)
    ok &= _check("no interval -> all zero, flagged no_interval",
                 s.sum() == 0 and f == ["no_interval"])

    s, f = rasterise([0.5, 0.5], 10.0, 10)
    ok &= _check("[0.5, 0.5] -> all zero, flagged degenerate",
                 s.sum() == 0 and f == ["degenerate_interval"])

    s, f = rasterise([0.8, 0.2], 10.0, 10)
    ok &= _check("reversed interval -> all zero, flagged degenerate",
                 s.sum() == 0 and f == ["degenerate_interval"])

    s, f = rasterise([0.0, 1.6], 10.0, 10)
    ok &= _check("end past 1.0 is clamped and flagged",
                 s.sum() == 10 and f == ["out_of_unit_range"])

    s, f = rasterise([0.31, 0.34], 10.0, 10)
    ok &= _check("[3.1, 3.4) contains no integer second -> zero, flagged",
                 s.sum() == 0 and f == ["covers_no_frame"])

    # A non-integer duration: 101.22 s is 102 frames (frame_eval_common's own
    # check), and half of it is 50.61 s, so frames 0..50 are marked.
    n = len(fec.frame_times(101.22))
    s, f = rasterise([0.0, 0.5], 101.22, n)
    ok &= _check("non-integer duration: [0, 0.5] of 101.22 s marks 0..50",
                 n == 102 and s.sum() == 51 and s[50] == 1.0 and s[51] == 0.0)

    s, _ = rasterise([0.121, 0.826], 100.0, 100)
    ok &= _check("the prompt's own example [0.121, 0.826] of 100 s -> 13..82",
                 s.sum() == 70 and s[12] == 0.0 and s[13] == 1.0
                 and s[82] == 1.0 and s[83] == 0.0)

    # Rasterised prediction and gold come off the same function, so a
    # prediction that exactly names the gold span must reproduce it.
    gold = fec.build_gt_array([(4.0, 9.0)], 20.0)
    pred, _ = rasterise([4.0 / 20.0, 9.0 / 20.0], 20.0, len(gold))
    ok &= _check("prediction equal to the gold span reproduces the gold array",
                 np.array_equal(pred, gold.astype(float))
                 and frame_iou(pred, gold) == 1.0)

    ok &= _check("frame_iou of disjoint sets is 0",
                 frame_iou(np.array([1, 1, 0, 0]),
                           np.array([0, 0, 1, 1])) == 0.0)
    ok &= _check("frame_iou 2 shared of 4 union is 0.5",
                 frame_iou(np.array([1, 1, 1, 0]),
                           np.array([0, 1, 1, 1])) == 0.5)
    ok &= _check("frame_iou of two empty sets is None",
                 frame_iou(np.zeros(4), np.zeros(4)) is None)

    ok &= _check("interval_iou matches a hand-computed case",
                 abs(interval_iou([0.0, 0.5], [0.25, 1.0]) - 0.25) < 1e-12)
    ok &= _check("interval_iou of identical intervals is 1",
                 interval_iou([0.2, 0.6], [0.2, 0.6]) == 1.0)

    env = gold_envelope(np.array([0, 1, 1, 0, 1, 0, 0, 0, 0, 0]), 10.0)
    ok &= _check("gold envelope spans first to last positive frame",
                 env == [0.1, 0.5])
    ok &= _check("gold envelope of an all-negative video is None",
                 gold_envelope(np.zeros(10), 10.0) is None)

    ok &= _check("spans_from_array finds both runs",
                 spans_from_array([0, 1, 1, 0, 1]) == [(1, 3), (4, 5)])

    # Parse -> rasterise, end to end on synthetic text.
    text = ("<answer>classified as <which>Abnormal</which> ... "
            "<when>[0.25, 0.75]</when></answer>")
    p = parse_generation(text)
    s, f = rasterise(p["when"], 8.0, 8)
    ok &= _check("end to end: abnormal [0.25, 0.75] of 8 s -> frames 2..5",
                 np.array_equal(s, np.array([0, 0, 1, 1, 1, 1, 0, 0.]))
                 and not f)

    p = parse_generation("<which>Normal</which> nothing here")
    s, f = rasterise(None if p["parse_status"] != "positive_interval"
                     else p["when"], 8.0, 8)
    ok &= _check("end to end: a Normal verdict rasterises to all zero",
                 s.sum() == 0)

    # Length agreement against the real gold arrays, if they are on disk.
    for corpus in hdata.CORPORA:
        try:
            gt = hdata.gt_arrays(corpus, "test")
            dur = gt_durations(corpus, "test")
        except FileNotFoundError:
            print("%-62s SKIP  (gold not on disk)" % ("gold grid: " + corpus))
            continue
        bad = []
        for vid, arr in gt.items():
            s, _ = rasterise([0.0, 1.0], dur[vid], len(arr))
            if len(s) != len(arr) or s.sum() != len(arr):
                bad.append(vid)
        ok &= _check("gold grid: %s, %d videos rasterise to gold length"
                     % (corpus, len(gt)), not bad,
                     "" if not bad else "%d mismatches, e.g. %s"
                     % (len(bad), bad[:3]))

    print("")
    print("selftest %s" % ("PASSED" if ok else "FAILED"))
    return ok


if __name__ == "__main__":
    sys.exit(main())
