"""Shared frame-level localization evaluation machinery.

Protocol: docs/duplex/FRAME_EVAL_PROTOCOL.md. This module is the single
implementation of the frame grid, the span-to-frame conversion, and the
two pooled statistics (rank ROC-AUC, step-wise average precision) that
every method in the reproduction study is scored with. It is corpus-
general: nothing here knows about HateMM, MultiHateClip, or any
particular scorer.

The rank formulas are lifted unchanged from the pilot code that produced
the frozen HateMM endpoint (scripts/duplex/sentinel_localization_pilot.py
rank_auc, scripts/duplex/frame_level_eval_hatemm.py average_precision).
They were verified against scipy.stats.mannwhitneyu to 16 significant
digits; ``python frame_eval_common.py --selftest`` re-runs that
verification along with grid and conversion checks.

CPU only. No model calls, no I/O beyond what a caller passes in.
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
from scipy import stats

FPS_DEFAULT = 1.0


# ------------------------------------------------------------------- grid
def frame_times(duration, fps=FPS_DEFAULT):
    """Frame timestamps t = 0, 1/fps, 2/fps, ... while t < duration.

    Half-open by construction: a video of duration exactly 30.0 s at
    1 fps yields t = 0..29, i.e. 30 frames.
    """
    duration = float(duration)
    if not (duration > 0):
        raise ValueError("duration must be positive, got %r" % (duration,))
    fps = float(fps)
    if not (fps > 0):
        raise ValueError("fps must be positive, got %r" % (fps,))
    n_max = int(math.ceil(duration * fps)) + 1
    t = np.arange(n_max, dtype=float) / fps
    return t[t < duration]


def build_gt_array(spans, duration, fps=FPS_DEFAULT):
    """Per-frame 0/1 gold labels for one video.

    ``spans`` is an iterable of (start, end) seconds. A frame is positive
    iff its timestamp lies inside some span under half-open containment,
    start <= t < end. Degenerate spans (end <= start) contribute nothing;
    callers that must *account* for them should drop them upstream, where
    the drop can be counted and reported.

    Returns a uint8 array of length len(frame_times(duration, fps)).
    """
    t = frame_times(duration, fps)
    labels = np.zeros(len(t), dtype=np.uint8)
    for span in spans or []:
        start, end = float(span[0]), float(span[1])
        if not (end > start):
            continue
        labels[(t >= start) & (t < end)] = 1
    return labels


def spans_to_frame_scores(spans, values, duration, fps=FPS_DEFAULT,
                          uncovered=np.nan):
    """Spread per-span scores onto the frame grid.

    ``spans`` and ``values`` are parallel sequences; the first span that
    contains a frame's timestamp (half-open) supplies that frame's score.
    Frames no span contains take ``uncovered``. This is the method-side
    counterpart of build_gt_array and is what a transcript-chunk scorer
    calls; the uncovered-frame value is a method-side choice, never part
    of the gold.
    """
    t = frame_times(duration, fps)
    scores = np.full(len(t), float(uncovered), dtype=float)
    filled = np.zeros(len(t), dtype=bool)
    for span, value in zip(spans, values):
        start, end = float(span[0]), float(span[1])
        if not (end > start):
            continue
        hit = (t >= start) & (t < end) & (~filled)
        scores[hit] = float(value)
        filled |= hit
    return scores, filled


# ------------------------------------------------------------- statistics
def _split_pos_neg(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    if scores.shape != labels.shape:
        raise ValueError("scores %s and labels %s differ in shape"
                         % (scores.shape, labels.shape))
    mask = labels.astype(bool)
    return scores[mask], scores[~mask]


def rank_auc(pos, neg):
    """Mann-Whitney rank ROC-AUC from separated positive/negative scores.

    Verbatim the pilot implementation; ties take midranks, so the value
    equals scipy.stats.mannwhitneyu(pos, neg).statistic / (n_pos * n_neg).
    Returns None when either class is empty.
    """
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([np.asarray(pos, float), np.asarray(neg, float)])
    r = stats.rankdata(allv)
    rpos = r[:len(pos)].sum()
    return float((rpos - len(pos) * (len(pos) + 1) / 2.0)
                 / (len(pos) * len(neg)))


def rank_roc_auc(scores, labels):
    """Rank ROC-AUC over a score/label vector. None if single-class."""
    pos, neg = _split_pos_neg(scores, labels)
    return rank_auc(pos, neg)


def _average_precision_pos_neg(pos, neg):
    """Rank-based average precision (step-wise AP, ties collapsed).

    Scores are sorted descending; tied scores form one group, and the
    precision/recall pair is read after each complete group, so the
    result does not depend on the input order within a tie.
    """
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(-scores, kind="mergesort")
    scores = scores[order]
    labels = labels[order]
    n_pos = float(len(pos))
    ap = 0.0
    prev_recall = 0.0
    tp = 0.0
    seen = 0.0
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j < n and scores[j] == scores[i]:
            j += 1
        tp += float(labels[i:j].sum())
        seen += float(j - i)
        recall = tp / n_pos
        precision = tp / seen
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        i = j
    return float(ap)


def average_precision(scores, labels):
    """Step-wise average precision (PR-AUC) over a score/label vector."""
    pos, neg = _split_pos_neg(scores, labels)
    return _average_precision_pos_neg(pos, neg)


# ------------------------------------------------------------- evaluation
def describe_macro(values):
    v = [x for x in values if x is not None]
    if not v:
        return {"macro_auc": None, "macro_auc_sd": None,
                "macro_auc_median": None, "n_videos_both_classes": 0}
    a = np.asarray(v, float)
    return {
        "macro_auc": float(a.mean()),
        "macro_auc_sd": float(a.std(ddof=1)) if len(a) > 1 else None,
        "macro_auc_median": float(np.median(a)),
        "n_videos_both_classes": int(len(a)),
    }


def evaluate(per_video, macro_over=None):
    """Pooled and per-video frame-level evaluation.

    ``per_video`` maps video_id -> (scores, labels), both 1-D and equal
    length, one entry per frame of the 1 fps grid. Frames of all videos
    are pooled for the primary statistics (the LAVAD convention the
    frame-prediction literature reports against); the per-video macro AUC
    is computed alongside over videos carrying both frame classes, so the
    pooled number never hides the within-video ranking quality.

    ``macro_over`` optionally restricts the macro to a set of video ids
    (e.g. the hate-labelled videos only). Videos outside it still count
    towards the pooled statistics.
    """
    ids = sorted(per_video)
    all_scores, all_labels = [], []
    per_video_auc = {}
    for vid in ids:
        s, y = per_video[vid]
        s = np.asarray(s, dtype=float)
        y = np.asarray(y)
        if s.shape != y.shape:
            raise ValueError("video %s: scores %s vs labels %s"
                             % (vid, s.shape, y.shape))
        all_scores.append(s)
        all_labels.append(y)
        if macro_over is not None and vid not in macro_over:
            continue
        a = rank_roc_auc(s, y)
        if a is not None:
            per_video_auc[vid] = a
    scores = np.concatenate(all_scores) if all_scores else np.zeros(0)
    labels = np.concatenate(all_labels) if all_labels else np.zeros(0)
    n_pos = int(np.asarray(labels).astype(bool).sum())
    n_frames = int(len(labels))
    out = {
        "n_videos": len(ids),
        "n_frames": n_frames,
        "n_pos": n_pos,
        "n_neg": n_frames - n_pos,
        "positive_rate": (n_pos / float(n_frames)) if n_frames else None,
        "roc_auc": rank_roc_auc(scores, labels),
        "pr_auc": average_precision(scores, labels),
    }
    macro = describe_macro(list(per_video_auc.values()))
    macro["per_video_auc"] = per_video_auc
    out["per_video"] = macro
    return out


# ---------------------------------------------------------------- selftest
def _check(name, ok, detail=""):
    print("%-58s %s%s" % (name, "OK" if ok else "FAIL",
                          ("  " + detail) if detail else ""))
    return bool(ok)


def selftest(seed=20260818):
    """Re-run the verifications the frozen HateMM endpoint depended on."""
    rng = np.random.default_rng(seed)
    ok = True

    # 1. rank ROC-AUC against scipy.stats.mannwhitneyu, 16 digits.
    worst = 0.0
    for trial in range(200):
        n_pos = int(rng.integers(1, 60))
        n_neg = int(rng.integers(1, 60))
        # Coarse quantisation on purpose: forces heavy ties, which is the
        # regime the locator's half-integer z values actually live in.
        pos = np.round(rng.normal(0.4, 1.0, n_pos) * 4) / 4
        neg = np.round(rng.normal(0.0, 1.0, n_neg) * 4) / 4
        ours = rank_auc(pos, neg)
        ref = stats.mannwhitneyu(pos, neg, alternative="two-sided",
                                 method="asymptotic").statistic
        ref = float(ref) / (n_pos * n_neg)
        worst = max(worst, abs(ours - ref))
    ok &= _check("rank_auc == scipy.mannwhitneyu / (n_pos*n_neg)",
                 worst <= 1e-15, "max abs diff %.3e" % worst)

    # 2. rank_roc_auc(scores, labels) agrees with rank_auc(pos, neg).
    worst = 0.0
    for trial in range(100):
        n = int(rng.integers(4, 200))
        s = np.round(rng.normal(0, 1, n) * 4) / 4
        y = (rng.random(n) < 0.3).astype(np.uint8)
        if y.all() or not y.any():
            continue
        worst = max(worst, abs(rank_roc_auc(s, y) - rank_auc(s[y == 1],
                                                             s[y == 0])))
    ok &= _check("rank_roc_auc(scores, labels) == rank_auc(pos, neg)",
                 worst == 0.0, "max abs diff %.3e" % worst)

    # 3. AP against a brute-force tie-collapsed reference.
    def ap_reference(scores, labels):
        order = np.argsort(-np.asarray(scores, float), kind="mergesort")
        s = np.asarray(scores, float)[order]
        y = np.asarray(labels)[order].astype(float)
        total = y.sum()
        ap, prev_r = 0.0, 0.0
        i = 0
        while i < len(s):
            j = i
            while j < len(s) and s[j] == s[i]:
                j += 1
            tp = y[:j].sum()
            r = tp / total
            p = tp / j
            ap += (r - prev_r) * p
            prev_r = r
            i = j
        return float(ap)

    worst = 0.0
    for trial in range(100):
        n = int(rng.integers(4, 200))
        s = np.round(rng.normal(0, 1, n) * 4) / 4
        y = (rng.random(n) < 0.3).astype(np.uint8)
        if y.all() or not y.any():
            continue
        worst = max(worst, abs(average_precision(s, y) - ap_reference(s, y)))
    ok &= _check("average_precision == tie-collapsed reference AP",
                 worst <= 1e-15, "max abs diff %.3e" % worst)

    # 4. AP is invariant to input order within a tie.
    s = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    y1 = np.array([1, 0, 1, 0, 1])
    y2 = np.array([0, 1, 1, 1, 0])
    ok &= _check("average_precision invariant to within-tie order",
                 average_precision(s, y1) == average_precision(s, y2))

    # 5. Perfect and inverted rankings.
    s = np.array([3.0, 2.0, 1.0, 0.0])
    ok &= _check("rank_roc_auc perfect ranking == 1.0",
                 rank_roc_auc(s, np.array([1, 1, 0, 0])) == 1.0)
    ok &= _check("rank_roc_auc inverted ranking == 0.0",
                 rank_roc_auc(s, np.array([0, 0, 1, 1])) == 0.0)
    ok &= _check("rank_roc_auc all-tied == 0.5",
                 rank_roc_auc(np.ones(6), np.array([1, 1, 1, 0, 0, 0]))
                 == 0.5)
    ok &= _check("rank_roc_auc single-class is None",
                 rank_roc_auc(np.arange(5.0), np.ones(5, dtype=np.uint8))
                 is None)

    # 6. Frame grid: half-open, integer-second, 1 fps.
    ok &= _check("frame_times(30.0) == 0..29",
                 np.array_equal(frame_times(30.0), np.arange(30.0)))
    ok &= _check("frame_times(30.4) == 0..30",
                 np.array_equal(frame_times(30.4), np.arange(31.0)))
    ok &= _check("frame_times(0.5) == [0]",
                 np.array_equal(frame_times(0.5), np.array([0.0])))
    ok &= _check("frame_times(101.22) has 102 frames",
                 len(frame_times(101.22)) == 102)

    # 7. Span containment is half-open [start, end).
    g = build_gt_array([(2.0, 5.0)], 10.0)
    ok &= _check("build_gt_array [2,5) marks t=2,3,4 only",
                 np.array_equal(g, np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                                            dtype=np.uint8)))
    ok &= _check("build_gt_array dtype is uint8", g.dtype == np.uint8)
    ok &= _check("build_gt_array drops degenerate spans",
                 build_gt_array([(3.0, 3.0), (7.0, 6.0)], 10.0).sum() == 0)
    ok &= _check("build_gt_array unions overlapping spans",
                 build_gt_array([(1.0, 4.0), (3.0, 6.0)], 10.0).sum() == 5)
    ok &= _check("build_gt_array clips spans past duration",
                 build_gt_array([(8.0, 99.0)], 10.0).sum() == 2)
    ok &= _check("build_gt_array empty spans is all-negative",
                 build_gt_array([], 10.0).sum() == 0)

    # 8. Score spreading: first covering span wins, uncovered flagged.
    s, filled = spans_to_frame_scores([(0.0, 3.0), (5.0, 7.0)],
                                      [2.0, -1.0], 8.0, uncovered=-99.0)
    ok &= _check("spans_to_frame_scores places and floors correctly",
                 np.array_equal(s, np.array([2, 2, 2, -99, -99, -1, -1,
                                             -99], dtype=float))
                 and filled.sum() == 5)

    # 9. Pooled evaluation equals the direct call on concatenated frames.
    per_video = {}
    cat_s, cat_y = [], []
    for k in range(12):
        n = int(rng.integers(5, 40))
        sv = np.round(rng.normal(0, 1, n) * 4) / 4
        yv = (rng.random(n) < 0.35).astype(np.uint8)
        per_video["v%02d" % k] = (sv, yv)
        cat_s.append(sv)
        cat_y.append(yv)
    res = evaluate(per_video)
    cat_s = np.concatenate(cat_s)
    cat_y = np.concatenate(cat_y)
    ok &= _check("evaluate pooled ROC-AUC == direct pooled call",
                 res["roc_auc"] == rank_roc_auc(cat_s, cat_y))
    ok &= _check("evaluate pooled PR-AUC == direct pooled call",
                 res["pr_auc"] == average_precision(cat_s, cat_y))
    ok &= _check("evaluate frame counts add up",
                 res["n_frames"] == len(cat_y)
                 and res["n_pos"] == int(cat_y.sum()))

    print("")
    print("selftest %s" % ("PASSED" if ok else "FAILED"))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true",
                    help="run the unit checks and exit")
    args = ap.parse_args()
    if not args.selftest:
        ap.print_help()
        return 0
    return 0 if selftest() else 1


if __name__ == "__main__":
    sys.exit(main())
