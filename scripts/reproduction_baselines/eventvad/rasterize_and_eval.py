#!/usr/bin/env python
"""EventVAD stage 3: event scores -> 1 fps arrays -> the study's evaluator.

CPU only.

Upstream's own rasteriser is `src/evaluate.py`, which does not run: line 44
reads `for line in f:s` and the file fails to compile. What it intended is
`scores[s:e] = score` over a frame grid whose length is the largest segment end
seen, scored against a UCF-Crime `tag.txt`. This study scores against the
frozen gold arrays in `results/reproduction/gt/`, on the 1 fps grid
`docs/duplex/FRAME_EVAL_PROTOCOL.md` fixes, through the same
`frame_eval_common.evaluate` every other baseline goes through.

The mapping, which is the whole of the adaptation
    An event covers decoded frames `[s, e)`, i.e. seconds `[s/fps, e/fps)`.
    Gold second `i` covers `[i, i+1)` and takes the score of the event
    containing its midpoint `i + 0.5`, clamped to the last event. That is the
    convention the MACIL-SD port already uses to cross a non-integer grid
    ratio (README, "Scores back down"), and it is a lookup rather than a
    `np.repeat` because `fps` is not an integer multiple of 1 here either.

Unparsed events
    An event whose text carried no number scores 0.0 and is counted. The
    alternative -- dropping the video -- would change the cohort between arms
    and make two arms' pooled numbers incomparable. `frame_eval.json` records
    `n_events_unparsed` and `frac_frames_unparsed`, so a number can be read
    against how much of it was filled in.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINES = os.path.dirname(HERE)
for _p in (HERE, BASELINES):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hate_common import data as hdata            # noqa: E402
import eval_baseline_scores as ebs               # noqa: E402
import prompt as pmod                            # noqa: E402
import segment_events as seg                     # noqa: E402

UNPARSED_FILL = 0.0


def rasterise(events, scores, n_gold, fps):
    """Score per gold second, by midpoint lookup. Returns (array, n_filled)."""
    out = np.empty(n_gold, dtype=np.float64)
    if not events:
        out[:] = UNPARSED_FILL
        return out, n_gold
    starts = np.asarray([e[0] for e in events], dtype=np.float64)
    vals = np.asarray([UNPARSED_FILL if s is None else s for s in scores],
                      dtype=np.float64)
    filled = np.asarray([s is None for s in scores])

    mid_frames = (np.arange(n_gold) + 0.5) * fps
    # events are contiguous and sorted, so the containing event is the last
    # one whose start is at or before the midpoint.
    idx = np.searchsorted(starts, mid_frames, side="right") - 1
    idx = np.clip(idx, 0, len(events) - 1)
    out[:] = vals[idx]
    return out, int(filled[idx].sum())


def load_scores(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "%s not found -- run score_events.py first" % path)
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("error"):
                continue
            out[rec["video_id"]] = rec
    return out


def run(corpus, split, arm, out_root, json_out=None):
    dest = seg.out_dir(corpus, out_root)
    suffix = "" if arm == pmod.DEFAULT_ARM else "_" + arm
    recs = load_scores(os.path.join(dest, "event_scores%s.jsonl" % suffix))
    gt = hdata.gt_arrays(corpus, split)
    labels = hdata.load_labels(corpus)
    hate_ids = {v for v, y in labels.items() if y == 1}

    scores, per_video = {}, {}
    tot_events = tot_unparsed = tot_frames = tot_filled = 0
    parse_status = {}
    range_rules = {}
    for vid, rec in recs.items():
        if vid not in gt:
            continue
        n_gold = len(gt[vid])
        fps = float(rec["decode_fps"])
        events = [(e["start"], e["end"]) for e in rec["events"]]
        vals = [e["score"] for e in rec["events"]]
        for e in rec["events"]:
            parse_status[e["parse_status"]] = \
                parse_status.get(e["parse_status"], 0) + 1
            range_rules[e["range_rule"]] = range_rules.get(e["range_rule"], 0) + 1
        arr, n_filled = rasterise(events, vals, n_gold, fps)
        scores[vid] = arr
        tot_events += len(events)
        tot_unparsed += sum(1 for v in vals if v is None)
        tot_frames += n_gold
        tot_filled += n_filled
        per_video[vid] = {"n_events": len(events),
                          "n_unparsed": sum(1 for v in vals if v is None),
                          "n_frames": n_gold}

    res = ebs.evaluate_scores(scores, gt, hate_ids=hate_ids)
    ev_counts = np.array([v["n_events"] for v in per_video.values()])
    res["eventvad"] = {
        "arm": arm,
        "n_videos_scored": len(scores),
        "n_events": tot_events,
        "n_events_unparsed": tot_unparsed,
        "frac_events_unparsed": (tot_unparsed / tot_events) if tot_events else 0.0,
        "frac_frames_unparsed": (tot_filled / tot_frames) if tot_frames else 0.0,
        "events_per_video_mean": float(ev_counts.mean()) if len(ev_counts) else 0.0,
        "events_per_video_median": float(np.median(ev_counts)) if len(ev_counts) else 0.0,
        "events_per_video_max": int(ev_counts.max()) if len(ev_counts) else 0,
        "parse_status": parse_status,
        "range_rules": range_rules,
        "unparsed_fill": UNPARSED_FILL,
    }

    sc_path = os.path.join(dest, "scores%s.jsonl" % suffix)
    with open(sc_path, "w", encoding="utf-8") as fh:
        for vid in sorted(scores):
            fh.write(json.dumps({
                "video_id": vid, "n_frames": int(len(scores[vid])),
                "score_event": [float(x) for x in scores[vid]]}) + "\n")

    json_out = json_out or os.path.join(dest, "frame_eval%s.json" % suffix)
    with open(json_out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, default=float)

    print(ebs.format_report(res, "EventVAD %s (arm %s)" % (corpus, arm)))
    print("\nevents: %d over %d videos (median %.0f, max %d per video); "
          "%d unparsed (%.2f%% of events, %.2f%% of frames)"
          % (tot_events, len(scores), res["eventvad"]["events_per_video_median"],
             res["eventvad"]["events_per_video_max"], tot_unparsed,
             100 * res["eventvad"]["frac_events_unparsed"],
             100 * res["eventvad"]["frac_frames_unparsed"]))
    print("parse status: %s" % json.dumps(parse_status, sort_keys=True))
    print("range rules : %s" % json.dumps(range_rules, sort_keys=True))
    print("\nwrote %s\n      %s" % (sc_path, json_out))
    return res


# ------------------------------------------------------------------- tests
def _check(name, ok, detail=""):
    print("  %s %s%s" % ("PASS" if ok else "FAIL", name,
                         ("  -- " + detail) if detail else ""))
    return bool(ok)


def selftest():
    print("rasterize_and_eval selftest")
    ok = True

    # 3 events over 90 frames at 30 fps = 3 s of gold.
    events = [(0, 30), (30, 60), (60, 90)]
    arr, filled = rasterise(events, [0.1, 0.9, 0.5], 3, 30.0)
    ok &= _check("one event per second at 30 fps",
                 np.allclose(arr, [0.1, 0.9, 0.5]) and filled == 0, str(arr))

    # An event boundary inside a second: the midpoint decides.
    events = [(0, 45), (45, 90)]
    arr, _ = rasterise(events, [0.2, 0.8], 3, 30.0)
    ok &= _check("midpoint decides a straddled second",
                 np.allclose(arr, [0.2, 0.8, 0.8]), str(arr))

    # Gold longer than the decoded stream: clamp to the last event.
    events = [(0, 30), (30, 50)]
    arr, _ = rasterise(events, [0.3, 0.7], 5, 30.0)
    ok &= _check("gold past the last event clamps",
                 np.allclose(arr, [0.3, 0.7, 0.7, 0.7, 0.7]), str(arr))

    # Unparsed events are filled and counted.
    events = [(0, 30), (30, 60)]
    arr, filled = rasterise(events, [None, 0.7], 2, 30.0)
    ok &= _check("unparsed event fills and counts",
                 np.allclose(arr, [UNPARSED_FILL, 0.7]) and filled == 1,
                 "%s filled=%d" % (arr, filled))

    # A single event covers everything.
    arr, _ = rasterise([(0, 300)], [0.4], 10, 30.0)
    ok &= _check("single event covers the whole video",
                 np.allclose(arr, 0.4))

    # 25 fps, non-integer ratio.
    events = [(0, 25), (25, 63)]
    arr, _ = rasterise(events, [0.1, 0.9], 3, 25.0)
    ok &= _check("25 fps grid maps by midpoint",
                 np.allclose(arr, [0.1, 0.9, 0.9]), str(arr))

    print("  %s" % ("all passed" if ok else "FAILURES"))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=list(hdata.CORPORA))
    ap.add_argument("--split", default="test")
    ap.add_argument("--arm", default=pmod.DEFAULT_ARM, choices=sorted(pmod.ARMS))
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.corpus:
        ap.error("--corpus is required unless --selftest")
    run(args.corpus, args.split, args.arm, args.out_root, args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
