#!/usr/bin/env python
"""EventVAD stage 1: cut each video into events. Needs a GPU.

Replaces upstream `src/event_seg/main.py` plus `video_processing.process_video`.

The one structural departure is that this writes **boundaries, not video**
(patch E5). Upstream re-encodes every segment to its own `segment_XXXX.mp4`
with `cv2.VideoWriter` and hands the paths to the scorer. Three reasons not to:
the re-encode is a lossy generation loss between the frames the segmenter
measured and the frames the scorer sees; `cv2.VideoWriter` cannot write the
AV1 inputs back out and its mp4v fallback would silently change the pixels for
a quarter of the MultiHateClip English split; and 525 test videos cut into
events would write tens of thousands of files for no gain, since the scorer
needs 16 frames out of each event and can seek them in the source. The
boundaries are the entire content of upstream's segment directory, so nothing
is lost.

Output, one JSON object per line, in
`runs/legacy_1fps/lab1/reproduction/baselines/eventvad/<corpus>/events.jsonl`.

    video_id, probe, n_frames, decode_fps, events, boundary, event_diag,
    timings

`events` is a list of `[start_frame, end_frame)` pairs that partition
`range(n_frames)` on the decoded stream, which `decode_fps` maps to seconds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINES = os.path.dirname(HERE)
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASELINES))
for _p in (HERE, BASELINES):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hate_common import data as hdata            # noqa: E402
import boundary as bnd                           # noqa: E402
import config as cfgmod                          # noqa: E402
import graph as gmod                             # noqa: E402
import video_io                                  # noqa: E402

VIDEO_DIRS = {
    "hatemm": "/home/jehc223/data/HateMM/video",
    "mhclip_en": "/home/jehc223/data/Multihateclip/English/video_mp4",
    "mhclip_zh": "/home/jehc223/data/Multihateclip/Chinese/video",
}
VIDEO_EXT = ".mp4"


def video_path(corpus, video_id):
    return os.path.join(VIDEO_DIRS[corpus], video_id + VIDEO_EXT)


def out_dir(corpus, root=None):
    root = root or os.path.join(PROJECT_ROOT, "results", "reproduction",
                                "baselines", "eventvad")
    return os.path.join(root, corpus)


def cohort(corpus, split, limit=None):
    """Split ids that have gold, in manifest order. Same rule as vadr1."""
    gt = hdata.gt_arrays(corpus, split)
    ids = [v for v in hdata.load_split(corpus, split) if v in gt]
    dropped = [v for v in hdata.load_split(corpus, split) if v not in gt]
    if limit:
        ids = ids[:limit]
    return ids, dropped


def already_done(path):
    if not os.path.isfile(path):
        return set()
    done = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("error"):
                done.discard(rec["video_id"])
            else:
                done.add(rec["video_id"])
    return done


def segment_one(path, extractor, cfg):
    """Decode, featurise, build the graph, propagate, cut. Returns a record."""
    timings = {}
    t0 = time.time()
    pr = video_io.probe(path, cfg)
    timings["probe"] = time.time() - t0

    t0 = time.time()
    clip_feats, flow_feats = extractor.extract(video_io.iter_frames(pr))
    timings["features"] = time.time() - t0
    n = clip_feats.shape[0]

    t0 = time.time()
    adj = gmod.build_dynamic_graph(clip_feats, flow_feats, pr.decode_fps, cfg)
    timings["graph"] = time.time() - t0

    nodes = gmod.fuse_node_features(clip_feats, flow_feats, cfg)
    t0 = time.time()
    propagated = gmod.graph_propagation(nodes, adj, cfg)
    timings["propagation"] = time.time() - t0

    bounds, bdiag = bnd.detect_boundaries(propagated, pr.decode_fps, cfg)
    events, ediag = bnd.events_from_boundaries(bounds, n, pr.decode_fps, cfg)

    return {
        "probe": pr.as_dict(),
        "n_frames": int(n),
        "decode_fps": float(pr.decode_fps),
        "n_edges": int(adj.nnz),
        "events": [[int(a), int(b)] for a, b in events],
        "boundaries": [int(b) for b in bounds],
        "boundary": bdiag,
        "event_diag": ediag,
        "timings": {k: round(v, 3) for k, v in timings.items()},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=list(hdata.CORPORA))
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--raft-ckpt", default=None)
    ap.add_argument("--restart", action="store_true",
                    help="ignore an existing events.jsonl and start over")
    ap.add_argument("--selftest", action="store_true")
    cfgmod.add_config_args(ap)
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if not args.corpus:
        ap.error("--corpus is required unless --selftest")

    cfg = cfgmod.config_from_args(args)
    dest = out_dir(args.corpus, args.out_root)
    os.makedirs(dest, exist_ok=True)
    ev_path = os.path.join(dest, "events.jsonl")
    if args.restart and os.path.isfile(ev_path):
        os.remove(ev_path)

    ids, dropped = cohort(args.corpus, args.split, args.limit)
    done = already_done(ev_path)
    todo = [v for v in ids if v not in done]
    print("%s: %d in cohort (%d split ids without gold), %d already done, "
          "%d to do" % (args.corpus, len(ids), len(dropped), len(done),
                        len(todo)))
    print("config: %s" % json.dumps(cfg.as_dict(), sort_keys=True))
    if not todo:
        print("nothing to do")
        return 0

    from features import FeatureExtractor, DEFAULT_RAFT_CKPT
    extractor = FeatureExtractor(
        device=args.device, raft_ckpt=args.raft_ckpt or DEFAULT_RAFT_CKPT,
        raft_iters=cfg.raft_iters, chunk_size=cfg.chunk_size)

    meta = {
        "corpus": args.corpus, "split": args.split,
        "config": cfg.as_dict(), "n_cohort": len(ids),
        "split_ids_without_gold": dropped,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(dest, "segment_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    started = time.time()
    frames_seen = 0
    with open(ev_path, "a", encoding="utf-8") as out:
        for k, vid in enumerate(todo, 1):
            path = video_path(args.corpus, vid)
            rec = {"video_id": vid, "video_path": path}
            t0 = time.time()
            try:
                rec.update(segment_one(path, extractor, cfg))
                frames_seen += rec["n_frames"]
            except Exception as exc:                    # noqa: BLE001
                rec["error"] = "%s: %s" % (type(exc).__name__, exc)
            rec["wall_s"] = round(time.time() - t0, 3)
            out.write(json.dumps(rec) + "\n")
            out.flush()

            elapsed = time.time() - started
            rate = frames_seen / elapsed if elapsed > 0 else 0.0
            eta = (len(todo) - k) * (elapsed / k) / 60.0
            print("[%4d/%4d] %-24s %6d frames %5d events %7.1fs "
                  "| %5.1f frame/s | eta %5.1f min"
                  % (k, len(todo), vid, rec.get("n_frames", -1),
                     len(rec.get("events", [])), rec["wall_s"], rate, eta),
                  flush=True)

    print("done in %.1f min" % ((time.time() - started) / 60.0))
    return 0


# ------------------------------------------------------------------- tests
def _check(name, ok, detail=""):
    print("  %s %s%s" % ("PASS" if ok else "FAIL", name,
                         ("  -- " + detail) if detail else ""))
    return bool(ok)


def selftest():
    """CPU-only checks over the pure-numpy half of the stage."""
    print("segment_events selftest")
    ok = True
    cfg = cfgmod.build_config("paper")

    rng = np.random.RandomState(0)
    n, fps = 400, 30.0
    planted = (150, 270)
    clip = np.concatenate([
        rng.randn(150, 512) * 0.1 + 3.0,
        rng.randn(120, 512) * 0.1 - 2.0,
        rng.randn(130, 512) * 0.1 + 1.0]).astype(np.float32)
    flow = (rng.randn(n, 2) @ np.eye(2, 128)).astype(np.float32)

    adj = gmod.build_dynamic_graph(clip, flow, fps, cfg)
    ok &= _check("adjacency is square and symmetric",
                 adj.shape == (n, n) and (abs(adj - adj.T) > 1e-6).nnz == 0,
                 "nnz=%d" % adj.nnz)
    nodes = gmod.fuse_node_features(clip, flow, cfg)
    ok &= _check("node features are (n, 640)", nodes.shape == (n, 640))
    prop = gmod.graph_propagation(nodes, adj, cfg)
    ok &= _check("propagation preserves shape and stays finite",
                 prop.shape == nodes.shape and np.isfinite(prop).all())
    ok &= _check("propagation centres the features (Eq. 8)",
                 abs(prop.mean(axis=0)).max() < 1e-3,
                 "max|mean|=%.2e" % abs(prop.mean(axis=0)).max())

    # The divergence signal must peak at the planted transitions before any
    # question about smoothing arises.
    s = bnd.divergence(prop)
    top2 = sorted(int(i) + 1 for i in np.argsort(-s)[:2])
    ok &= _check("divergence peaks at the planted transitions",
                 top2 == list(planted), "peaks at %s, planted %s"
                 % (top2, list(planted)))

    bounds, diag = bnd.detect_boundaries(prop, fps, cfg)
    events, ediag = bnd.events_from_boundaries(bounds, n, fps, cfg)
    ok &= _check("default ma_mode recovers a boundary",
                 len(bounds) >= 1, "bounds=%s diag=%s" % (list(bounds), diag))
    ok &= _check("events partition the frame range",
                 events[0][0] == 0 and events[-1][1] == n
                 and all(a[1] == b[0] for a, b in zip(events, events[1:])),
                 "%d events" % len(events))

    # The measurement DESIGN G7 and boundary.py record: a centred normaliser
    # of the same width as the smoothed bump cannot see it.
    ratios = {}
    for mode in ("upstream", "trailing_aligned", "centered"):
        cfg_m = cfgmod.build_config("paper", ma_mode=mode)
        b_m, d_m = bnd.detect_boundaries(prop, fps, cfg_m)
        ratios[mode] = (len(b_m), list(b_m))
    ok &= _check("trailing window fires where the centred one does not",
                 ratios["upstream"][0] >= 1 and ratios["centered"][0] == 0,
                 "%s" % ratios)
    ok &= _check("trailing_aligned reports later than upstream",
                 (not ratios["trailing_aligned"][1]
                  or ratios["trailing_aligned"][1][0] > ratios["upstream"][1][0]),
                 "%s vs %s" % (ratios["trailing_aligned"][1],
                               ratios["upstream"][1]))

    # gamma is re-read per second: the same physical decay at any decode rate.
    cfg_s = cfgmod.build_config("paper")
    ok &= _check("gamma rescales with the decode rate",
                 abs(cfg_s.gamma_per_frame(30.0) - 0.6) < 1e-12
                 and abs(cfg_s.gamma_per_frame(15.0) - 1.2) < 1e-12,
                 "30fps=%.4f 15fps=%.4f" % (cfg_s.gamma_per_frame(30.0),
                                            cfg_s.gamma_per_frame(15.0)))
    print("  %s" % ("all passed" if ok else "FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
