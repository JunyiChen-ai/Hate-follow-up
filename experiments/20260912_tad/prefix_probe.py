#!/usr/bin/env python3
"""Pre-decision representation probe: does the prefix encode which window is hateful?

Every ceiling measured so far reads the model AFTER it has decided (the Yes/No log-odds of a window branch,
or the hidden state at that branch's answer position). This script reads it BEFORE: one prefix forward per
video, then the hidden states of the image tokens of each frame, pooled per 8-second window. Those vectors
are the model's contextual encoding of what it saw inside that window, with the whole video in context and
no question asked.

If a classifier fitted on the test labels beats the frozen read (.758 / .621 window-level within), the
ordering exists in the representation and a read-out method is possible. If it does not, the model does
not encode it and no method built on this model at this granularity can recover it.

Limitation, stated: only the image tokens are located. Frames are identifiable by the image token id and
come in order with known timestamps; transcript lines would need a character-to-token mapping through the
image expansion, which is not built. So this probes the visual side of the prefix only, which is the side
HateClipSeg depends on (removing frames costs it .080 pooled ROC, removing the transcript .006).

Writes runs/<exp_id>/<run>/prefix_hidden/<ds>__<vid>.npz with one fp16 vector per window that contains at
least one frame. No labels are read here.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CODE_PATH = "experiments/20260912_tad/prefix_probe.py"
sys.path.insert(0, str(ROOT))
from src.mllm_judge import Judge, MODEL  # noqa: E402
from src.video_inputs import (fixed_windows, frame_paths, load_asr, load_manifest,  # noqa: E402
                              within_defined_ids)

SEED = 0


@torch.no_grad()
def prefix_states(judge, enc):
    """last_hidden_state over the whole prefix (one forward, no cache reuse)."""
    out = judge.model.model(**judge.model_inputs(enc), use_cache=False)
    h = out.last_hidden_state[0]
    del out
    return h


def frame_blocks(input_ids, image_token_id, n_frames):
    """Start/end index of each contiguous run of image tokens, in order."""
    ids = input_ids.tolist()
    blocks, i = [], 0
    while i < len(ids):
        if ids[i] == image_token_id:
            j = i
            while j < len(ids) and ids[j] == image_token_id:
                j += 1
            blocks.append((i, j))
            i = j
        else:
            i += 1
    if len(blocks) != n_frames:
        raise AssertionError(f"found {len(blocks)} image blocks for {n_frames} frames")
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="prefix_probe")
    ap.add_argument("--exp-id", default="20260912_tad")
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"])
    ap.add_argument("--manifest", default=str(ROOT / "data/omsl_v6_inputs/manifests/all_test.jsonl"))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--window-seconds", type=float, default=8.0)
    ap.add_argument("--only-within-defined", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(SEED)

    out_dir = ROOT / "runs" / args.exp_id / args.run_name
    (out_dir / "prefix_hidden").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out_dir / "run.log"), logging.StreamHandler(sys.stdout)])
    logging.info("host %s", socket.gethostname())
    (out_dir / "run.pid").write_text(str(os.getpid()))
    (out_dir / "config.json").write_text(json.dumps(
        {**vars(args), "code_path": CODE_PATH, "date": time.strftime("%Y-%m-%d"),
         "host": socket.gethostname(), "seed": SEED}, indent=2))

    rows = load_manifest(args.manifest, args.datasets)
    if args.only_within_defined:
        keep = within_defined_ids(args.datasets)
        rows = [r for r in rows if (r["dataset"], r["video_id"]) in keep]
    rows.sort(key=lambda r: (r["dataset"], r["video_id"]))
    if args.limit:
        rows = rows[:args.limit]
    asr = {ds: load_asr(ds) for ds in args.datasets}
    judge = Judge(model_id=args.model)

    t0, n_ok = time.time(), 0
    for n, row in enumerate(rows):
        ds, vid, dur = row["dataset"], row["video_id"], float(row["duration"])
        frames = frame_paths(ds, vid, args.frames, "k20")
        if not frames:
            continue
        segs = asr[ds].get(vid, [])
        msgs, image_files = judge.prefix_messages(frames, segs, with_context=True, with_frames=True)
        _, enc = judge.encode_prefix(msgs, image_files)
        try:
            h = prefix_states(judge, enc)
            blocks = frame_blocks(enc["input_ids"][0], judge.model.config.image_token_id, len(frames))
        except Exception as exc:  # noqa: BLE001
            logging.warning("%s: %s", vid, exc)
            continue
        wins = fixed_windows(dur, args.window_seconds)
        per = {}
        for (t, _), (a, b) in zip(frames, blocks):
            wi = min(int(t // args.window_seconds), len(wins) - 1)
            per.setdefault(wi, []).append(h[a:b].mean(0).float().cpu().numpy())
        if not per:
            continue
        keys = sorted(per)
        np.savez_compressed(out_dir / "prefix_hidden" / f"{ds}__{vid}.npz",
                            keys=np.asarray(keys),
                            H=np.stack([np.mean(per[k], 0) for k in keys]).astype(np.float16),
                            n_windows=np.asarray([len(wins)]))
        del h
        n_ok += 1
        if n % 20 == 0:
            logging.info("%d/%d %s windows_with_frames=%d %.0fs", n, len(rows), vid, len(per), time.time() - t0)
    logging.info("DONE %d videos in %.0fs", n_ok, time.time() - t0)


if __name__ == "__main__":
    main()
