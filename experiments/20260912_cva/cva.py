#!/usr/bin/env python3
"""CVA: counterfactual attribution of the whole-video verdict to time intervals (label-free).

One Qwen3-VL-8B prefix forward per video, then N + 1 reads on that cached prefix:

    z_video   = log-odds of "does this video violate the rules"
    z_excl(i) = the same question, with the model told to disregard interval i
    s_i       = z_video - z_excl(i)

The video-constant nuisance the 2026-09-12 diagnosis measured (the target group is worth more log-odds
than the window actually being positive, and it is the same group all video long) sits in BOTH terms and
cancels, because both are the same question asked on the same context. See README section 2.

Reads: 1 prefix forward + (N + 1) branch reads on its cache, block-diagonal, so each read equals an
independent two-turn call. SPVL-r2 costs 1 + 1 + up to 2N; this is cheaper. No labels are read anywhere
in this file. score_curve is the RAW per-frame s (the value of the window covering that frame);
cva_compose.py adds the intercept and the rank residual, exactly as for SPVL-r2.
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
CODE_PATH = "experiments/20260912_cva/cva.py"
sys.path.insert(0, str(ROOT))
from src.mllm_judge import MODEL, VIDEO_QUESTION, Judge  # noqa: E402
from src.video_inputs import FPS, fixed_windows, frame_paths, load_asr, load_manifest  # noqa: E402

SEED = 0
WINDOW_SECONDS = 8.0
FRAMES = 20


def exclusion_question(t1, t2):
    """Declared verbatim in README section 4. Do not edit without re-declaring."""
    return (f"Now disregard everything between {t1:.1f}s and {t2:.1f}s of this video -- both what is "
            f"shown and what is said in that interval -- and judge only the remaining parts. "
            f"Considering only those remaining parts, does this video violate any of the above rules?"
            f'\n\nAnswer "Yes" or "No".')


def keeponly_question(t1, t2):
    """Control arm (README section 5): the positive counterpart of the exclusion read."""
    return (f"Now disregard everything except what happens between {t1:.1f}s and {t2:.1f}s of this "
            f"video, and judge only that interval. Considering only that interval, does this video "
            f'violate any of the above rules?\n\nAnswer "Yes" or "No".')


def frame_window_index(duration, wins):
    """Frame centre -> window index on the 4 fps grid. Same mapping SPVL-r2 uses; never a proportional
    resize (that difference silently inflated within in experiments/20260912_tad/)."""
    n = int(math.ceil(duration * FPS))
    centres = (np.arange(n) + 0.5) / FPS
    idx = np.minimum((centres // WINDOW_SECONDS).astype(int), len(wins) - 1)
    return n, idx


def score_video(judge, row, segments, args, verify=False):
    vid, ds, dur = row["video_id"], row["dataset"], float(row["duration"])
    frames = frame_paths(ds, vid, FRAMES, "k20")
    if not frames:
        return None, {"error": "no frames cached"}
    msgs, image_files = judge.prefix_messages(frames, segments, with_context=True, with_frames=True)
    prefix_text, enc = judge.encode_prefix(msgs, image_files)
    wins = fixed_windows(dur, WINDOW_SECONDS)
    qfn = keeponly_question if args.read == "keeponly" else exclusion_question
    questions = [qfn(a, b) for a, b in wins]

    b0_ids, _ = judge.branch_ids(msgs, VIDEO_QUESTION)
    branch_ids = [judge.branch_ids(msgs, q)[0] for q in questions]
    if verify:
        judge.seam_check_tokens(msgs, image_files, enc["input_ids"][0].tolist(), VIDEO_QUESTION, b0_ids)
        for q, b in list(zip(questions, branch_ids))[:2]:
            judge.seam_check_tokens(msgs, image_files, enc["input_ids"][0].tolist(), q, b)

    cache = judge.prefix_cache(enc)
    z_video = judge.cached_margin(cache, b0_ids)
    z_excl = [judge.cached_margin(cache, b) for b in branch_ids]
    if verify:
        checks = {"video_q_cache_vs_plain_dz": abs(z_video - judge.plain_margin(msgs, image_files, VIDEO_QUESTION))}
        k = min(3, len(questions))
        checks["excl_cache_vs_plain_max_dz"] = max(
            (abs(z_excl[i] - judge.plain_margin(msgs, image_files, questions[i])) for i in range(k)), default=0.0)
        checks["peak_mem_GB"] = torch.cuda.max_memory_allocated() / 1e9
        # a position or mask error shifts a read by several nats; bf16 kernel noise stays around 1
        if max(checks["video_q_cache_vs_plain_dz"], checks["excl_cache_vs_plain_max_dz"]) >= 3.0:
            raise SystemExit(f"VERIFY GATE FAILED: {json.dumps(checks)}")
    else:
        checks = None
    del cache

    s = np.asarray([z_video - z for z in z_excl], dtype=float)
    n, idx = frame_window_index(dur, wins)
    curve = s[idx]
    info = {"z_video": float(z_video), "n_windows": len(wins),
            "prefix_tokens": int(enc["input_ids"].shape[1]), "n_branches": len(wins) + 1,
            "windows": [{"i": i, "start": float(a), "end": float(b), "z": float(s[i]),
                         "z_excl": float(z_excl[i])} for i, (a, b) in enumerate(wins)]}
    if checks:
        info["verify"] = checks
    return curve, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--exp-id", default="20260912_cva")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--manifest", default=str(ROOT / "data/omsl_v6_inputs/manifests/all_test.jsonl"))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--read", choices=["exclude", "keeponly"], default="exclude")
    ap.add_argument("--verify-first", type=int, default=2, help="videos to run the cache-vs-plain check on")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(SEED)

    out_dir = ROOT / "runs" / args.exp_id / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out_dir / "run.log"), logging.StreamHandler(sys.stdout)])
    logging.info("host %s", socket.gethostname())
    (out_dir / "run.pid").write_text(str(os.getpid()))
    (out_dir / "config.json").write_text(json.dumps(
        {**vars(args), "code_path": CODE_PATH, "date": time.strftime("%Y-%m-%d"), "host": socket.gethostname(),
         "seed": SEED, "window_seconds": WINDOW_SECONDS, "frames": FRAMES,
         "exclusion_question": exclusion_question(0.0, 8.0)}, indent=2))

    rows = load_manifest(args.manifest, args.datasets)
    rows.sort(key=lambda r: (r["dataset"], r["video_id"]))
    if args.limit:
        rows = rows[:args.limit]
    asr = {ds: load_asr(ds) for ds in args.datasets}
    judge = Judge(model_id=args.model)
    logging.info("%d videos, read=%s", len(rows), args.read)

    fh = open(out_dir / "predictions.jsonl", "w")
    t0, n_ok = time.time(), 0
    for k, row in enumerate(rows):
        ds, vid, dur = row["dataset"], row["video_id"], float(row["duration"])
        try:
            curve, info = score_video(judge, row, asr[ds].get(vid, []), args, verify=k < args.verify_first)
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            logging.warning("%s/%s failed: %s", ds, vid, exc)
            fh.write(json.dumps({"dataset": ds, "video_id": vid, "error": str(exc), "method": "cva"}) + "\n")
            continue
        if curve is None:
            fh.write(json.dumps({"dataset": ds, "video_id": vid, "error": info["error"], "method": "cva"}) + "\n")
            continue
        fh.write(json.dumps({"schema_version": 1, "method": f"cva_{args.read}", "dataset": ds, "video_id": vid,
                             "duration": dur, "native_rate": FPS, "score_curve": [float(x) for x in curve],
                             "intervals": [], "error": None, "calls": info["n_branches"], "seed": SEED,
                             "code_path": CODE_PATH, "extra": info}) + "\n")
        fh.flush()
        n_ok += 1
        if k % 20 == 0:
            logging.info("%d/%d %s N=%d zv=%.2f %.0fs", k, len(rows), vid, info["n_windows"],
                         info["z_video"], time.time() - t0)
    fh.close()
    logging.info("DONE %d/%d videos in %.0fs", n_ok, len(rows), time.time() - t0)


if __name__ == "__main__":
    main()
