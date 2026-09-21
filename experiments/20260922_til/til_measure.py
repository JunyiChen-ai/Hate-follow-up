#!/usr/bin/env python3
"""TIL measurement pass: the SPVL-r2 interval measurements on one 8 s window grid with a chosen start offset.

Per video: shared prefix (rules + 20 frames + transcript) -> whole-video Yes/No log-odds z_video -> the model's
own answer appended as a stance turn -> for every window of the grid, an isolated visual branch and an isolated
speech branch (prefix KV cache deep-copied per branch), each read as Yes/No log-odds. Nothing else: no
hypothesis, no chain, no revision. Grid A: --window-offset 0 (0-8, 8-16, ...); grid B: --window-offset 4
(4-12, 12-20, ...). Labels are never read here.
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
sys.path.insert(0, str(ROOT))
from src.mllm_judge import Judge, MODEL, VIDEO_QUESTION  # noqa: E402
from src.video_inputs import FPS, frame_paths, load_asr, load_manifest, window_text  # noqa: E402

CODE_PATH = "experiments/20260922_til/til_measure.py"
SEED = 0
FILL_UNCOVERED = -12.0  # window with no branch at all (no speech and no frames); never happens with frames=20


def grid_windows(duration, seconds, offset):
    """Windows [offset + k*S, offset + (k+1)*S) clipped to the duration; the leading [0, offset) is not measured."""
    wins = []
    t = float(offset)
    while t < duration - 1e-9:
        wins.append((t, min(t + seconds, duration)))
        t += seconds
    if not wins:
        wins = [(0.0, duration)]
    return wins


def yesno_question(i, n, t1, t2, text, kind):
    """SPVL-r2 evidence wording (identical to experiments/20260911_hvl/hvl.py yesno_question with has_hyp=False)."""
    body = text.strip() if text and text.strip() else "(no speech)"
    head = f"Consider only window {i + 1} of {n}, from {t1:.1f}s to {t2:.1f}s of this video."
    if kind == "visual":
        ctx = (f"{head} Look only at the frames whose timestamps fall inside this window and judge the visual "
               f"content alone (imagery, gestures, symbols, on-screen text), ignoring the speech.\n\n")
        q = "Is THIS window one of the segments where visual content that violates the above rules occurs?"
    else:
        ctx = f"{head} Judge only what is spoken in this window: {body}\n\n"
        q = "Is THIS window one of the segments where speech that violates the above rules occurs?"
    return ctx + q + '\n\nAnswer "Yes" or "No".'


def score_video(judge, row, segments, args, verify=False):
    vid, ds, dur = row["video_id"], row["dataset"], float(row["duration"])
    wins = grid_windows(dur, args.window_seconds, args.window_offset)
    frames = frame_paths(ds, vid, args.frames, "k20")
    if not frames:
        return None, {"error": "no frames cached"}
    msgs, image_files = judge.prefix_messages(frames, segments, with_context=True, with_frames=True)
    prefix_text, enc = judge.encode_prefix(msgs, image_files)
    prefix_ids = enc["input_ids"][0].tolist()
    info = {"prefix_tokens": len(prefix_ids), "n_windows": len(wins), "img_tokens": judge.img_tokens[:1], "n_frames": len(frames)}
    cache = judge.prefix_cache(enc)
    b0, b0_text = judge.branch_ids(msgs, VIDEO_QUESTION)
    if verify:
        judge.seam_check_tokens(msgs, image_files, prefix_ids, VIDEO_QUESTION, b0)
    z_video = judge.cached_margin(cache, b0, in_place=True)
    stance = "Yes" if z_video > 0 else "No"
    a0, a0_text = judge.answer_ids(msgs, VIDEO_QUESTION, stance)
    judge.extend_cache(cache, a0)
    history = [{"role": "user", "content": [{"type": "text", "text": VIDEO_QUESTION}]}, judge.turn("assistant", stance)]
    head = prefix_text + b0_text + a0_text
    wtexts = [window_text(segments, a, b) for a, b in wins]
    per = [dict() for _ in wins]
    n_branch = 0
    for kind in ("visual", "speech"):
        for i, ((a, b), t) in enumerate(zip(wins, wtexts)):
            if kind == "speech" and not (t and t.strip()):
                continue  # SPVL-r2 semantics: no speech branch without speech
            q = yesno_question(i, len(wins), a, b, t, kind)
            bids, _ = judge.branch_ids(msgs, q, history, head_text=head)
            per[i][kind] = judge.cached_margin(cache, bids, in_place=False)
            n_branch += 1
    z_win = [max(d.values()) if d else FILL_UNCOVERED for d in per]
    del cache
    info.update({"n_branches": n_branch, "z_video": z_video, "stance": stance})
    if verify:
        z_ref = judge.plain_margin(msgs, image_files, VIDEO_QUESTION)
        info["verify"] = {"video_q_cache_vs_plain_dz": abs(z_video - z_ref), "peak_mem_GB": torch.cuda.max_memory_allocated() / 1e9}
        if abs(z_video - z_ref) >= 3.0:
            raise SystemExit(f"VERIFY GATE FAILED: {info['verify']}")
    L = int(math.ceil(dur * FPS))
    centers = (np.arange(L) + 0.5) / FPS
    idx = np.clip(np.floor((centers - args.window_offset) / args.window_seconds).astype(int), 0, len(wins) - 1)
    curve = np.asarray(z_win, dtype=float)[idx]
    pred = {"schema_version": 1, "method": args.method_name, "dataset": ds, "video_id": vid, "duration": dur,
            "native_rate": FPS, "score_curve": [float(x) for x in curve], "intervals": [], "error": None,
            "calls": 2, "seed": SEED, "code_path": CODE_PATH,
            "extra": {"z_video": z_video, "stance": stance, "window_offset": args.window_offset,
                      "windows": [{"i": i, "start": a, "end": b, "z": z, **{f"z_{k}": v for k, v in per[i].items()}}
                                  for i, ((a, b), z) in enumerate(zip(wins, z_win))],
                      **{k: v for k, v in info.items() if k != "verify"}}}
    return pred, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--exp-id", default="20260922_til")
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"])
    ap.add_argument("--manifest", default=str(ROOT / "data/omsl_v6_inputs/manifests/all_test.jsonl"))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--window-seconds", type=float, default=8.0)
    ap.add_argument("--window-offset", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(SEED)
    out_dir = ROOT / "runs" / args.exp_id / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out_dir / "run.log"), logging.StreamHandler(sys.stdout)])
    logging.info("host %s", socket.gethostname())
    (out_dir / "run.pid").write_text(str(os.getpid()))
    args.method_name = f"til_measure_w{args.window_seconds:g}_off{args.window_offset:g}"
    cfg = dict(vars(args))
    cfg.update({"code_path": CODE_PATH, "date": time.strftime("%Y-%m-%d"), "host": socket.gethostname(), "seed": SEED,
                "fill_uncovered": FILL_UNCOVERED, "video_question": VIDEO_QUESTION,
                "window_question_visual": yesno_question(0, 1, 0.0, 8.0, "", "visual"),
                "window_question_speech": yesno_question(0, 1, 0.0, 8.0, "<text>", "speech")})
    rows = load_manifest(args.manifest, args.datasets)
    if args.limit:
        rows = rows[:args.limit]
    asr = {ds: load_asr(ds) for ds in args.datasets}
    logging.info("videos %d  config %s", len(rows), args.method_name)
    judge = Judge(model_id=args.model)
    import transformers
    cfg.update({"family": judge.family, "same_turn": judge.same_turn, "transformers": transformers.__version__, "torch": torch.__version__})
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    pred_path = out_dir / "predictions.jsonl"
    done = set()
    if pred_path.exists():
        for line in open(pred_path):
            r = json.loads(line)
            if not r.get("error"):
                done.add((r["dataset"], r["video_id"]))
    n_err, t0 = 0, time.time()
    with open(pred_path, "a") as fh:
        for n, row in enumerate(rows):
            key = (row["dataset"], row["video_id"])
            if key in done:
                continue
            segments = asr[row["dataset"]].get(row["video_id"], [])
            try:
                pred, info = score_video(judge, row, segments, args, verify=(args.verify_only or n == 0))
                if pred is None:
                    raise RuntimeError(info.get("error", "unknown"))
                if "verify" in info:
                    logging.info("VERIFY %s %s", row["video_id"], json.dumps(info["verify"]))
                    (out_dir / "verify.json").write_text(json.dumps({"video_id": row["video_id"], **info}, indent=2, default=str))
                    if args.verify_only:
                        logging.info("DONE verify-only")
                        return
                fh.write(json.dumps(pred, ensure_ascii=False) + "\n")
                fh.flush()
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                n_err += 1
                logging.error("OOM %s %s", row["video_id"], str(exc)[:200])
                fh.write(json.dumps({"method": args.method_name, "dataset": row["dataset"], "video_id": row["video_id"],
                                     "score_curve": [], "intervals": [], "error": "OOM"}) + "\n")
            except Exception as exc:
                n_err += 1
                logging.exception("FAILED %s: %s", row["video_id"], exc)
                fh.write(json.dumps({"method": args.method_name, "dataset": row["dataset"], "video_id": row["video_id"],
                                     "score_curve": [], "intervals": [], "error": f"{type(exc).__name__}: {exc}"}) + "\n")
            if (n + 1) % 10 == 0:
                logging.info("progress %d/%d  %.1fs/video  errors %d", n + 1, len(rows), (time.time() - t0) / (n + 1), n_err)
    logging.info("DONE videos=%d errors=%d elapsed=%.0fs", len(rows), n_err, time.time() - t0)


if __name__ == "__main__":
    main()
