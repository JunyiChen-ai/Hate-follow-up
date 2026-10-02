#!/usr/bin/env python3
"""TIL measurement pass: the SPVL-r2 interval measurements on one 8 s window grid with a chosen start offset.

Per video: shared prefix (rules + 20 frames + transcript) -> whole-video Yes/No log-odds z_video -> the model's
own answer appended as a stance turn -> for every window of the grid, an isolated visual branch and an isolated
speech branch (prefix KV cache deep-copied per branch), each read as Yes/No log-odds. Nothing else: no
hypothesis, no chain, no revision. Grid A: --window-offset 0 (0-8, 8-16, ...); grid B: --window-offset 4
(4-12, 12-20, ...). Labels are never read here.

Ablation flags (experiments/20260928_infer README §11 and §12): --branches joint, --isolation sequential, --frames 0 (no
frames in the prefix, so no visual branch), --no-transcript-context, --stance none (no verdict turn before the window
branches), --windows asr (ASR segments as windows; gaps stay unscored).
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
from src.mllm_judge import Judge, MODEL, VIDEO_QUESTION, yesno_question  # noqa: E402
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


def score_video(judge, row, segments, args, verify=False):
    vid, ds, dur = row["video_id"], row["dataset"], float(row["duration"])
    if getattr(args, "windows", "fixed") == "fixed":
        wins = grid_windows(dur, args.window_seconds, args.window_offset)
        wtexts = [window_text(segments, a, b) for a, b in wins]
    else:  # asr: one window per ASR segment; a video without transcript has no windows (curve stays FILL_UNCOVERED)
        wins = [(s, e) for s, e, _ in segments]
        wtexts = [t for _, _, t in segments]
    frames = frame_paths(ds, vid, args.frames, "k20") if args.frames > 0 else []
    if args.frames > 0 and not frames:
        return None, {"error": "no frames cached"}
    with_context = not getattr(args, "no_transcript_context", False)
    msgs, image_files = judge.prefix_messages(frames, segments, with_context=with_context, with_frames=args.frames > 0)
    prefix_text, enc = judge.encode_prefix(msgs, image_files)
    prefix_ids = enc["input_ids"][0].tolist()
    info = {"prefix_tokens": len(prefix_ids), "n_windows": len(wins), "img_tokens": judge.img_tokens[:1], "n_frames": len(frames)}
    cache = judge.prefix_cache(enc)
    b0, b0_text = judge.branch_ids(msgs, VIDEO_QUESTION)
    if verify:
        judge.seam_check_tokens(msgs, image_files, prefix_ids, VIDEO_QUESTION, b0)
    if getattr(args, "stance", "verdict") == "none":
        # ablation: no stance turn. The verdict is read on a copy; the window branches follow the prefix directly.
        z_video = judge.cached_margin(cache, b0, in_place=False)
        stance, history, head = None, [], prefix_text
    else:
        z_video = judge.cached_margin(cache, b0, in_place=True)
        stance = "Yes" if z_video > 0 else "No"
        a0, a0_text = judge.answer_ids(msgs, VIDEO_QUESTION, stance)
        judge.extend_cache(cache, a0)
        history = [{"role": "user", "content": [{"type": "text", "text": VIDEO_QUESTION}]}, judge.turn("assistant", stance)]
        head = prefix_text + b0_text + a0_text
    per = [dict() for _ in wins]
    n_branch = 0
    kinds = {"dual": ("visual", "speech"), "joint": ("joint",)}[getattr(args, "branches", "dual")]
    sequential = getattr(args, "isolation", "copy") == "sequential"
    # isolated (default): every branch on a deep copy of the prefix cache, kind-major order (order is irrelevant).
    # sequential (experiments/20260928_infer README §11): window-major order on one cache that keeps every branch's
    # tokens, so a branch sees the question text and assistant header of all earlier branches.
    order = ([(i, k) for i in range(len(wins)) for k in kinds] if sequential else
             [(i, k) for k in kinds for i in range(len(wins))])
    # SPVL-r2 semantics: no speech branch without speech; no visual branch without frames (--frames 0)
    order = [(i, k) for i, k in order if not (k == "speech" and not (wtexts[i] and wtexts[i].strip()))
             and not (k == "visual" and args.frames == 0)]
    if not order and wins:  # no frames and no speech anywhere: one joint branch per window (as spvl.py)
        order = [(i, "joint") for i in range(len(wins))]
    for i, kind in order:
        (a, b), t = wins[i], wtexts[i]
        q = yesno_question(i, len(wins), a, b, t, kind)
        bids, _ = judge.branch_ids(msgs, q, history, head_text=head)
        per[i][kind] = judge.cached_margin(cache, bids, in_place=sequential)
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
    if getattr(args, "windows", "fixed") == "fixed":
        idx = np.clip(np.floor((centers - args.window_offset) / args.window_seconds).astype(int), 0, len(wins) - 1)
        curve = np.asarray(z_win, dtype=float)[idx]
    else:  # ASR windows painted with their z; uncovered frames keep FILL_UNCOVERED (as spvl.py)
        curve = np.full(L, FILL_UNCOVERED, dtype=float)
        for (a, b), z in zip(wins, z_win):
            i0, i1 = max(0, int(math.floor(a * FPS))), min(L, int(math.ceil(b * FPS)))
            if i1 > i0:
                curve[i0:i1] = np.maximum(curve[i0:i1], z)
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
    ap.add_argument("--branches", choices=["dual", "joint"], default="dual",
                    help="dual (default): a visual and a speech branch per window; joint: one branch per window "
                         "(experiments/20260928_infer README §11)")
    ap.add_argument("--isolation", choices=["copy", "sequential"], default="copy",
                    help="copy (default): each branch on a deep copy of the prefix cache; sequential: branches in "
                         "window order on one cache that keeps their tokens, so later branches see earlier ones")
    ap.add_argument("--no-transcript-context", action="store_true",
                    help="ablation (20260928_infer README §12): no transcript in the prefix")
    ap.add_argument("--stance", choices=["verdict", "none"], default="verdict",
                    help="verdict (default): the model's own whole-video answer precedes the window branches; "
                         "none: ablation without that turn")
    ap.add_argument("--windows", choices=["fixed", "asr"], default="fixed",
                    help="fixed (default): the 8 s grid; asr: ASR segments as windows (ablation; gaps unscored)")
    args = ap.parse_args()
    torch.manual_seed(SEED)
    out_dir = ROOT / "runs" / args.exp_id / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out_dir / "run.log"), logging.StreamHandler(sys.stdout)])
    logging.info("host %s", socket.gethostname())
    (out_dir / "run.pid").write_text(str(os.getpid()))
    args.method_name = f"til_measure_w{args.window_seconds:g}_off{args.window_offset:g}" + (
        "" if args.branches == "dual" and args.isolation == "copy" else f"_{args.branches}_{args.isolation}") + (
        "" if args.frames == 20 else f"_f{args.frames}") + ("_noctx" if args.no_transcript_context else "") + (
        "_nostance" if args.stance == "none" else "") + ("_asr" if args.windows == "asr" else "")
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
