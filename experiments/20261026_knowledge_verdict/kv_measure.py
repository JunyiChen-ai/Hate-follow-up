#!/usr/bin/env python3
"""Candidate 51, reader side: the r6 read (experiments/20260922_til/til_measure.py score_video, copied here because
experiment directories must not import each other) with the agent's knowledge inserted.

Arms (README §3): base, K (notes in the prefix + window notes), Kp (prefix notes only), KJ (notes + judgment in the
prefix, window notes), J (judgment only), J0 (the judgment written without the notes), Kother (another subset video's
notes, fixed derangement, seed 0, times scaled to this video), Kshift (K with the window notes moved by half the video),
Kmark (K with the window notes' content withheld, time spans kept).
Optional --stance-override FILE (json {"<dataset>/<video_id>": "Yes"|"No"}) replaces the model's own stance turn for the
listed videos (verdict-oracle diagnostic); z_video itself is still read before the stance turn.
Labels are never read here. Output: runs/20261026_knowledge_verdict/<run-name>_<arm>/predictions.jsonl.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (EXP, ROOT, load_pack, prefix_insert, rescaled_pack, shifted_notes, subset_rows,  # noqa: E402
                    window_insert, window_notes)

sys.path.insert(0, str(ROOT))
from src.mllm_judge import Judge, MODEL, VIDEO_QUESTION, yesno_question  # noqa: E402
from src.video_inputs import FPS, frame_paths, load_asr, window_text  # noqa: E402

CODE_PATH = "experiments/20261026_knowledge_verdict/kv_measure.py"
SEED = 0
FILL_UNCOVERED = -12.0
ARMS = {  # arm: (notes in prefix, judgment in prefix, window insert, pack of another video)
    "base": (False, None, None, False),
    "K": (True, None, "own", False),
    "Kp": (True, None, None, False),
    "KJ": (True, "judgment", "own", False),
    "J": (False, "judgment", None, False),
    "J0": (False, "judgment0", None, False),
    "Kother": (True, None, "own", True),
    "Kshift": (True, None, "shift", False),
    "Kmark": (True, None, "mark", False),
}


def grid_windows(duration, seconds, offset):
    wins = []
    t = float(offset)
    while t < duration - 1e-9:
        wins.append((t, min(t + seconds, duration)))
        t += seconds
    if not wins:
        wins = [(0.0, duration)]
    return wins


def derangement(n, seed=0):
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(n)
        if n < 2 or not np.any(p == np.arange(n)):
            return p


def score_video(judge, row, segments, pack, arm, stance_override, verify=False, window_seconds=8.0):
    with_notes, judgment_key, window_mode, _ = ARMS[arm]
    vid, ds, dur = row["video_id"], row["dataset"], float(row["duration"])
    wins = grid_windows(dur, window_seconds, 0.0)
    wtexts = [window_text(segments, a, b) for a, b in wins]
    frames = frame_paths(ds, vid, 20, "k20")
    if not frames:
        return None, {"error": "no frames cached"}
    msgs, image_files = judge.prefix_messages(frames, segments, with_context=True, with_frames=True)
    insert = prefix_insert(pack, with_notes, judgment_key)
    if insert:
        msgs = judge._append_text(msgs, insert)
    prefix_text, enc = judge.encode_prefix(msgs, image_files)
    prefix_ids = enc["input_ids"][0].tolist()
    info = {"prefix_tokens": len(prefix_ids), "n_windows": len(wins), "n_frames": len(frames), "arm": arm,
            "prefix_insert_chars": len(insert), "n_notes": len((pack or {}).get("notes") or []) if with_notes else 0,
            "judgment_in_prefix": bool(judgment_key and (pack or {}).get(judgment_key))}
    cache = judge.prefix_cache(enc)
    b0, b0_text = judge.branch_ids(msgs, VIDEO_QUESTION)
    if verify:
        judge.seam_check_tokens(msgs, image_files, prefix_ids, VIDEO_QUESTION, b0)
    z_video = judge.cached_margin(cache, b0, in_place=True)
    stance = "Yes" if z_video > 0 else "No"
    forced = stance_override.get(f"{ds}/{vid}")
    if forced is not None:
        stance = forced
    a0, a0_text = judge.answer_ids(msgs, VIDEO_QUESTION, stance)
    judge.extend_cache(cache, a0)
    history = [{"role": "user", "content": [{"type": "text", "text": VIDEO_QUESTION}]}, judge.turn("assistant", stance)]
    head = prefix_text + b0_text + a0_text
    notes = (pack or {}).get("notes") or []
    if window_mode == "shift":
        notes = shifted_notes(notes, dur)
    per = [dict() for _ in wins]
    n_branch, noted = 0, 0
    order = [(i, k) for k in ("visual", "speech") for i in range(len(wins))]
    order = [(i, k) for i, k in order if not (k == "speech" and not (wtexts[i] and wtexts[i].strip()))]
    for i, kind in order:
        (a, b), t = wins[i], wtexts[i]
        q = yesno_question(i, len(wins), a, b, t, kind)
        if window_mode:
            w_ins = window_insert(notes, a, b, last=i == len(wins) - 1, mark=window_mode == "mark")
            if w_ins:
                q = w_ins + q
                noted += kind == "visual"
        bids, _ = judge.branch_ids(msgs, q, history, head_text=head)
        per[i][kind] = judge.cached_margin(cache, bids, in_place=False)
        n_branch += 1
    z_win = [max(d.values()) if d else FILL_UNCOVERED for d in per]
    del cache
    info.update({"n_branches": n_branch, "z_video": z_video, "stance": stance, "stance_forced": forced is not None,
                 "windows_with_notes": noted})
    if verify:
        z_ref = judge.plain_margin(msgs, image_files, VIDEO_QUESTION)
        info["verify"] = {"video_q_cache_vs_plain_dz": abs(z_video - z_ref), "peak_mem_GB": torch.cuda.max_memory_allocated() / 1e9}
        if abs(z_video - z_ref) >= 3.0:
            raise SystemExit(f"VERIFY GATE FAILED: {info['verify']}")
    L = int(math.ceil(dur * FPS))
    centers = (np.arange(L) + 0.5) / FPS
    idx = np.clip(np.floor(centers / window_seconds).astype(int), 0, len(wins) - 1)
    curve = np.asarray(z_win, dtype=float)[idx]
    pred = {"schema_version": 1, "method": f"kv_{arm}", "dataset": ds, "video_id": vid, "duration": dur,
            "native_rate": FPS, "score_curve": [float(x) for x in curve], "intervals": [], "error": None,
            "calls": 2, "seed": SEED, "code_path": CODE_PATH,
            "extra": {"z_video": z_video, "stance": stance, "window_offset": 0.0,
                      "windows": [{"i": i, "start": a, "end": b, "z": z, **{f"z_{k}": v for k, v in per[i].items()},
                                   "n_notes": len(window_notes(notes, a, b, i == len(wins) - 1)) if window_mode else 0}
                                  for i, ((a, b), z) in enumerate(zip(wins, z_win))],
                      **{k: v for k, v in info.items() if k != "verify"}}}
    return pred, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="read")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg", "DeHate"])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stance-override", default="")
    args = ap.parse_args()
    if args.limit and args.run_name == "read":
        raise SystemExit("a --limit smoke test needs a --run-name other than 'read' (resume would keep its rows)")
    torch.manual_seed(SEED)
    base = ROOT / "runs" / EXP
    out_dirs = {a: base / f"{args.run_name}_{a}" for a in args.arms}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    log_dir = base / args.run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(log_dir / "run.log"), logging.StreamHandler(sys.stdout)])
    logging.info("host %s  code %s", socket.gethostname(), CODE_PATH)
    (log_dir / "run.pid").write_text(str(os.getpid()))
    override = json.loads(Path(args.stance_override).read_text()) if args.stance_override else {}
    judge = Judge(model_id=args.model)
    import transformers
    cfg = dict(vars(args))
    cfg.update({"code_path": CODE_PATH, "date": time.strftime("%Y-%m-%d"), "host": socket.gethostname(), "seed": SEED,
                "arms_def": {a: ARMS[a] for a in args.arms}, "video_question": VIDEO_QUESTION,
                "transformers": transformers.__version__, "torch": torch.__version__, "model_id": args.model,
                "n_override": len(override)})
    (log_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    jobs = []
    for ds in args.datasets:
        rows = subset_rows(ds)
        if args.limit:
            rows = rows[:args.limit]
        packs = [load_pack(ds, r["video_id"]) for r in rows]
        perm = derangement(len(rows))
        asr = load_asr(ds)
        for j, r in enumerate(rows):
            jobs.append((r, asr.get(r["video_id"], []), packs[j], rescaled_pack(packs[perm[j]], float(r["duration"])),
                         rows[perm[j]]["video_id"]))
    missing = sum(p is None for _, _, p, _, _ in jobs)
    logging.info("videos %d  arms %s  missing packs %d  overrides %d", len(jobs), args.arms, missing, len(override))
    done = {a: set() for a in args.arms}
    for a in args.arms:
        p = out_dirs[a] / "predictions.jsonl"
        if p.exists():
            done[a] = {(r["dataset"], r["video_id"]) for r in map(json.loads, open(p)) if not r.get("error")}
    fhs = {a: open(out_dirs[a] / "predictions.jsonl", "a") for a in args.arms}
    n_err, t0 = 0, time.time()
    for n, (row, segments, pack, other, other_vid) in enumerate(jobs):
        for a in args.arms:
            key = (row["dataset"], row["video_id"])
            if key in done[a]:
                continue
            use = other if ARMS[a][3] else pack
            try:
                pred, info = score_video(judge, row, segments, use, a, override, verify=(n == 0))
                if pred is None:
                    raise RuntimeError(info.get("error", "unknown"))
                if ARMS[a][3]:
                    pred["extra"]["pack_from"] = other_vid
                if "verify" in info:
                    logging.info("VERIFY %s %s %s", a, row["video_id"], json.dumps(info["verify"]))
                fhs[a].write(json.dumps(pred, ensure_ascii=False) + "\n")
                fhs[a].flush()
            except Exception as exc:
                n_err += 1
                torch.cuda.empty_cache()
                logging.exception("FAILED %s %s: %s", a, row["video_id"], exc)
                fhs[a].write(json.dumps({"method": f"kv_{a}", "dataset": row["dataset"], "video_id": row["video_id"],
                                         "score_curve": [], "intervals": [], "error": f"{type(exc).__name__}: {exc}"}) + "\n")
        if (n + 1) % 10 == 0:
            logging.info("progress %d/%d  %.1fs/video  errors %d", n + 1, len(jobs), (time.time() - t0) / (n + 1), n_err)
    logging.info("DONE videos=%d errors=%d elapsed=%.0fs", len(jobs), n_err, time.time() - t0)


if __name__ == "__main__":
    main()
