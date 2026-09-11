#!/usr/bin/env python3
"""TAD: separate topic from act in the per-window judgement (label-free).

SPVL-r2 unchanged, plus one extra short read per window: a topic question asking only whether the window
refers to a group defined by a protected characteristic, explicitly ignoring whether the reference is
hostile. The corrected window score is a_i - beta * t_i with beta from this video's own two score vectors.

Writes runs/<exp_id>/<run_name>/predictions.jsonl with score_curve = the raw act curve (a) and
extra.windows carrying a / t / visual / speech, so tad_compose.py can build every declared variant
without re-running inference. No labels are read in this file.
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
CODE_PATH = "experiments/20260912_tad/tad.py"
sys.path.insert(0, str(ROOT))
from src.mllm_judge import Judge, MODEL, VIDEO_QUESTION  # noqa: E402
from src.video_inputs import (FPS, fixed_windows, frame_paths, load_asr, load_manifest,  # noqa: E402
                              window_text, within_defined_ids)

SEED = 0
FILL_UNCOVERED = -12.0

TOPIC_QUESTION = (
    "Does THIS window refer to, mention, or depict a group defined by a protected characteristic "
    "(such as race, ethnicity, national origin, religion, gender, sexual orientation, disability, "
    "or immigration status)? Ignore whether the reference is hostile, neutral, or supportive; answer "
    "only about whether such a group is referred to."
)


# Round 2: the window decision as a speech act towards the group, not a binary rule violation.
# The five options are the confusion set the hate definition itself carves out: the 2026-08 policy text
# says "Quotation, neutral reporting, counterspeech, satire, and condemnation are not endorsement".
ACTS = ("attacks", "reports", "quotes", "condemns", "unrelated")
ACT_QUESTION = (
    "In THIS window, what does the video do with respect to a group defined by a protected "
    "characteristic (race, ethnicity, national origin, religion, gender, sexual orientation, "
    "disability, immigration status)?\n"
    "  attacks   - it attacks, dehumanises, threatens, excludes or stereotypes such a group, and the "
    "video endorses this\n"
    "  reports   - it neutrally describes or reports on such a group or on someone else's attack\n"
    "  quotes    - it quotes or shows someone else's words without endorsing them\n"
    "  condemns  - it criticises or condemns such an attack\n"
    "  unrelated - it does not refer to such a group at all\n\n"
    "Answer with exactly one word from: attacks, reports, quotes, condemns, unrelated."
)


def window_head(i, n, t1, t2):
    return f"Consider only window {i + 1} of {n}, from {t1:.1f}s to {t2:.1f}s of this video."


def act_type_question(i, n, t1, t2, text):
    body = text.strip() if text and text.strip() else "(no speech)"
    return (f"{window_head(i, n, t1, t2)} Transcript in this window: {body}\n"
            f"Look at the frames whose timestamps fall inside this window.\n\n{ACT_QUESTION}")


def act_question(i, n, t1, t2, text, kind):
    """The SPVL-r2 evidence question, visual or speech branch (verbatim wording)."""
    body = text.strip() if text and text.strip() else "(no speech)"
    head = window_head(i, n, t1, t2)
    if kind == "visual":
        ctx = (f"{head} Look only at the frames whose timestamps fall inside this window and judge the visual "
               f"content alone (imagery, gestures, symbols, on-screen text), ignoring the speech.\n\n")
        q = "Is THIS window one of the segments where visual content that violates the above rules occurs?"
    elif kind == "speech":
        ctx = f"{head} Judge only what is spoken in this window: {body}\n\n"
        q = "Is THIS window one of the segments where speech that violates the above rules occurs?"
    else:
        raise ValueError(kind)
    return ctx + q + '\n\nAnswer "Yes" or "No".'


def topic_question(i, n, t1, t2, text):
    body = text.strip() if text and text.strip() else "(no speech)"
    return (f"{window_head(i, n, t1, t2)} Transcript in this window: {body}\n"
            f"Look at the frames whose timestamps fall inside this window.\n\n"
            f"{TOPIC_QUESTION}\n\nAnswer \"Yes\" or \"No\".")


def score_video(judge, row, segments, args, verify=False):
    vid, ds, dur = row["video_id"], row["dataset"], float(row["duration"])
    wins = fixed_windows(dur, args.window_seconds)
    wtexts = [window_text(segments, a, b) for a, b in wins]
    frames = frame_paths(ds, vid, args.frames, "k20") if args.frames > 0 else []
    if args.frames > 0 and not frames:
        return None, {"error": "no frames cached"}
    msgs, image_files = judge.prefix_messages(frames, segments, with_context=True, with_frames=args.frames > 0)
    prefix_text, enc = judge.encode_prefix(msgs, image_files)
    prefix_ids = enc["input_ids"][0].tolist()
    cache = judge.prefix_cache(enc)

    b0, b0_text = judge.branch_ids(msgs, VIDEO_QUESTION)
    if verify:
        judge.seam_check_tokens(msgs, image_files, prefix_ids, VIDEO_QUESTION, b0)
    z_video = judge.cached_margin(cache, b0, in_place=True)
    stance = "Yes" if z_video > 0 else "No"
    a0, a0_text = judge.answer_ids(msgs, VIDEO_QUESTION, stance)
    judge.extend_cache(cache, a0)
    history = [{"role": "user", "content": [{"type": "text", "text": VIDEO_QUESTION}]},
               judge.turn("assistant", stance)]
    head = prefix_text + b0_text + a0_text

    n = len(wins)
    act_ids = getattr(judge, "_act_ids", None)
    if args.acts and act_ids is None:
        act_ids = [judge.label_ids(w) for w in ACTS]
        flat = [t for c in act_ids for t in c]
        if len(flat) != len(set(flat)):
            raise SystemExit(f"act token sets are not disjoint: {dict(zip(ACTS, act_ids))}")
        judge._act_ids = act_ids
        logging.info("act ids %s", dict(zip(ACTS, act_ids)))
    per = []
    for i, ((t1, t2), txt) in enumerate(zip(wins, wtexts)):
        rec = {"i": i, "start": t1, "end": t2}
        for kind in ("visual", "speech"):
            if kind == "speech" and not (txt and txt.strip()):
                continue
            if kind == "visual" and args.frames == 0:
                continue
            q = act_question(i, n, t1, t2, txt, kind)
            bids, _ = judge.branch_ids(msgs, q, history, head_text=head)
            rec[f"z_{kind}"] = judge.cached_margin(cache, bids)
        vals = [rec[k] for k in ("z_visual", "z_speech") if k in rec]
        rec["a"] = max(vals) if vals else FILL_UNCOVERED
        if args.topic:
            q = topic_question(i, n, t1, t2, txt)
            bids, _ = judge.branch_ids(msgs, q, history, head_text=head)
            rec["t"] = judge.cached_margin(cache, bids)
        if args.acts:
            q = act_type_question(i, n, t1, t2, txt)
            bids, _ = judge.branch_ids(msgs, q, history, head_text=head)
            lp = judge.cached_choices(cache, bids, act_ids)
            rec["acts"] = {k: float(v) for k, v in zip(ACTS, lp)}
            other = [v for k, v in zip(ACTS, lp) if k != "attacks"]
            m = max(other)
            rec["act_margin"] = float(lp[0] - (m + math.log(sum(math.exp(v - m) for v in other))))
        rec["z"] = rec["a"]  # score_curve is the act curve; tad_compose builds the corrected variants
        per.append(rec)

    info = {"prefix_tokens": len(prefix_ids), "n_windows": n, "img_tokens": judge.img_tokens[:1],
            "n_reads": 1 + sum(len([k for k in r if k.startswith("z_")]) + int("t" in r) for r in per)}
    if verify:
        z_ref = judge.plain_margin(msgs, image_files, VIDEO_QUESTION)
        info["verify"] = {"video_q_cache_vs_plain_dz": abs(z_video - z_ref),
                          "peak_mem_GB": torch.cuda.max_memory_allocated() / 1e9}
        if abs(z_video - z_ref) >= 3.0:
            raise SystemExit(f"VERIFY GATE FAILED: {info['verify']}")
    del cache

    L = int(math.ceil(dur * FPS))
    centers = (np.arange(L) + 0.5) / FPS
    idx = np.minimum((centers // args.window_seconds).astype(int), n - 1)
    curve = np.asarray([r["a"] for r in per], dtype=float)[idx]
    pred = {"schema_version": 1, "method": args.method_name, "dataset": ds, "video_id": vid,
            "duration": dur, "native_rate": FPS, "score_curve": [float(x) for x in curve], "intervals": [],
            "error": None, "calls": 2, "seed": SEED, "code_path": CODE_PATH,
            "extra": {"z_video": z_video, "stance": stance, "windows": per,
                      **{k: v for k, v in info.items() if k != "verify"}}}
    return pred, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--exp-id", default="20260912_tad")
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"])
    ap.add_argument("--manifest", default=str(ROOT / "data/omsl_v6_inputs/manifests/all_test.jsonl"))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--window-seconds", type=float, default=8.0)
    ap.add_argument("--topic", type=int, default=1, help="0 reproduces SPVL-r2 (act branches only)")
    ap.add_argument("--acts", type=int, default=0, help="1 adds the speech-act read (round 2)")
    ap.add_argument("--only-within-defined", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--method-name", default=None)
    args = ap.parse_args()
    torch.manual_seed(SEED)

    out_dir = ROOT / "runs" / args.exp_id / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out_dir / "run.log"), logging.StreamHandler(sys.stdout)])
    logging.info("host %s", socket.gethostname())
    (out_dir / "run.pid").write_text(str(os.getpid()))
    args.method_name = args.method_name or (f"tad_f{args.frames}_w{args.window_seconds:g}"
                                            f"_topic{args.topic}_acts{args.acts}")
    cfg = dict(vars(args))
    cfg.update({"code_path": CODE_PATH, "date": time.strftime("%Y-%m-%d"), "host": socket.gethostname(),
                "seed": SEED, "topic_question": TOPIC_QUESTION, "video_question": VIDEO_QUESTION,
                "act_question": ACT_QUESTION, "acts": list(ACTS)})
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    rows = load_manifest(args.manifest, args.datasets)
    if args.only_within_defined:
        keep = within_defined_ids(args.datasets)
        rows = [r for r in rows if (r["dataset"], r["video_id"]) in keep]
    rows.sort(key=lambda r: (r["dataset"], r["video_id"]))
    if args.limit:
        rows = rows[:args.limit]
    asr = {ds: load_asr(ds) for ds in args.datasets}

    judge = Judge(model_id=args.model)
    done = set()
    pred_path = out_dir / "predictions.jsonl"
    if pred_path.exists():  # resume; rows that carry an error are re-scored
        for line in open(pred_path):
            r = json.loads(line)
            if not r.get("error"):
                done.add((r["dataset"], r["video_id"]))
    fh = open(pred_path, "a")
    t0, n_ok = time.time(), 0
    for k, row in enumerate(rows):
        if (row["dataset"], row["video_id"]) in done:
            continue
        try:
            pred, info = score_video(judge, row, asr[row["dataset"]].get(row["video_id"], []), args,
                                     verify=(n_ok == 0))
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            logging.exception("video %s failed: %s", row["video_id"], exc)
            fh.write(json.dumps({"dataset": row["dataset"], "video_id": row["video_id"],
                                 "error": str(exc), "method": args.method_name}) + "\n")
            fh.flush()
            continue
        if pred is None:
            fh.write(json.dumps({"dataset": row["dataset"], "video_id": row["video_id"],
                                 "error": info.get("error"), "method": args.method_name}) + "\n")
            fh.flush()
            continue
        if info.get("verify"):
            (out_dir / "verify.json").write_text(json.dumps(info["verify"], indent=2))
        fh.write(json.dumps(pred, ensure_ascii=False) + "\n")
        fh.flush()
        n_ok += 1
        if k % 10 == 0:
            logging.info("%d/%d %s windows=%d reads=%d %.0fs", k + 1, len(rows), row["video_id"],
                         info.get("n_windows", 0), info.get("n_reads", 0), time.time() - t0)
    fh.close()
    logging.info("DONE %d videos in %.0fs (%.2f s/video)", n_ok, time.time() - t0,
                 (time.time() - t0) / max(n_ok, 1))


if __name__ == "__main__":
    main()
