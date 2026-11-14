#!/usr/bin/env python3
"""Scope read (README.md §3): per video, one isolated Yes/No branch on the Reader's prefix that asks whether the
hostile content is aimed at a protected group. The question text is the `z_target` question of
experiments/20260927_dvd/dvd_measure.py, copied here (experiment directories do not import each other).

Prefix variants (the prefixes of the reader ablation arms, src/reader_arms.build_context):
  full            rules + 20 frames + transcript (the adopted reader's prefix)
  no_transcript   prefix without the whole transcript
  no_prefix_frames  prefix without the 20 frames
Each record also holds the whole-video verdict read on the same prefix (`z_video`), as a check against the stored
reads of the corresponding arm. Labels are never read here.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.mllm_judge import Judge, MODEL, VIDEO_QUESTION  # noqa: E402
from src.video_inputs import frame_paths, load_asr, load_manifest  # noqa: E402

CODE_PATH = "experiments/20261114_exp_gates/scope_read.py"
SEED = 0
SCOPE_QUESTION = ("Is the hostile or demeaning content in this video aimed at a group of people defined by a protected "
                  "attribute (race, ethnicity, national origin, religion, caste, sex or gender, gender identity, sexual "
                  "orientation, disability, or immigration status), or at people because they belong to such a group?"
                  "\n\nAnswer \"No\" if it is aimed only at specific individuals, public figures, officials, "
                  "organizations, or groups not defined by a protected attribute, or if the video has no hostile content."
                  "\n\nAnswer \"Yes\" or \"No\".")
# Question from HateClipSeg's own label dimensions (README §6; scripts/build_gt_4fps_hate_only.py line 10, order
# [Normal, Hateful, Insulting, Sexual, Violence, Self-Harm] from the HateClipSeg paper). Text fixed by the main session
# (brief 3, 2026-11-15) before the run; no glosses.
HCS_TAXONOMY_QUESTION = ("Does this video contain content of any of these kinds: hateful, insulting, sexual, violence, "
                         "or self-harm?\n\nAnswer \"No\" if the video contains none of these.\n\nAnswer \"Yes\" or \"No\".")
QUESTIONS = {"protected": SCOPE_QUESTION, "hcs_taxonomy": HCS_TAXONOMY_QUESTION}
VARIANTS = {"full": (True, True), "no_transcript": (False, True), "no_prefix_frames": (True, False)}


@torch.no_grad()
def score_video(judge, row, segments, frames_k, variant, question=SCOPE_QUESTION):
    frames = frame_paths(row["dataset"], row["video_id"], frames_k, "k20")
    if not frames:
        return {"dataset": row["dataset"], "video_id": row["video_id"], "error": "no frames cached"}
    with_context, with_frames = VARIANTS[variant]
    judge.model.model.rope_deltas = None
    msgs, image_files = judge.prefix_messages(frames if with_frames else [], segments, with_context=with_context,
                                              with_frames=with_frames)
    _, enc = judge.encode_prefix(msgs, image_files)
    cache = judge.prefix_cache(enc)
    out = {}
    for name, q in (("z_video", VIDEO_QUESTION), ("z_target", question)):
        bids, _ = judge.branch_ids(msgs, q)
        out[name] = judge.cached_margin(cache, bids, in_place=False)
    n_tok = int(enc["input_ids"].shape[1])
    del cache
    return {"dataset": row["dataset"], "video_id": row["video_id"], "error": None, **out,
            "extra": {"prefix_tokens": n_tok, "n_frames": len(frames) if with_frames else 0, "variant": variant}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--variant", choices=list(VARIANTS), default="full")
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"])
    ap.add_argument("--manifest", default=str(ROOT / "data/omsl_v6_inputs/manifests/all_test.jsonl"))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--question", choices=list(QUESTIONS), default="protected",
                    help="scope question stored as z_target (hcs_taxonomy: HateClipSeg's own categories)")
    ap.add_argument("--out-root", default=str(ROOT / "runs/20261114_exp_gates/scope_reads"))
    args = ap.parse_args()
    torch.manual_seed(SEED)
    out_dir = Path(args.out_root) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out_dir / "run.log"), logging.StreamHandler(sys.stdout)])
    logging.info("host %s", socket.gethostname())
    (out_dir / "run.pid").write_text(str(os.getpid()))
    rows = load_manifest(args.manifest, args.datasets)
    if args.limit:
        rows = rows[:args.limit]
    asr = {ds: load_asr(ds) for ds in args.datasets}
    judge = Judge(model_id=args.model)
    import transformers
    (out_dir / "config.json").write_text(json.dumps(
        {**vars(args), "code_path": CODE_PATH, "date": time.strftime("%Y-%m-%d"), "host": socket.gethostname(),
         "seed": SEED, "video_question": VIDEO_QUESTION, "scope_question": QUESTIONS[args.question],
         "transformers": transformers.__version__, "torch": torch.__version__}, indent=2, ensure_ascii=False))
    pred_path = out_dir / "reads.jsonl"
    done = set()
    if pred_path.exists():
        done = {(r["dataset"], r["video_id"]) for r in map(json.loads, open(pred_path)) if not r.get("error")}
    logging.info("videos %d (already done %d) variant %s question %s model %s", len(rows), len(done), args.variant,
                 args.question, args.model)
    n_err, t0 = 0, time.time()
    with open(pred_path, "a") as fh:
        for n, row in enumerate(rows):
            if (row["dataset"], row["video_id"]) in done:
                continue
            try:
                rec = score_video(judge, row, asr[row["dataset"]].get(row["video_id"], []), args.frames, args.variant,
                                  QUESTIONS[args.question])
            except Exception as e:  # noqa: BLE001  (one failed video must not stop the corpus; it is counted)
                rec = {"dataset": row["dataset"], "video_id": row["video_id"], "error": f"{type(e).__name__}: {e}"}
            n_err += bool(rec.get("error"))
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if n % 20 == 0:
                logging.info("progress %d/%d  %.1fs/video  errors %d", n, len(rows), (time.time() - t0) / max(1, n + 1), n_err)
    logging.info("DONE videos=%d errors=%d elapsed=%.0fs", len(rows), n_err, time.time() - t0)
    if n_err:
        raise SystemExit(f"FAILED {n_err} videos with errors")


if __name__ == "__main__":
    main()
