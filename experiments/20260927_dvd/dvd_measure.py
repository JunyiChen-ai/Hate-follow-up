#!/usr/bin/env python3
"""DVD reads (README.md §3): per video, the SPVL-r2 prefix (rules + 20 frames + transcript) and four isolated Yes/No
branches on it: the existing whole-video verdict (a check against the cached reads) and three definition conditions,
T (protected-group target), E (endorsement), A (attack; diagnostic arm only). Each branch runs on a deep copy of the
prefix cache, so the branches do not see each other. Labels are never read here.
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

CODE_PATH = "experiments/20260927_dvd/dvd_measure.py"
SEED = 0
ANSWER = '\n\nAnswer "Yes" or "No".'
QUESTIONS = {
    "z_target": ("Is the hostile or demeaning content in this video aimed at a group of people defined by a protected "
                 "attribute (race, ethnicity, national origin, religion, caste, sex or gender, gender identity, sexual "
                 "orientation, disability, or immigration status), or at people because they belong to such a group?"
                 "\n\nAnswer \"No\" if it is aimed only at specific individuals, public figures, officials, "
                 "organizations, or groups not defined by a protected attribute, or if the video has no hostile content."
                 + ANSWER),
    "z_endorse": ("Does the creator or speaker of this video endorse or promote the hateful message themselves?"
                  "\n\nAnswer \"No\" if the video reports on, quotes, condemns, satirizes, or discusses hateful content "
                  "without endorsing it, if slurs appear only as in-group or reclaimed usage (for example in music) "
                  "without hostility toward the group, or if the video has no hateful message." + ANSWER),
    "z_attack": ("Does this video attack, demean, dehumanize, threaten, or express contempt for people?"
                 "\n\nAnswer \"No\" for neutral discussion, news reporting, or criticism of ideas or actions that does "
                 "not show contempt for people." + ANSWER),
}


def score_video(judge, row, segments, frames_k, verify=False):
    frames = frame_paths(row["dataset"], row["video_id"], frames_k, "k20")
    if not frames:
        return None, {"error": "no frames cached"}
    msgs, image_files = judge.prefix_messages(frames, segments, with_context=True, with_frames=True)
    prefix_text, enc = judge.encode_prefix(msgs, image_files)
    prefix_ids = enc["input_ids"][0].tolist()
    cache = judge.prefix_cache(enc)
    out = {}
    for name, q in [("z_video", VIDEO_QUESTION)] + list(QUESTIONS.items()):
        bids, _ = judge.branch_ids(msgs, q)
        if verify:
            judge.seam_check_tokens(msgs, image_files, prefix_ids, q, bids)
        out[name] = judge.cached_margin(cache, bids, in_place=False)
    del cache
    info = {"prefix_tokens": len(prefix_ids), "n_frames": len(frames)}
    if verify:
        z_ref = judge.plain_margin(msgs, image_files, QUESTIONS["z_target"])
        info["verify"] = {"target_cache_vs_plain_dz": abs(out["z_target"] - z_ref),
                          "peak_mem_GB": torch.cuda.max_memory_allocated() / 1e9}
        if abs(out["z_target"] - z_ref) >= 3.0:
            raise SystemExit(f"VERIFY GATE FAILED: {info['verify']}")
    return {"dataset": row["dataset"], "video_id": row["video_id"], "error": None, **out, "extra": info}, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--exp-id", default="20260927_dvd")
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"])
    ap.add_argument("--manifest", default=str(ROOT / "data/omsl_v6_inputs/manifests/all_test.jsonl"))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(SEED)
    out_dir = ROOT / "runs" / args.exp_id / args.run_name
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
         "seed": SEED, "video_question": VIDEO_QUESTION, "questions": QUESTIONS, "transformers": transformers.__version__,
         "torch": torch.__version__}, indent=2, ensure_ascii=False))
    pred_path = out_dir / "reads.jsonl"
    done = set()
    if pred_path.exists():
        done = {(r["dataset"], r["video_id"]) for r in map(json.loads, open(pred_path)) if not r.get("error")}
    logging.info("videos %d (already done %d)", len(rows), len(done))
    n_err, t0 = 0, time.time()
    with open(pred_path, "a") as fh:
        for n, row in enumerate(rows):
            if (row["dataset"], row["video_id"]) in done:
                continue
            try:
                rec, info = score_video(judge, row, asr[row["dataset"]].get(row["video_id"], []), args.frames,
                                        verify=(n == 0))
                if rec is None:
                    raise RuntimeError(info.get("error", "unknown"))
                if "verify" in info:
                    logging.info("VERIFY %s %s", row["video_id"], json.dumps(info["verify"]))
                fh.write(json.dumps(rec) + "\n"); fh.flush()
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache(); n_err += 1
                logging.error("OOM %s %s", row["video_id"], str(exc)[:200])
                fh.write(json.dumps({"dataset": row["dataset"], "video_id": row["video_id"], "error": "OOM"}) + "\n")
            except Exception as exc:  # noqa: BLE001
                n_err += 1
                logging.exception("FAILED %s: %s", row["video_id"], exc)
                fh.write(json.dumps({"dataset": row["dataset"], "video_id": row["video_id"],
                                     "error": f"{type(exc).__name__}: {exc}"}) + "\n")
            if (n + 1) % 20 == 0:
                logging.info("progress %d/%d  %.1fs/video  errors %d", n + 1, len(rows), (time.time() - t0) / (n + 1), n_err)
    logging.info("DONE videos=%d errors=%d elapsed=%.0fs", len(rows), n_err, time.time() - t0)


if __name__ == "__main__":
    main()
