#!/usr/bin/env python3
"""Video-level ordinal read (README §10, brief 4 H2): on the adopted Reader's prefix (rules + 20 frames + transcript,
unchanged), isolated Yes/No branches per video: the verdict question (`z_video`, check against the stored reads), the
protected-group scope question (`z_target`, check against runs/20260927_dvd) and ten ordinal questions "is the
hatefulness of this video at least tau", tau = 0.1 .. 1.0 (`m_ord`, ten Yes - No margins, same read-out as z_video).
Each branch is read on its own deep copy of the prefix cache (one by one, no batching). The probing idea is taken from
Probe-VAD (arXiv 2609.17211), applied to the whole video on our prefix. No label is read.

Derived quantities (CPU, gate_table.py): p_tau = sigmoid(m_tau), non-increasing projection over tau, s = mean,
z_ord = logit(clip(s, 1e-4, 1 - 1e-4)); s_raw = mean of the unprojected p_tau; z_mean = mean of the ten margins.
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

CODE_PATH = "experiments/20261114_exp_gates/ordinal_read.py"
SEED = 0
# Copied from scope_read.py SCOPE_QUESTION (= experiments/20260927_dvd z_target question); experiment files do not import
# each other's modules beyond this directory, and the text must stay identical.
SCOPE_QUESTION = ("Is the hostile or demeaning content in this video aimed at a group of people defined by a protected "
                  "attribute (race, ethnicity, national origin, religion, caste, sex or gender, gender identity, sexual "
                  "orientation, disability, or immigration status), or at people because they belong to such a group?"
                  "\n\nAnswer \"No\" if it is aimed only at specific individuals, public figures, officials, "
                  "organizations, or groups not defined by a protected attribute, or if the video has no hostile content."
                  "\n\nAnswer \"Yes\" or \"No\".")
# Text fixed by the main session (brief 4, 2026-11-15) before the run.
ORD_TEMPLATE = ("Use an ordered hatefulness scale from 0 to 1: 0 means ordinary non-hateful content; 0.5 means clearly "
                "concerning or plausibly hateful content; 1 means unmistakably severe, explicit hateful content. "
                "Intermediate values preserve this order.\n\nOn this scale, is the hatefulness of this video at least "
                "{tau:.1f}?\n\nAnswer \"Yes\" or \"No\".")
TAUS = [k / 10 for k in range(1, 11)]


@torch.no_grad()
def score_video(judge, row, segments, frames_k):
    frames = frame_paths(row["dataset"], row["video_id"], frames_k, "k20")
    if not frames:
        return {"dataset": row["dataset"], "video_id": row["video_id"], "error": "no frames cached"}
    judge.model.model.rope_deltas = None
    msgs, image_files = judge.prefix_messages(frames, segments, with_context=True, with_frames=True)
    _, enc = judge.encode_prefix(msgs, image_files)
    cache = judge.prefix_cache(enc)
    out = {}
    for name, q in (("z_video", VIDEO_QUESTION), ("z_target", SCOPE_QUESTION)):
        bids, _ = judge.branch_ids(msgs, q)
        out[name] = judge.cached_margin(cache, bids, in_place=False)
    m = []
    for tau in TAUS:
        bids, _ = judge.branch_ids(msgs, ORD_TEMPLATE.format(tau=tau))
        m.append(judge.cached_margin(cache, bids, in_place=False))
    n_tok = int(enc["input_ids"].shape[1])
    del cache
    return {"dataset": row["dataset"], "video_id": row["video_id"], "error": None, **out, "m_ord": m,
            "extra": {"prefix_tokens": n_tok, "n_frames": len(frames), "variant": "full"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"])
    ap.add_argument("--manifest", default=str(ROOT / "data/omsl_v6_inputs/manifests/all_test.jsonl"))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stored", nargs="+", required=True,
                    help="stored full-reader predictions.jsonl for the z_video check (|diff| <= 1e-3)")
    ap.add_argument("--out-root", default=str(ROOT / "runs/20261114_exp_gates/ordinal_reads"))
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
    stored = {}
    for p in args.stored:
        for r in map(json.loads, open(ROOT / p)):
            if not r.get("error"):
                stored[(r["dataset"], r["video_id"])] = r["extra"]["z_video"]
    asr = {ds: load_asr(ds) for ds in args.datasets}
    judge = Judge(model_id=args.model)
    import transformers
    (out_dir / "config.json").write_text(json.dumps(
        {**vars(args), "code_path": CODE_PATH, "date": time.strftime("%Y-%m-%d"), "host": socket.gethostname(),
         "seed": SEED, "video_question": VIDEO_QUESTION, "scope_question": SCOPE_QUESTION,
         "ordinal_questions": [ORD_TEMPLATE.format(tau=t) for t in TAUS],
         "transformers": transformers.__version__, "torch": torch.__version__}, indent=2, ensure_ascii=False))
    pred_path = out_dir / "reads.jsonl"
    done = set()
    if pred_path.exists():
        done = {(r["dataset"], r["video_id"]) for r in map(json.loads, open(pred_path)) if not r.get("error")}
    logging.info("videos %d (already done %d) model %s", len(rows), len(done), args.model)
    n_err, n_chk, worst, t0 = 0, 0, 0.0, time.time()
    with open(pred_path, "a") as fh:
        for n, row in enumerate(rows):
            if (row["dataset"], row["video_id"]) in done:
                continue
            try:
                rec = score_video(judge, row, asr[row["dataset"]].get(row["video_id"], []), args.frames)
            except Exception as e:  # noqa: BLE001  (one failed video must not stop the corpus; it is counted)
                rec = {"dataset": row["dataset"], "video_id": row["video_id"], "error": f"{type(e).__name__}: {e}"}
            key = (row["dataset"], row["video_id"])
            if not rec.get("error") and key in stored:
                d = abs(rec["z_video"] - stored[key]); worst = max(worst, d); n_chk += 1
                if d > 1e-3:
                    rec["error"] = f"z_video check failed: |{rec['z_video']:.4f} - {stored[key]:.4f}| > 1e-3"
            n_err += bool(rec.get("error"))
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if n % 20 == 0:
                logging.info("progress %d/%d  %.1fs/video  errors %d  z_video check max |diff| %.2e over %d",
                             n, len(rows), (time.time() - t0) / max(1, n + 1), n_err, worst, n_chk)
    logging.info("DONE videos=%d errors=%d z_video check max |diff| %.2e over %d videos elapsed=%.0fs",
                 len(rows), n_err, worst, n_chk, time.time() - t0)
    if n_err:
        raise SystemExit(f"FAILED {n_err} videos with errors")


if __name__ == "__main__":
    main()
