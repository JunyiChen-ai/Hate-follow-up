#!/usr/bin/env python3
"""Per-segment presence reads (candidate 49 draft; scout_report3 candidate 1): the r6 verdict question on each of J
equal-duration segments of the video, each segment seen alone.

Per video (duration D):
  - z_video: the r6 verdict on the full r6 prefix (20 uniform frames + full timestamped transcript), read through the
    same subset-prefix path with the full inputs; a check against runs/20260926_glr/base_gridA (DeHate:
    runs/20260927_dehate_external/reads_gridA);
  - J segments [k D / J, (k + 1) D / J). Segment k's prefix is the r6 prompt material unchanged (system message, intro,
    rules, reader instruction; src/mllm_judge.Judge.prefix_messages) with
      frames:     the r6 frames (frames_k20, t_i = (i + .5) D / 20) whose timestamp t satisfies t1 <= t < t2 (the last
                  segment is closed on the right); with 20 frames and J = 4 that is frames 5k..5k+4;
      transcript: the Whisper chunks (src/video_inputs.load_asr, untimed chunks filled as in r6) cut to [t1, t2) by
                  the r6 window-text rule (src/video_inputs.window_text): a chunk inside the segment is kept whole, a
                  chunk crossing a boundary is cut at word boundaries in proportion to time and shown with its
                  clipped times; the last segment extends to the latest chunk end so no word is lost. Adjacent
                  segments therefore split every chunk's words with no loss and no overlap;
      timestamps: absolute (seconds from the video start), as in r6;
    and VIDEO_QUESTION is read as Yes/No log-odds z_k exactly as z_video (own prefix cache per segment, nothing shared
    between segments, fp32 Yes/No sets of src/mllm_judge).
  - m_start[k]: r6 8 s grid windows whose start lies in segment k (a partition of the windows; 0 when the segment is
    shorter than 8 s and holds no window start); m_overlap[k]: grid windows overlapping segment k (>= 1);
    len_windows[k] = segment length / 8 s.
No label file is read. GPU only inside Slurm (launch/lab1_segments.sbatch); --dry-run is CPU only.

  python experiments/20261025_m3_coverage/read_segments.py --run-name <name> [--datasets ...] [--J 4] [--limit N] [--verify]
  python experiments/20261025_m3_coverage/read_segments.py --dry-run --datasets HateMM HateClipSeg DeHate [--dry-out F.jsonl]

--verify: on the first 2 videos of every corpus, the cached read of z_video and of every segment is compared with one
ordinary full call and the token seam is checked (run.log, verify.jsonl); the run stops when |dz| >= 3 or a seam fails.
Output: runs/<exp-id>/<run-name>/{segments.jsonl, config.json, run.log, run.pid, verify.jsonl}; one line per video.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.mllm_judge import MODEL, VIDEO_QUESTION, Judge  # noqa: E402
from src.prefix_questions import (GATE_DZ, MANIFEST, REFERENCE, equal_segments, processor_judge,  # noqa: E402
                                  read_subset_verdict, reference_reads, selected_rows, subset_inputs, subset_seam_ok)
from src.video_inputs import fixed_windows, frame_paths, load_asr, window_text  # noqa: E402

CODE_PATH = "experiments/20261025_m3_coverage/read_segments.py"
IMPLEMENTATION = ("per-segment presence read v1: J equal-duration segments, each with the r6 prompt material, the r6 "
                  "frames inside [t1, t2) and the transcript cut by the r6 window-text rule (absolute timestamps), "
                  "VIDEO_QUESTION read as Yes/No log-odds on its own prefix cache; 2026-10-25")
SEED = 0
NATIVE_FRAMES = 20
WINDOW_S = 8.0
N_VERIFY = 2


def plan(row, segments, frames, J):
    """Segment spans and inputs of one video (CPU)."""
    D = float(row["duration"])
    tend = max([e for _, e, _ in segments] + [D])
    wins = fixed_windows(D, WINDOW_S)
    starts = [a for a, _ in wins]
    out = []
    for k, (t1, t2) in enumerate(equal_segments(D, J)):
        last = k == J - 1
        f, p = subset_inputs(frames, segments, t1, t2, last, tend)
        out.append({"k": k, "t1": t1, "t2": t2, "frames": f, "pieces": p, "frame_times": [t for t, _ in f],
                    "n_frames": len(f), "n_pieces": len(p), "n_words": sum(len(x[2].split()) for x in p),
                    "m_start": sum(1 for a in starts if a >= t1 and (last or a < t2)),
                    "m_overlap": sum(1 for a, b in wins if min(b, t2) - max(a, t1) > 1e-9), "len_windows": (t2 - t1) / WINDOW_S,
                    "text_bound": max(t2, tend) if last else t2})
    return out


def public(seg):
    return {k: v for k, v in seg.items() if k not in ("frames", "pieces", "text_bound")}


@torch.no_grad()
def read_video(j, row, segments, J, verify):
    ds, vid = row["dataset"], row["video_id"]
    frames = frame_paths(ds, vid, NATIVE_FRAMES, "k20")
    if not frames:
        raise RuntimeError("no frames cached")
    full = read_subset_verdict(j, frames, segments, verify)
    segs = plan(row, segments, frames, J)
    out = []
    for s in segs:
        r = read_subset_verdict(j, s["frames"], s["pieces"], verify)
        out.append({**public(s), **r})
    rec = {"z_video": full["z"], "prefix_tokens": full["prefix_tokens"], "n_frames": len(frames),
           "z_seg": [s["z"] for s in out], "segments": [{k: v for k, v in s.items() if k != "verify"} for s in out]}
    if verify:
        rec["verify"] = {"video": full["verify"], "segments": [s["verify"] for s in out],
                         "peak_mem_GB": torch.cuda.max_memory_allocated() / 1e9}
    return rec


def verify_bad(v):
    checks = [v["video"]] + v["segments"]
    return [f"{'video' if i == 0 else f'segment {i - 1}'}: {c}" for i, c in enumerate(checks)
            if not c["seam_ok"] or c["dz"] >= GATE_DZ]


# ----------------------------------------------------------------------------------------------- CPU dry run

def dry_run(args, rows, asr, ref):
    j = processor_judge(args.model)
    qid_cache = {}
    per = []
    shown = False
    for n, row in enumerate(rows):
        ds, vid = row["dataset"], row["video_id"]
        segments = asr[ds].get(vid, [])
        frames = frame_paths(ds, vid, NATIVE_FRAMES, "k20")
        segs = plan(row, segments, frames, args.J)
        words_total = sum(len(t.split()) for _, _, t in segments if t)
        line = {"dataset": ds, "video_id": vid, "duration": float(row["duration"]), "n_frames_video": len(frames),
                "prefix_tokens_r6": (ref.get((ds, vid)) or {}).get("prefix_tokens"), "words_total": words_total,
                "words_partition_ok": sum(s["n_words"] for s in segs) == words_total,
                "window_text_rule_ok": all(" ".join(x[2] for x in s["pieces"]) == window_text(segments, s["t1"], s["text_bound"])
                                           for s in segs),
                "frames_partition_ok": sum(s["n_frames"] for s in segs) == len(frames), "segments": []}
        for s in segs:
            msgs, files = j.prefix_messages(s["frames"], s["pieces"])
            text, enc = j.encode_prefix(msgs, files)
            qid, _ = j.branch_ids(msgs, VIDEO_QUESTION)
            ok = subset_seam_ok(j, msgs, files, enc["input_ids"][0].tolist(), qid)
            line["segments"].append({**public(s), "prefix_tokens": int(enc["input_ids"].shape[1]),
                                     "branch_tokens": len(qid), "seam_ok": ok})
            if not shown and s["k"] == 1:
                print(f"---- {ds}/{vid} segment 1 [{s['t1']:.2f}, {s['t2']:.2f}) prefix text:\n{text}\n----")
                shown = True
        per.append(line)
        if (n + 1) % 200 == 0:
            print(f"dry-run {n + 1}/{len(rows)}", flush=True)
    if args.dry_out:
        Path(args.dry_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.dry_out, "w") as fh:
            for line in per:
                fh.write(json.dumps(line) + "\n")
    for ds in args.datasets:
        L = [x for x in per if x["dataset"] == ds]
        S = [s for x in L for s in x["segments"]]
        seg_tok = np.array([sum(s["prefix_tokens"] + s["branch_tokens"] for s in x["segments"]) for x in L], float)
        r6_tok = np.array([x["prefix_tokens_r6"] or np.nan for x in L], float)
        ratio = seg_tok / r6_tok
        w = np.array([s["n_words"] for s in S])
        summary = {
            "videos": len(L), "segments": len(S),
            "seam_fail": sum(not s["seam_ok"] for s in S),
            "words_partition_fail": sum(not x["words_partition_ok"] for x in L),
            "window_text_rule_fail": sum(not x["window_text_rule_ok"] for x in L),
            "frames_partition_fail": sum(not x["frames_partition_ok"] for x in L),
            "videos_not_20_frames": [(x["video_id"], x["n_frames_video"]) for x in L if x["n_frames_video"] != NATIVE_FRAMES],
            "frames_per_segment": dict(sorted(Counter(s["n_frames"] for s in S).items())),
            "segments_no_frames": sum(s["n_frames"] == 0 for s in S),
            "segments_no_words": int((w == 0).sum()), "videos_no_words": sum(x["words_total"] == 0 for x in L),
            "videos_with_speech_but_some_segment_silent": sum(x["words_total"] > 0 and any(s["n_words"] == 0 for s in x["segments"]) for x in L),
            "words_per_segment_p10_p50_p90": [float(np.percentile(w, q)) for q in (10, 50, 90)],
            "m_start_per_segment": dict(sorted(Counter(s["m_start"] for s in S).items())),
            "m_overlap_per_segment_p10_p50_p90": [float(np.percentile([s["m_overlap"] for s in S], q)) for q in (10, 50, 90)],
            "segments_m_start_zero": sum(s["m_start"] == 0 for s in S),
            "segment_prefix_tokens_p10_p50_p90": [float(np.percentile([s["prefix_tokens"] for s in S], q)) for q in (10, 50, 90)],
            "sum_segment_tokens_over_r6_prefix_mean_median_p90": [float(np.nanmean(ratio)), float(np.nanmedian(ratio)),
                                                                   float(np.nanpercentile(ratio, 90))],
            "corpus_tokens_segments_plus_video_over_r6": float((np.nansum(seg_tok) + np.nansum(r6_tok)) / np.nansum(r6_tok)),
        }
        print(f"SUMMARY {ds}", json.dumps(summary))
    print("DONE dry-run", len(rows))


# ----------------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name")
    ap.add_argument("--exp-id", default="20261025_m3_coverage")
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"], choices=sorted(MANIFEST))
    ap.add_argument("--J", type=int, default=4, help="segments per video")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--limit", type=int, default=0, help="first N videos of every selected corpus")
    ap.add_argument("--verify", action="store_true", help="cache vs plain call on the first 2 videos of every corpus")
    ap.add_argument("--dry-run", action="store_true", help="CPU only, no weights; writes only --dry-out if given")
    ap.add_argument("--dry-out", help="dry run: per-video jsonl of the segment inputs and seams")
    args = ap.parse_args()
    rows = selected_rows(args.datasets, args.limit)
    asr = {ds: load_asr(ds) for ds in args.datasets}
    ref = reference_reads(args.datasets)
    if args.dry_run:
        dry_run(args, rows, asr, ref)
        return
    if not args.run_name:
        raise SystemExit("--run-name is required")
    torch.manual_seed(SEED)
    out = ROOT / "runs" / args.exp_id / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out / "run.log"), logging.StreamHandler(sys.stdout)])
    logging.info("host %s", socket.gethostname())
    (out / "run.pid").write_text(str(os.getpid()))
    j = Judge(model_id=args.model)
    import transformers
    cfg = dict(vars(args))
    cfg.update({"host": socket.gethostname(), "date": time.strftime("%Y-%m-%d"), "implementation": IMPLEMENTATION,
                "code": f"{CODE_PATH} + src/prefix_questions.py; working copy, {time.strftime('%Y-%m-%d')}",
                "command": "python -u " + " ".join(sys.argv), "family": j.family, "transformers": transformers.__version__,
                "torch": torch.__version__, "seed": SEED, "GT_read": False, "question": VIDEO_QUESTION,
                "yes_ids": j.yes_ids, "no_ids": j.no_ids, "native_frames": NATIVE_FRAMES, "window_seconds": WINDOW_S,
                "segment_rule": {"spans": "[k D / J, (k + 1) D / J), last closed on the right",
                                 "frames": "r6 frames_k20 with t1 <= t < t2", "timestamps": "absolute",
                                 "transcript": "r6 window-text rule (proportional word cut at boundaries), last segment "
                                               "extended to the latest chunk end"},
                "manifests": {ds: str(MANIFEST[ds].relative_to(ROOT)) for ds in args.datasets},
                "reference_reads": {ds: str(REFERENCE[ds].relative_to(ROOT)) for ds in args.datasets},
                "verify_gates": {"dz": GATE_DZ, "seam": "exact token equality"}, "n_videos": len(rows)})
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    logging.info("videos %d  J %d", len(rows), args.J)
    path = out / "segments.jsonl"
    done = set()
    if path.exists():
        done = {(r["dataset"], r["video_id"]) for r in map(json.loads, open(path)) if not r.get("error")}
    verify_ids = {(r["dataset"], r["video_id"]) for ds in args.datasets
                  for r in [x for x in rows if x["dataset"] == ds][:N_VERIFY]} if args.verify else set()
    n_err, t0, dzs, n_new = 0, time.time(), [], 0
    with open(path, "a") as fh:
        for n, row in enumerate(rows):
            key = (row["dataset"], row["video_id"])
            if key in done:
                continue
            base = {"dataset": key[0], "video_id": key[1], "host": socket.gethostname(), "J": args.J,
                    "duration": float(row["duration"])}
            try:
                begin = time.time()
                rec = read_video(j, row, asr[key[0]].get(key[1], []), args.J, key in verify_ids)
                r6 = ref.get(key)
                rec.update({"z_video_r6": r6 and r6["z_video"], "dz_r6": None if r6 is None else abs(rec["z_video"] - r6["z_video"]),
                            "prefix_tokens_r6": r6 and r6["prefix_tokens"], "seconds": time.time() - begin})
                if r6 is not None:
                    dzs.append(rec["dz_r6"])
                n_new += 1
                if "verify" in rec:
                    logging.info("VERIFY %s/%s %s", *key, json.dumps(rec["verify"]))
                    with open(out / "verify.jsonl", "a") as vf:
                        vf.write(json.dumps({"dataset": key[0], "video_id": key[1], **rec["verify"]}) + "\n")
                    bad = verify_bad(rec["verify"])
                    if bad:
                        raise SystemExit(f"VERIFY GATE FAILED {key}: {bad}")
                fh.write(json.dumps({**base, "error": None, **rec}) + "\n")
                fh.flush()
                logging.info("%d/%d %s/%s z_video %.3f (r6 %s)  z_seg %s  frames %s words %s  %.1fs", n + 1, len(rows), *key,
                             rec["z_video"], None if r6 is None else round(r6["z_video"], 3),
                             [round(z, 2) for z in rec["z_seg"]], [s["n_frames"] for s in rec["segments"]],
                             [s["n_words"] for s in rec["segments"]], rec["seconds"])
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                n_err += 1
                logging.error("OOM %s %s", key[1], str(exc)[:200])
                fh.write(json.dumps({**base, "error": "OOM"}) + "\n")
            except Exception as exc:  # noqa: BLE001
                n_err += 1
                logging.exception("FAILED %s: %s", key[1], exc)
                fh.write(json.dumps({**base, "error": f"{type(exc).__name__}: {exc}"}) + "\n")
            if (n + 1) % 20 == 0:
                logging.info("progress %d/%d  %.1fs/video  errors %d", n + 1, len(rows), (time.time() - t0) / max(1, n_new), n_err)
    if dzs:
        s = sorted(dzs)
        logging.info("z_video vs stored r6 reads: median |dz| %.4f  max %.4f  over %d videos", s[len(s) // 2], s[-1], len(s))
    logging.info("DONE videos=%d errors=%d elapsed=%.0fs", len(rows), n_err, time.time() - t0)


if __name__ == "__main__":
    main()
