#!/usr/bin/env python3
"""Reader ablation arms of the adopted window-evidence reader (README §2). Qwen3-VL-8B, frozen; GPU inside Slurm only;
no label file is read.

Arms (src/reader_arms.py + the adopted src/stance_cache, src/window_evidence):
  base              the adopted reader (must equal the stored adopted reads; --smoke checks it)
  no_transcript     A1 prefix without the whole transcript
  no_prefix_frames  A2 prefix without the 20 frames (window visual branches keep their 0.5 fps frames)
  no_stance         A3 verdict read, not written back; windows follow the prefix directly
  asr_windows       A4 Whisper segments as windows (gaps unscored)
  no_isolation      A5 branches see each other in order; the model's Yes/No written back after every branch
  joint             A6 one branch per window: window frames + aligned words, joint question; key z_joint
Arm-major order (every video of one arm, then the next arm). Output runs/20261101_reader_ablation/<name>/
{records/<arm>/<ds>/<vid>.json, <arm>/predictions.jsonl, <arm>/config.json, config.json, run.log, run.pid}.
Resumable: (arm, video) pairs with a record are not read again.
--smoke: first two videos of HateMM, HateClipSeg, DeHate, every arm, plus no_isolation on the longest HateMM video.
Timing run (main-session request 2026-11-01): --arms base --name timing_<corpus>, one corpus per job; each record keeps
seconds (whole video, after model load), prefix_seconds (prefix forward + verdict + stance turn), window_seconds,
prefix_tokens and peak_GiB; the base check against the stored adopted reads runs on every video.
"""
import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.mllm_judge import MODEL, Judge, yesno_question  # noqa: E402
from src.reader_arms import SequentialReader, asr_windows, build_context, record  # noqa: E402
from src.stance_cache import margin  # noqa: E402
from src.video_inputs import fixed_windows, frame_paths, load_asr, load_manifest, window_text  # noqa: E402
from src.window_evidence import WINDOW_S, aligned_slice, aligned_words, frames_margin, in_window, local_frames  # noqa: E402

EXP = ROOT / "runs/20261101_reader_ablation"
NATIVE_FRAMES = 20
ARMS = ["base", "no_transcript", "no_prefix_frames", "no_stance", "asr_windows", "no_isolation", "joint"]
SAME_VERDICT = {"base", "no_stance", "asr_windows", "no_isolation", "joint"}  # prefix and verdict question as the base
MANIFESTS = {"HateMM": ROOT / "data/omsl_v6_inputs/manifests/all_test.jsonl", "HateClipSeg": ROOT / "data/omsl_v6_inputs/manifests/all_test.jsonl",
             "DeHate": ROOT / "data/manifests/DeHate_test.jsonl"}
STORED = {"HateMM": ROOT / "runs/20261014_m1_audio_bound/vl_aligned/vl_text_aligned/predictions.jsonl",
          "HateClipSeg": ROOT / "runs/20261014_m1_audio_bound/vl_aligned/vl_text_aligned/predictions.jsonl",
          "DeHate": ROOT / "runs/20261031_window_evidence/DeHate/adopted/predictions.jsonl"}
CODE = "experiments/20261101_reader_ablation/read_arms.py"
TOL = 1e-3


def stored(path, ds):
    return {r["video_id"]: r for r in map(json.loads, open(path)) if not r.get("error") and r["dataset"] == ds}


def tick():
    torch.cuda.synchronize()
    return time.perf_counter()


def visual_read(j, cache, ctx, frames, q):
    """Adopted visual branch: window frames next to the question; a window without a frame asks the question alone."""
    return frames_margin(j, cache, ctx, frames, q) if frames else margin(j, cache, ctx, q)


@torch.no_grad()
def read(j, arm, row, segments, rec):
    begin = tick()
    torch.cuda.reset_peak_memory_stats()
    native = frame_paths(row["dataset"], row["video_id"], NATIVE_FRAMES)
    assert 0 < len(native) <= NATIVE_FRAMES
    frames = local_frames(row)
    cache, ctx = build_context(j, native, segments, with_context=arm != "no_transcript", with_frames=arm != "no_prefix_frames",
                               stance=arm != "no_stance")
    prefix_secs = tick() - begin  # 20 frames + transcript prefix forward, verdict read and stance turn (timing only)
    if arm == "asr_windows":
        segs = asr_windows(segments)
        wins = [(s, e) for s, e, _ in segs]
        bodies = [aligned_slice(rec, s, e, t) for s, e, t in segs]
    else:
        wins = fixed_windows(float(row["duration"]), WINDOW_S)
        bodies = [aligned_slice(rec, t1, t2, window_text(segments, t1, t2)) for t1, t2 in wins]
    n = len(wins)
    seq = SequentialReader(j, cache, ctx) if arm == "no_isolation" else None
    reads, traces = [], []
    for i, ((t1, t2), body) in enumerate(zip(wins, bodies)):
        inside = in_window(frames, t1, t2)
        has_speech = body is not None and body.strip() != ""
        r = {}
        if arm == "joint":
            q = yesno_question(i, n, t1, t2, body, "joint")
            r["z_joint"] = visual_read(j, cache, ctx, inside, q)
        else:
            vq = yesno_question(i, n, t1, t2, body, "visual")
            sq = yesno_question(i, n, t1, t2, body, "speech") if has_speech else None
            if seq is not None:
                r["z_visual"] = seq.read(inside, vq)
                if sq is not None:
                    r["z_speech"] = seq.read([], sq)
            else:
                r["z_visual"] = visual_read(j, cache, ctx, inside, vq)
                if sq is not None:
                    r["z_speech"] = margin(j, cache, ctx, sq)
        reads.append(r)
        traces.append(dict(i=i, bounds=[t1, t2], local_times=[t for t, _ in inside], words=len(body.split()) if has_speech else 0, **r))
    secs = tick() - begin
    pred = record(row, ctx, wins, reads, f"reader_ablation_{arm}", secs, CODE, segment_windows=arm == "asr_windows")
    out = dict(model=MODEL, host=socket.gethostname(), arm=arm, GT_read=False, global_margin=ctx["global_margin"], stance=ctx["stance"],
               prefix_tokens=ctx["prefix_tokens"], stance_cache_tokens=ctx["stance_cache_tokens"],
               final_cache_tokens=cache.get_seq_length(), aligned_status=None if rec is None else rec["status"], windows=n,
               windows_with_frames=sum(bool(t["local_times"]) for t in traces), seconds=secs, prefix_seconds=prefix_secs,
               window_seconds=secs - prefix_secs, branches=sum(len(r) for r in reads),
               peak_GiB=torch.cuda.max_memory_allocated() / 2 ** 30, traces=traces, prediction=pred)
    del cache, seq
    return out


def check(bundle, ref):
    """Max |difference| to the stored adopted reads of the same video, for the parts the arm shares with the base."""
    arm, out = bundle["arm"], {}
    if ref is None:
        return out
    ws = ref["extra"]["windows"]
    mine = bundle["prediction"]["extra"]["windows"]
    if arm in SAME_VERDICT:
        out["z_video"] = abs(ref["extra"]["z_video"] - bundle["global_margin"])
    if arm == "base":
        assert len(ws) == len(mine), (len(ws), len(mine))
        out["z_visual"] = max(abs(a["z_visual"] - b["z_visual"]) for a, b in zip(ws, mine))
        out["z_speech"] = max((abs(a["z_speech"] - b["z_speech"]) for a, b in zip(ws, mine) if "z_speech" in a and "z_speech" in b), default=0.0)
        out["speech_presence"] = float(sum(("z_speech" in a) != ("z_speech" in b) for a, b in zip(ws, mine)))
        out["score_curve"] = max(abs(a - b) for a, b in zip(ref["score_curve"], bundle["prediction"]["score_curve"]))
        assert len(ref["score_curve"]) == len(bundle["prediction"]["score_curve"])
    if arm == "no_isolation":
        out["first_visual"] = abs(ws[0]["z_visual"] - mine[0]["z_visual"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["HateMM"])
    ap.add_argument("--arms", nargs="+", default=[a for a in ARMS if a != "base"], choices=ARMS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--name", default="")
    a = ap.parse_args()
    datasets = ["HateMM", "HateClipSeg", "DeHate"] if a.smoke else a.datasets
    arms = ARMS if a.smoke else a.arms
    name = a.name or ("smoke" if a.smoke else "_".join(datasets))
    out = EXP / name
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.FileHandler(out / "run.log"), logging.StreamHandler()])
    logging.info("host %s", socket.gethostname())
    (out / "run.pid").write_text(str(os.getpid()))
    rows = {ds: load_manifest(MANIFESTS[ds], [ds]) for ds in datasets}
    jobs = []  # (arm, row)
    if a.smoke:
        longest = max(rows["HateMM"], key=lambda r: float(r["duration"]))
        rows = {ds: rs[:2] for ds, rs in rows.items()}
        jobs = [(arm, r) for arm in arms for ds in datasets for r in rows[ds]] + [("no_isolation", longest)]
    else:
        jobs = [(arm, r) for arm in arms for ds in datasets for r in rows[ds]]
    import transformers
    commit = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%h %cd", "--date=short"], capture_output=True, text=True).stdout.strip()
    cfg = dict(host=socket.gethostname(), date=time.strftime("%Y-%m-%d"), model=MODEL, arms=arms, window_seconds=WINDOW_S, native_frames=NATIVE_FRAMES,
               GT_read=False, smoke=a.smoke, datasets=datasets, videos={ds: len(r) for ds, r in rows.items()}, torch=torch.__version__,
               transformers=transformers.__version__, code=f"{CODE} + src/reader_arms.py; commit {commit}", command="python -u " + " ".join(sys.argv))
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    refs = {ds: stored(STORED[ds], ds) for ds in datasets}
    asr = {ds: load_asr(ds) for ds in datasets}
    words = {ds: aligned_words(ds) for ds in datasets}
    torch.manual_seed(0)
    j = Judge(MODEL)
    worst, errors, t0 = {}, 0, time.perf_counter()
    for k, (arm, row) in enumerate(jobs, 1):
        ds, vid = row["dataset"], row["video_id"]
        path = out / "records" / arm / ds / f"{vid}.json"
        if path.exists():
            continue
        try:
            bundle = read(j, arm, row, asr[ds].get(vid, []), words[ds].get(vid))
        except Exception as ex:  # noqa: BLE001
            errors += 1
            logging.exception("ERROR %s %s/%s %s", arm, ds, vid, ex)
            torch.cuda.empty_cache()
            continue
        bundle["check"] = check(bundle, refs[ds].get(vid))
        for c, v in bundle["check"].items():
            worst[f"{arm}:{c}"] = max(worst.get(f"{arm}:{c}", 0.0), v)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".partial")
        tmp.write_text(json.dumps(bundle, allow_nan=False) + "\n")
        tmp.replace(path)
        logging.info("%d/%d %s %s/%s windows=%d with_frames=%d cache=%d %.1fs peak=%.1fGiB check=%s elapsed=%.0fs", k, len(jobs), arm, ds, vid,
                     bundle["windows"], bundle["windows_with_frames"], bundle["final_cache_tokens"], bundle["seconds"], bundle["peak_GiB"],
                     {c: round(v, 5) for c, v in bundle["check"].items()}, time.perf_counter() - t0)
    logging.info("worst differences to stored adopted reads (this session): %s", worst)
    bad = {c: v for c, v in worst.items() if v > TOL}
    missing = 0
    for arm in arms:
        recs = []
        for ds in datasets:
            for row in rows[ds]:
                path = out / "records" / arm / ds / f"{row['video_id']}.json"
                if path.exists():
                    recs.append(json.loads(path.read_text())["prediction"])
                else:
                    missing += 1
        (out / arm).mkdir(exist_ok=True)
        (out / arm / "config.json").write_text(json.dumps({**cfg, "arm": arm}, indent=2) + "\n")
        (out / arm / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        logging.info("arm %s: %d records", arm, len(recs))
    if a.smoke and not (out / "records" / "no_isolation" / "HateMM" / f"{longest['video_id']}.json").exists():
        missing += 1
    logging.info("errors this session %d, missing %d", errors, missing)
    if missing or bad:
        logging.info("FAILED missing=%d differences_above_tolerance=%s", missing, bad)
        sys.exit(1)
    logging.info("DONE jobs=%d", len(jobs))


if __name__ == "__main__":
    main()
