#!/usr/bin/env python3
"""CLARA (github.com/yuchenzhang-1/CLARA @468a6bc) step 1-2 inputs, CPU: clips and sampled frames.

    python clara_prep.py --dataset DS --workers 8

For every train / val video (a whole video, as published) and every test-cohort video:
- clips: the authors' `get_video_clip.py` functions, unchanged (speech clips = Whisper segments, silences >= 1 s
  become silent clips, wav duration authoritative, frame budgets allocated in proportion to clip duration);
- frames: the authors' `get_frame.py` bin-centre indices, read sequentially with cv2 and saved as JPEG quality 95;
  budget 40 for train / val videos;
- rationale frames (every video, video level): the frames of the 100-frame budget in clip order, 20 of them taken
  uniformly (`get_frames_for_rationale.py` `uniform_subsample`, k = 20, budget dir frame_100 "for my model").
For a test video, additionally, per consecutive 8-s window [a, b):
- the window's Whisper segments (cut at the window edges, text sliced proportionally at word boundaries, times
  relative to a) go through the same clip builder with total duration b - a;
- the window's frame budget keeps the video's own sampling rate: n = min(40, max(1, round(40 * (b - a) / D)))
  with D the video's wav duration (the authors' clip duration); these n frames are allocated to the window's clips
  by the authors' rule.
Outputs: data/clara_raw/<DS>/<vid>/{clipinfo.json, clip_XXX/frames/frame_40/, rationale_frames/,
win_JJJ/{clipinfo.json, clip_XXX/frames/frame_w/}}.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "detwin"))
import common as C  # noqa: E402

DATA = Path(os.environ.get("CLARA_DATA", C.REPO / "data"))  # env override: smoke tests only

CLARA_DIR = C.REPO / "third_party" / "CLARA"
RAW = DATA / "clara_raw"
BUDGET = 40          # run_clara.sh BUDGET
RAT_BUDGET = 100     # get_frames_for_rationale.py: "frame_100 for my model"
RAT_K = 20           # 20 frames, as the rationale prompt states ("20 frames sampled uniformly")


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, CLARA_DIR / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GVC = _load("clara_get_video_clip", "data_preprocess/get_raw_fetures/get_video_clip.py")
GF = _load("clara_get_frame", "data_preprocess/get_raw_fetures/get_frame.py")
GFR = _load("clara_get_frames_for_rationale", "data_preprocess/get_raw_fetures/get_frames_for_rationale.py")


def video_clipinfo(segments: list[dict], wav_dur: float | None) -> tuple[list[dict], float, str]:
    """`get_video_clip.process_one_json` without the file I/O."""
    if wav_dur is None:
        dur = max([float(s["end"]) for s in segments], default=0.0)
        src = "segments_end_fallback"
        segs = segments
    else:
        dur = float(wav_dur)
        src = "wav_duration"
        segs, _, _ = GVC.clip_segments_to_duration(segments, dur)
    clips = GVC.build_updated_clips_no_merge_speech(segments=segs, total_duration=dur,
                                                    silence_merge_threshold=GVC.SILENCE_THRESHOLD)
    GVC.allocate_frames_multi_budgets(clips, [BUDGET, RAT_BUDGET], out_field="num_frames_allocated")
    return clips, dur, src


def frame_targets(clips, budget_key: str, offset: float, fps: float, max_idx: int):
    """[(clip_idx, local_i, frame index)] for one budget (get_frame.py `center_of_bin_frame_indices`)."""
    out = []
    for c in clips:
        k = int(c["num_frames_allocated"].get(budget_key, 0))
        if k <= 0:
            continue
        idxs = GF.center_of_bin_frame_indices(offset + float(c["start"]), offset + float(c["end"]), k, fps, max_idx)
        out += [(int(c["clip_idx"]), i, fi) for i, fi in enumerate(idxs)]
    return out


def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False) + "\n")


def save_jpg(p: Path, bgr) -> None:
    import cv2
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def prep_video(task) -> dict:
    import cv2
    cv2.setNumThreads(1)
    ds, vid, split, test_dur = task
    out = RAW / ds / vid
    done = out / "_PREP_DONE.json"
    if done.is_file():
        return json.loads(done.read_text())
    res = {"video_id": vid, "split": split}
    mp4 = C.video_path(ds, vid, split)
    if mp4 is None:
        res["status"] = "no_video"
        return res
    wav = C.wav_path(ds, vid)
    if not wav.is_file():
        err = C.extract_wav(mp4, wav)
        if err:
            res["wav_error"] = err
    wav_dur = GVC.wav_duration_sec(wav)
    segs_all = SEGS.get(vid, [])
    segments = [{"id": k, "start": s, "end": e, "text": t} for k, (s, e, t) in enumerate(segs_all)]
    clips, vdur, src = video_clipinfo(segments, wav_dur)
    res.update({"wav_duration": wav_dur, "clip_duration": vdur, "duration_source": src, "n_clips": len(clips)})
    if not clips:
        # authors: "no_clips" -> the video gets no embedding and is skipped (train/val); test handled at embedding
        res["status"] = "no_clips"
        write_json(out / "clipinfo.json", {"video_id": vid, "total_duration_sec": vdur, "clips": [],
                                           "duration_source": src})
    else:
        write_json(out / "clipinfo.json", {"video_id": vid, "total_duration_sec": vdur, "num_clips": len(clips),
                                           "clips": clips, "duration_source": src})
    fps, n_cap = C.probe_video(mp4)
    max_idx = n_cap - 1 if n_cap > 0 else 10 ** 12
    tasks = {}  # frame index -> list of output paths
    # rationale frames (all videos): budget-100 frames in clip order, 20 uniformly
    rat = frame_targets(clips, str(RAT_BUDGET), 0.0, fps, max_idx)
    rat_sel = GFR.uniform_subsample(rat, RAT_K) if rat else []
    for j, (_, _, fi) in enumerate(rat_sel):
        tasks.setdefault(fi, []).append(out / "rationale_frames" / f"frame_{j:03d}.jpg")
    res["rationale_frames"] = len(rat_sel)
    if split in ("train", "val"):
        for ci, li, fi in frame_targets(clips, str(BUDGET), 0.0, fps, max_idx):
            tasks.setdefault(fi, []).append(out / f"clip_{ci:03d}" / "frames" / f"frame_{BUDGET}" / f"frame_{li:03d}.jpg")
    else:
        wins = C.windows(test_dur)
        n_frames_w = []
        for j, (a, b) in enumerate(wins):
            wseg = C.window_segments(segs_all, a, b)
            wclips = GVC.build_updated_clips_no_merge_speech(segments=wseg, total_duration=b - a,
                                                             silence_merge_threshold=GVC.SILENCE_THRESHOLD)
            n = min(BUDGET, max(1, int(round(BUDGET * (b - a) / vdur)))) if vdur > 0 else BUDGET
            alloc = GVC.allocate_frames_one_budget(wclips, n)
            for c, k in zip(wclips, alloc):
                c["num_frames_allocated"] = {"w": int(k)}
            n_frames_w.append(n)
            wdir = out / f"win_{j:03d}"
            write_json(wdir / "clipinfo.json", {"video_id": vid, "window": j, "offset": a, "end": b,
                                                "total_duration_sec": b - a, "frame_budget": n,
                                                "num_clips": len(wclips), "clips": wclips})
            for ci, li, fi in frame_targets(wclips, "w", a, fps, max_idx):
                tasks.setdefault(fi, []).append(wdir / f"clip_{ci:03d}" / "frames" / "frame_w" / f"frame_{li:03d}.jpg")
        res["n_windows"] = len(wins)
        res["window_frames"] = int(sum(n_frames_w))
    got, n_dec = C.read_frames(mp4, set(tasks))
    res["n_frames_cap"], res["n_decoded"], res["fps"] = n_cap, n_dec, fps
    res["clamped_frames"] = sum(1 for fi in tasks if fi >= n_dec)
    if not got and tasks:
        res["status"] = "no_frames_decoded"
        return res
    for fi, paths in tasks.items():
        for p in paths:
            save_jpg(p, got[fi])
    res.setdefault("status", "ok")
    write_json(done, res)
    return res


SEGS: dict = {}


def _init(ds, durations):
    global SEGS
    SEGS = C.load_asr(ds, durations)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=C.DATASETS)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    ds = args.dataset
    log = C.RunLog(C.RUNS / "clara" / ds / "prep" / "run.log")
    sp = C.load_split(ds)
    test_dur = {x["video_id"]: float(x["duration"]) for x in sp["test"]}
    tasks = [(ds, x["video_id"], s, None) for s in ("train", "val") for x in sp[s]]
    tasks += [(ds, v, "test", d) for v, d in test_dur.items()]
    log(f"clara prep {ds}: {len(tasks)} videos, {args.workers} workers; code {C.code_version()}")
    t0 = time.time()
    rows = []
    with Pool(args.workers, initializer=_init, initargs=(ds, test_dur)) as pool:
        for k, r in enumerate(pool.imap_unordered(prep_video, tasks, chunksize=1)):
            rows.append(r)
            if (k + 1) % 200 == 0:
                log(f"  {k + 1}/{len(tasks)} ({time.time() - t0:.0f}s)")
    status = {}
    for r in rows:
        status.setdefault(r["status"], []).append(r["video_id"])
    rep = {"counts": {k: len(v) for k, v in status.items()}, "not_ok": {k: v for k, v in status.items() if k != "ok"},
           "wav_errors": {r["video_id"]: r["wav_error"] for r in rows if r.get("wav_error")},
           "clamped_frames": {r["video_id"]: r["clamped_frames"] for r in rows if r.get("clamped_frames")},
           "test_window_frames": int(sum(r.get("window_frames", 0) for r in rows)),
           "test_windows": int(sum(r.get("n_windows", 0) for r in rows)), "seconds": round(time.time() - t0, 1)}
    (C.RUNS / "clara" / ds / "prep" / "prep_report.json").write_text(json.dumps(rep, indent=2) + "\n")
    log(f"DONE prep: {rep['counts']}; windows {rep['test_windows']} with {rep['test_window_frames']} frames")
    bad_test = [r["video_id"] for r in rows if r["split"] == "test" and r["status"] in ("no_video", "no_frames_decoded")]
    if bad_test:
        log(f"FAILED: test videos without media: {bad_test}")
        sys.exit(1)


if __name__ == "__main__":
    main()
