#!/usr/bin/env python3
"""Shared plumbing for the new label-free baselines (EventVAD, PRISM, AnyAnomaly, VideoMind).

Only cohort, path, 4 fps grid, coverage and evaluator-call logic lives here. No label is read: the GT npz is
opened only for its video ids and its per-video frame count (the length a curve must cover), never for `y4`
values, except inside the evaluator subprocess.

Cohorts (hate_query.md section 3, run_plan.md section 1.1):
  HateMM 215 / HateClipSeg 118 = (dataset, video_id) of runs/20260926_twolevel/final_rawkey/predictions.jsonl
  DeHate 1151                  = test video ids of data/gt_4fps/DeHate.npz
The lists are written once to runs/20261008_baselines/cohort/<dataset>.txt and copied to the run machines.
"""
from __future__ import annotations

import json
import math
import os
import socket
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASETS = ("HateMM", "HateClipSeg", "DeHate")
EXPECTED = {"HateMM": 215, "HateClipSeg": 118, "DeHate": 1151}
COHORT_DIR = os.path.join(REPO, "runs", "20261008_baselines", "cohort")
GT_DIR = os.path.join(REPO, "data", "gt_4fps")
MANIFESTS = {
    "HateMM": os.path.join(REPO, "data", "omsl_v6_inputs", "manifests", "all_test.jsonl"),
    "HateClipSeg": os.path.join(REPO, "data", "omsl_v6_inputs", "manifests", "all_test.jsonl"),
    "DeHate": os.path.join(REPO, "data", "manifests", "DeHate_test.jsonl"),
}
COHORT_SOURCE = os.path.join(REPO, "runs", "20260926_twolevel", "final_rawkey", "predictions.jsonl")
RATE = 4.0


# ----------------------------------------------------------------------------- cohort and paths
def build_cohort_files():
    """Write runs/20261008_baselines/cohort/<ds>.txt from the sources in hate_query.md (lab1 only)."""
    os.makedirs(COHORT_DIR, exist_ok=True)
    ids = {"HateMM": set(), "HateClipSeg": set()}
    with open(COHORT_SOURCE) as fh:
        for line in fh:
            row = json.loads(line)
            if row["dataset"] in ids:
                ids[row["dataset"]].add(row["video_id"])
    gt = np.load(os.path.join(GT_DIR, "DeHate.npz"), allow_pickle=True)
    ids["DeHate"] = {str(v) for v, s in zip(gt["video_ids"], gt["split"]) if str(s) == "test"}
    for ds, vids in ids.items():
        if len(vids) != EXPECTED[ds]:
            raise SystemExit("cohort %s has %d ids, expected %d" % (ds, len(vids), EXPECTED[ds]))
        path = os.path.join(COHORT_DIR, ds + ".txt")
        with open(path, "w") as fh:
            fh.write("\n".join(sorted(vids)) + "\n")
        print("wrote %s (%d)" % (path, len(vids)))


def cohort(ds):
    path = os.path.join(COHORT_DIR, ds + ".txt")
    with open(path) as fh:
        vids = [x.strip() for x in fh if x.strip()]
    if len(vids) != EXPECTED[ds]:
        raise RuntimeError("%s: %d ids, expected %d" % (path, len(vids), EXPECTED[ds]))
    return vids


def local_path(path):
    """Manifest paths are written for uoa-lab1 (/home/jehc223/...); map them to this machine's home."""
    home = os.path.expanduser("~")
    for prefix in ("/home/jehc223/", "/home/junyi/"):
        if path.startswith(prefix):
            return os.path.join(home, path[len(prefix):])
    return path


CAMPUS_DATA = "/data/jehc223"          # raw-video root on the campus servers (CLAUDE.md: /data/jehc223/<dataset>/)


def resolve_video(ds, vid, manifest_path):
    """The manifest path mapped to this machine; if it is absent (HateClipSeg's manifest points at a removed
    pilot directory), the raw-video location of CLAUDE.md, ~/data/<dataset>/{videos,video,test}/<id>.<ext> on the
    lab machines, /data/jehc223/<dataset>/{videos,video,test}/<id>.<ext> on the campus servers."""
    cand = [local_path(manifest_path)]
    for root in (os.path.expanduser("~/data"), CAMPUS_DATA):
        for sub in ("videos", "video", "test"):
            for ext in (".mp4", ".webm", ".mkv"):
                cand.append(os.path.join(root, ds, sub, vid + ext))
    for c in cand:
        if os.path.isfile(c):
            return c
    return cand[0]


def manifest(ds):
    """video_id -> {duration, video_path} for the dataset's test manifest."""
    out = {}
    with open(MANIFESTS[ds]) as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("dataset") != ds:
                continue
            out[row["video_id"]] = {"duration": float(row["duration"]),
                                    "video_path": resolve_video(ds, row["video_id"], row["video_path"])}
    return out


def n_frames(duration):
    """Curve length on the 4 fps grid (run_plan.md 1.2)."""
    return int(math.ceil(duration * RATE - 1e-9))


# ----------------------------------------------------------------------------- transcripts (audio-visual variants)
ASR_DIR = os.path.join(REPO, "data", "asr_whisper_large_v3")
NO_SPEECH = "(no speech)"


def transcript_segments(ds, durations=None):
    """video_id -> [(start, end, text)] from data/asr_whisper_large_v3/<ds>/timestamped_chunks.jsonl (Whisper
    large-v3 segments, the transcript the Reader and the Qwen baselines read).

    Untimed chunks are kept (run_plan.md 1.3), with the rule of qwen3_text/text_llm.py `load_segments`: a missing
    start takes the previous chunk's end (0 for the first), a missing end takes the next chunk's start if that is
    later, else the video duration (test manifest). Videos without a row get an empty list."""
    dur = durations if durations is not None else {v: m["duration"] for v, m in manifest(ds).items()}
    out = {v: [] for v in dur}
    with open(os.path.join(ASR_DIR, ds, "timestamped_chunks.jsonl")) as fh:
        for line in fh:
            r = json.loads(line)
            v = r["video_id"]
            if v not in dur:
                continue
            ch = r.get("chunks") or []
            segs, prev_end = [], 0.0
            for i, c in enumerate(ch):
                s, e = c.get("start"), c.get("end")
                s = float(s) if s is not None else prev_end
                if e is None:
                    nxt = next((float(x["start"]) for x in ch[i + 1:] if x.get("start") is not None), None)
                    e = nxt if nxt is not None and nxt > s else dur[v]
                e = float(e)
                segs.append((s, e, c.get("text") or ""))
                prev_end = max(prev_end, e)
            out[v] = segs
    return out


def span_text(segments, t1, t2):
    """Transcript of [t1, t2] s: src/video_inputs.py `window_text` (segments sliced proportionally at word
    boundaries), stripped. Empty string when nothing is spoken in the span."""
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    from src.video_inputs import window_text
    return " ".join(window_text(segments, t1, t2).split())


def gt_lengths(ds):
    """video_id -> number of GT frames. Only the array length is read."""
    gt = np.load(os.path.join(GT_DIR, ds + ".npz"), allow_pickle=True)
    return {str(v): len(gt["y4"][i]) for i, v in enumerate(gt["video_ids"]) if str(gt["split"][i]) == "test"}


def transcode_h264(src, dst):
    """Decode-only fallback for files decord cannot open (some HateClipSeg webm/AV1): re-encode the video stream to
    H.264 at the same frame rate and size (libx264, CRF 18, no audio). The caller records that it was used."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", src, "-map", "0:v:0", "-c:v", "libx264", "-crf", "18",
           "-preset", "veryfast", "-pix_fmt", "yuv420p", "-an", dst]
    subprocess.run(cmd, check=True)
    return dst


def decord_ok(path):
    try:
        import decord
        vr = decord.VideoReader(path, num_threads=1)
        n = len(vr)
        vr.get_batch([0, max(0, n // 2), n - 1]).asnumpy()
        return n > 0 and vr.get_avg_fps() > 0
    except Exception:  # noqa: BLE001
        return False


# ----------------------------------------------------------------------------- 4 fps mapping
def units_to_4fps(starts, ends, values, n):
    """Frame i (centre (i + 0.5) / 4 s) takes the value of the unit containing that centre.

    Units are [start, end) in seconds, sorted and non-overlapping. Frames before the first unit take the first unit's
    value; frames after the last unit (or in a gap) take the value of the last unit that started before the centre.
    This is run_plan.md 1.2 ("Frames past the last unit take the last unit's value").
    """
    starts = np.asarray(starts, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    centres = (np.arange(n) + 0.5) / RATE
    idx = np.searchsorted(starts, centres, side="right") - 1
    idx = np.clip(idx, 0, len(starts) - 1)
    return values[idx]


def samples_to_4fps(times, values, n):
    """Point samples at `times` (s) -> 4 fps by the nearest sample in time."""
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    centres = (np.arange(n) + 0.5) / RATE
    j = np.searchsorted(times, centres)
    j0 = np.clip(j - 1, 0, len(times) - 1)
    j1 = np.clip(j, 0, len(times) - 1)
    pick = np.where(np.abs(times[j1] - centres) < np.abs(times[j0] - centres), j1, j0)
    return values[pick]


# ----------------------------------------------------------------------------- rows, coverage, evaluation
def row(method, ds, vid, duration, curve, native_rate, code_path, intervals=None, extra=None, calls=None,
        seed=None, error=None):
    return {"schema_version": 1, "method": method, "dataset": ds, "video_id": vid, "duration": float(duration),
            "native_rate": native_rate, "score_curve": [float(x) for x in curve], "intervals": intervals or [],
            "error": error, "calls": calls, "seed": seed, "code_path": code_path, "extra": extra or {}}


def apply_f2(rows_by_vid, ds):
    """Plan F2: a failed video gets a constant curve at the median frame score of the scored videos."""
    ok = [np.asarray(r["score_curve"], float) for r in rows_by_vid.values() if not r.get("error")]
    if not ok:
        raise RuntimeError("no scored video to take the F2 median from")
    med = float(np.median(np.concatenate(ok)))
    failed = []
    for vid, r in rows_by_vid.items():
        if r.get("error"):
            failed.append((vid, r["error"]))
            r["extra"] = dict(r.get("extra") or {}, fallback="F2", fallback_reason=r["error"], f2_value=med)
            r["score_curve"] = [med] * n_frames(r["duration"])
            r["error"] = None
    return med, failed


def coverage_check(ds, rows):
    """Exact-cohort rule: exactly the cohort ids, error null, finite score on every GT frame and every
    ceil(4 * duration) frame. Returns a report dict; report['ok'] is the verdict."""
    want = set(cohort(ds))
    lens = gt_lengths(ds)
    got = {r["video_id"]: r for r in rows if r["dataset"] == ds}
    missing = sorted(want - set(got))
    extra = sorted(set(got) - want)
    bad = []
    for vid in sorted(want & set(got)):
        r = got[vid]
        s = np.asarray(r["score_curve"], dtype=np.float64)
        need = max(lens.get(vid, 0), n_frames(r["duration"]))
        if r.get("error"):
            bad.append((vid, "error: %s" % r["error"]))
        elif len(s) < need:
            bad.append((vid, "curve %d < %d frames" % (len(s), need)))
        elif not np.isfinite(s[:need]).all():
            bad.append((vid, "%d non-finite frames" % int((~np.isfinite(s[:need])).sum())))
    ok = not missing and not bad and len(want & set(got)) == EXPECTED[ds]
    return {"dataset": ds, "ok": bool(ok), "n_cohort": len(want), "n_rows": len(got), "missing": missing,
            "extra_dropped": extra, "bad": bad}


def finalize(ds, rows, out_dir, method):
    """Coverage check, write predictions.jsonl (cohort rows only), run the canonical evaluator, check overlap."""
    os.makedirs(out_dir, exist_ok=True)
    rep = coverage_check(ds, rows)
    with open(os.path.join(out_dir, "coverage.json"), "w") as fh:
        json.dump(rep, fh, indent=2)
    print("coverage %s: ok=%s rows=%d missing=%d bad=%d extra=%d" % (
        ds, rep["ok"], rep["n_rows"], len(rep["missing"]), len(rep["bad"]), len(rep["extra_dropped"])))
    if not rep["ok"]:
        print("NOT EVALUATED: coverage failed", rep["missing"][:5], rep["bad"][:5])
        return None
    want = set(cohort(ds))
    pred = os.path.join(out_dir, "predictions.jsonl")
    with open(pred, "w") as fh:
        for r in sorted(rows, key=lambda r: r["video_id"]):
            if r["dataset"] == ds and r["video_id"] in want:
                fh.write(json.dumps(r) + "\n")
    metrics = os.path.join(out_dir, "metrics.json")
    cmd = [sys.executable, "-m", "src.eval.evaluate_four_datasets", "--predictions", pred, "--gt-dir", GT_DIR,
           "--out", metrics, "--datasets", ds]
    subprocess.run(cmd, cwd=REPO, check=True, stdout=subprocess.DEVNULL)
    with open(metrics) as fh:
        res = json.load(fh)
    for r in res["per_dataset"]:
        if r["n_videos_overlap"] != EXPECTED[ds] or r["n_videos_predicted"] != EXPECTED[ds]:
            raise RuntimeError("evaluator overlap %s != %d" % (r["n_videos_overlap"], EXPECTED[ds]))
        print("%s %s: ROC %.4f PR %.4f within %.4f (n=%d)" % (
            method, ds, r["frame_ROC_AUC"], r["frame_PR_AUC"], r["within_video_macro_ROC_AUC"],
            r["n_videos_overlap"]))
    return res


def start_run_log(out_dir, argv=None):
    """run.log first line = hostname (CLAUDE.md); also run.pid."""
    os.makedirs(out_dir, exist_ok=True)
    log = os.path.join(out_dir, "run.log")
    new = not os.path.isfile(log)
    with open(log, "a") as fh:
        if new:
            fh.write(socket.gethostname() + "\n")
        fh.write("%s pid=%d argv=%s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), os.getpid(),
                                           " ".join(argv or sys.argv)))
    with open(os.path.join(out_dir, "run.pid"), "w") as fh:
        fh.write(str(os.getpid()) + "\n")
    return log


def git_version():
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%h %cd", "--date=short"], cwd=REPO,
                             capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()
        return out + (" (dirty)" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


if __name__ == "__main__":
    if sys.argv[1:] == ["build-cohort"]:
        build_cohort_files()
    else:
        print("usage: lf_common.py build-cohort")
