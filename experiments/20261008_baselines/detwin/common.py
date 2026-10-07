"""Shared plumbing for the two video-level detectors run per 8-s window (SAGE, CLARA), 2026-10-08.

Both methods are trained as published on whole train/val videos with video-level labels, then applied to each
consecutive 8-s window of a test video as if the window were a short video. Every 4 fps frame takes its window's
P(hateful). This module holds what both methods share:

- splits and video-level labels (`data/weaksup_video_splits/<dataset>.json`, built by `prepare.py`);
- the fixed test cohorts (`hate_query.md` §3) and their durations;
- the 8-s window grid (same grid as our own method, `experiments/20260910_spvl/spvl.py` `fixed_windows`);
- Whisper large-v3 segments and their restriction to a window (proportional word slicing, the rule of
  `spvl.py` `window_text`; untimed chunks kept per `run_plan.md` §1.3);
- video lookup, 16 kHz mono wav extraction (the CLARA authors' ffmpeg command), sequential frame decoding with cv2;
- the window -> 4 fps conversion, the exact-cohort coverage check and the evaluator call.

No test label is read here. The GT file is opened only for its id list and frame counts; labels reach only
`src/eval/evaluate_four_datasets.py`.
"""
from __future__ import annotations

import datetime
import json
import math
import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
EXP_ID = "20261008_baselines"
RUNS = Path(os.environ.get("DETWIN_RUNS", REPO / "runs" / EXP_ID))  # env override: smoke tests only
DATASETS = ("HateMM", "HateClipSeg", "DeHate")
SEEDS = (2025, 234, 3407)
WIN = 8.0
FPS4 = 4.0
COHORT_SIZE = {"HateMM": 215, "HateClipSeg": 118, "DeHate": 1151}
SPLIT_DIR = Path(os.environ.get("DETWIN_SPLIT_DIR", REPO / "data" / "weaksup_video_splits"))  # smoke override
ASR_DIR = REPO / "data" / "asr_whisper_large_v3"
WAV_DIR = REPO / "data" / "wav16k_mono"
GT_DIR = REPO / "data" / "gt_4fps"
F2_STOP_SHARE = 0.01


# ------------------------------------------------------------------ splits ---
def load_split(ds: str) -> dict:
    """{"train": [{"video_id", "label"}], "val": [...], "test": [{"video_id", "duration"}]} (test has no label)."""
    return json.loads((SPLIT_DIR / f"{ds}.json").read_text())


def samples(ds: str, split: str) -> list[dict]:
    return load_split(ds)[split]


def gt_lengths(ds: str) -> dict[str, int]:
    """Test-split id -> GT frame count. Only array lengths are read."""
    z = np.load(GT_DIR / f"{ds}.npz", allow_pickle=True)
    return {str(v): int(len(z["y4"][i])) for i, v in enumerate(z["video_ids"]) if str(z["split"][i]) == "test"}


# ----------------------------------------------------------------- windows ---
def windows(duration: float, seconds: float = WIN) -> list[tuple[float, float]]:
    """Consecutive windows from 0; the last one may be shorter (spvl.py `fixed_windows`)."""
    n = max(1, int(math.ceil(duration / seconds - 1e-9)))
    return [(i * seconds, min((i + 1) * seconds, duration)) for i in range(n)]


def curve_length(duration: float) -> int:
    return max(1, int(math.ceil(duration * FPS4 - 1e-9)))


def window_curve(scores: list[float], duration: float) -> np.ndarray:
    """4 fps frame i (centre (i + 0.5) / 4 s) takes the score of the window containing its centre; frames past the
    last window take the last window's value. With 8-s windows this is window i // 32."""
    T = curve_length(duration)
    centres = (np.arange(T) + 0.5) / FPS4
    idx = np.clip(np.floor(centres / WIN).astype(int), 0, len(scores) - 1)
    return np.asarray(scores, dtype=np.float64)[idx]


# --------------------------------------------------------------- transcripts ---
def load_asr(ds: str, durations: dict[str, float] | None = None) -> dict[str, list[tuple[float, float, str]]]:
    """video id -> [(start, end, text)] from `all_splits_chunks.jsonl` (Whisper large-v3 segments).

    Untimed chunks (end null) are kept: end = next chunk's start, else the video duration (run_plan.md §1.3), else
    start + 30 s (Whisper's window) when no duration is known. Rows with an error have no segments."""
    out = {}
    path = ASR_DIR / ds / "all_splits_chunks.jsonl"
    with path.open() as fh:
        for line in fh:
            r = json.loads(line)
            chunks = [c for c in (r.get("chunks") or []) if c.get("start") is not None]
            segs = []
            for k, c in enumerate(chunks):
                s = float(c["start"])
                e = c.get("end")
                if e is None:
                    if k + 1 < len(chunks):
                        e = float(chunks[k + 1]["start"])
                    elif durations and r["video_id"] in durations:
                        e = float(durations[r["video_id"]])
                    elif r.get("wav_duration"):
                        e = float(r["wav_duration"])
                    else:
                        e = s + 30.0
                e = float(e)
                if e > s:
                    segs.append((s, e, c.get("text") or ""))
            out[r["video_id"]] = segs
    return out


def load_asr_text(ds: str) -> dict[str, str]:
    """video id -> full transcript text (Whisper's own `text` field)."""
    out = {}
    with (ASR_DIR / ds / "all_splits_chunks.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            out[r["video_id"]] = (r.get("text") or "").strip()
    return out


def _slice_words(s: float, e: float, text: str, lo: float, hi: float) -> str:
    words = text.split()
    if not words:
        return ""
    a = int(round((lo - s) / (e - s) * len(words)))
    b = int(round((hi - s) / (e - s) * len(words)))
    return " ".join(words[a:b]).strip()


def window_text(segments, t1: float, t2: float) -> str:
    """Transcript inside [t1, t2]: segments are sliced proportionally at word boundaries (spvl.py `window_text`)."""
    parts = []
    for s, e, text in segments:
        lo, hi = max(s, t1), min(e, t2)
        if hi <= lo or not text:
            continue
        if s >= t1 and e <= t2:
            parts.append(text)
            continue
        piece = _slice_words(s, e, text, lo, hi)
        if piece:
            parts.append(piece)
    return " ".join(p.strip() for p in parts if p.strip())


def window_segments(segments, t1: float, t2: float) -> list[dict]:
    """Segments restricted to [t1, t2], times relative to t1, text sliced like `window_text`.
    Output uses Whisper's segment keys (id, start, end, text) so the CLARA clip builder can take it unchanged."""
    out = []
    for k, (s, e, text) in enumerate(segments):
        lo, hi = max(s, t1), min(e, t2)
        if hi <= lo:
            continue
        piece = text.strip() if (s >= t1 and e <= t2) else _slice_words(s, e, text, lo, hi)
        out.append({"id": k, "start": lo - t1, "end": hi - t1, "text": piece})
    return out


# ------------------------------------------------------------------- media ---
def video_path(ds: str, vid: str, split: str) -> Path | None:
    home = Path.home() / "data"
    if ds == "HateMM":
        cands = [home / "HateMM/video" / f"{vid}.mp4"]
    elif ds == "HateClipSeg":
        cands = [home / "HateClipSeg/videos" / f"{vid}.{ext}" for ext in ("mp4", "webm", "mkv")]
        cands += [home / "HateClipSeg/video" / f"{vid}.mp4"]
    elif ds == "DeHate":
        cands = [home / "DeHate" / split / f"{vid}.mp4"]
    else:
        raise ValueError(ds)
    for c in cands:
        if c.is_file():
            return c
    return None


def wav_path(ds: str, vid: str) -> Path:
    return WAV_DIR / ds / f"{vid}.wav"


def extract_wav(mp4: Path, wav: Path, sr: int = 16000) -> str | None:
    """CLARA `get_audio.py` command: ffmpeg -vn -ar 16000 -c:a pcm_s16le -ac 1. Returns an error string or None."""
    wav.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(wav) + ".tmp.wav"
    cmd = ["ffmpeg", "-nostdin", "-y", "-i", str(mp4), "-vn", "-ar", str(sr), "-c:a", "pcm_s16le", "-ac", "1",
           "-f", "wav", tmp]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        if os.path.exists(tmp):
            os.remove(tmp)
        return (p.stderr.strip().splitlines() or ["ffmpeg failed"])[-1]
    os.replace(tmp, wav)
    return None


def load_wav(wav: Path):
    """16 kHz mono float32 numpy array, or None if the file is missing / unreadable."""
    import soundfile as sf
    if not wav.is_file():
        return None
    try:
        x, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    except Exception:
        return None
    assert sr == 16000, (wav, sr)
    return x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]


def probe_video(path: Path) -> tuple[float, int]:
    """cv2 CAP_PROP_FPS (30 if invalid, as CLARA get_frame.py) and CAP_PROP_FRAME_COUNT (may be <= 0 for webm)."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if not fps or fps != fps or fps <= 0 or fps > 1000:
        fps = 30.0
    return float(fps), n


def _read_frames_pyav(path: Path, wanted: set[int], tf) -> tuple[dict, int]:
    """Fallback for streams cv2's bundled FFmpeg cannot decode (AV1 webm/mkv in HateClipSeg): PyAV (libdav1d),
    frames in decode order, converted to BGR like cv2."""
    import av
    out: dict = {}
    last_needed = max(wanted)
    n = 0
    last = None
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        for fr in c.decode(st):
            if n in wanted:
                out[n] = tf(fr.to_ndarray(format="bgr24"))
            last = fr
            n += 1
            if n > last_needed:
                break
    if any(w not in out for w in wanted) and last is not None:
        lv = tf(last.to_ndarray(format="bgr24"))
        for w in wanted:
            out.setdefault(w, lv)
    return out, n


def read_frames(path: Path, wanted: set[int], transform=None) -> tuple[dict, int]:
    """Sequential cv2 read (as both authors' extractors); returns {index: transform(BGR frame)} for the wanted
    indices and the number of frames actually decoded. Unwanted frames are only grabbed (decoded, not converted).
    A wanted index past the last decodable frame gets the last decoded frame (the stream ended early); callers
    record how many indices were clamped this way."""
    import cv2
    tf = transform or (lambda x: x)
    out: dict = {}
    if not wanted:
        return out, 0
    last_needed = max(wanted)
    cap = cv2.VideoCapture(str(path))
    i = 0
    last_i, last_val = None, None
    while i <= last_needed:
        if i in wanted:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            out[i] = tf(frame)
            last_i, last_val = i, out[i]
        else:
            if not cap.grab():
                break
        i += 1
    cap.release()
    n_decoded = i
    if n_decoded == 0:
        return _read_frames_pyav(path, wanted, tf)
    if any(w not in out for w in wanted):
        if last_val is None or (last_i is not None and last_i < n_decoded - 1):
            # the last decoded frame was only grabbed; re-read it
            cap = cv2.VideoCapture(str(path))
            j, frame_last = 0, None
            while j < n_decoded:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frame_last = frame
                j += 1
            cap.release()
            if frame_last is not None:
                last_val = tf(frame_last)
        if last_val is not None:
            for w in wanted:
                if w not in out:
                    out[w] = last_val
    return out, n_decoded


def jpeg_roundtrip(bgr: np.ndarray, quality: int = 95) -> bytes:
    """Both authors save sampled frames as JPEG (cv2.imwrite, quality 95) and read them back; this keeps that step
    in memory."""
    import cv2
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    assert ok
    return buf.tobytes()


# ------------------------------------------------------------------ logging ---
class RunLog:
    """run.log whose first line is the host name."""

    def __init__(self, path: Path, append: bool = True):
        path.parent.mkdir(parents=True, exist_ok=True)
        fresh = not (append and path.exists() and path.stat().st_size > 0)
        self.fh = open(path, "a" if append else "w")
        if fresh:
            self.fh.write(f"host {socket.gethostname()}\n")
            self.fh.flush()

    def __call__(self, msg: str) -> None:
        line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
        self.fh.write(line + "\n")
        self.fh.flush()
        print(line, flush=True)


def code_version() -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                                text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/20261008_baselines/detwin",
                                "experiments/20261008_baselines/sage", "experiments/20261008_baselines/clara"],
                               cwd=REPO, capture_output=True, text=True).stdout.strip()
    except Exception:
        commit, dirty = "unknown", ""
    return (f"commit {commit}{' + uncommitted changes' if dirty else ''}, "
            f"{datetime.date.today().isoformat()}")


# ----------------------------------------------------------------- finalise ---
def finalize(method: str, ds: str, seed: int, out_dir: Path, window_scores: dict[str, list[float]],
             *, code_path: str, config: dict, log: RunLog, f1_windows: dict[str, list[int]] | None = None,
             failures: dict[str, str] | None = None) -> dict:
    """window_scores: cohort video id -> P(hateful) per 8-s window (len == len(windows(duration))).
    f1_windows: video id -> indices of windows that got the tail rule (F1, previous window's value).
    failures: video id -> reason for videos with no usable output (F2 candidates).
    Writes predictions.jsonl, coverage.json, config.json and, if the exact-cohort rule holds, metrics.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    split = load_split(ds)
    dur = {r["video_id"]: float(r["duration"]) for r in split["test"]}
    ids = sorted(dur)
    t_gt = gt_lengths(ds)
    assert len(ids) == COHORT_SIZE[ds] and set(ids) <= set(t_gt), (ds, len(ids))
    failures = dict(failures or {})
    f1_windows = f1_windows or {}
    curves, bad = {}, {}
    for v in ids:
        w = window_scores.get(v)
        if w is None:
            bad[v] = failures.get(v, "no_output")
            continue
        nw = len(windows(dur[v]))
        if len(w) != nw:
            bad[v] = f"{len(w)} window scores for {nw} windows"
            continue
        c = window_curve(w, dur[v])
        T = t_gt[v]
        if len(c) < T:
            bad[v] = f"curve shorter than GT ({len(c)} < {T})"
        elif not np.isfinite(c[:T]).all():
            bad[v] = f"{int((~np.isfinite(c[:T])).sum())} non-finite GT frames"
        else:
            curves[v] = c
    report = {"method": method, "dataset": ds, "seed": seed, "cohort_videos": len(ids),
              "videos_scored": len(curves), "f2_videos": bad, "f2_share": len(bad) / len(ids),
              "f1_tail_windows": {v: x for v, x in f1_windows.items() if x},
              "n_windows": int(sum(len(windows(dur[v])) for v in ids))}
    extra = {v: {"window_scores": [float(x) for x in window_scores[v]]} for v in curves}
    for v, x in f1_windows.items():
        if x and v in extra:
            extra[v]["fallback_F1_windows"] = x
    if bad:
        if len(bad) > F2_STOP_SHARE * len(ids):
            report["exact_test_set"] = False
            report["stopped"] = f"F2 needed for {len(bad)}/{len(ids)} videos (> 1 %); not evaluated"
            (out_dir / "coverage.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            log(f"FAILED exact-cohort rule: {report['stopped']}: {bad}")
            return report
        med = float(np.median(np.concatenate(list(curves.values()))))
        report["f2_constant"] = med
        for v, why in bad.items():
            log(f"F2 {ds}/{v}: {why} -> constant {med:.6g}")
            curves[v] = np.full(curve_length(dur[v]), med)
            extra[v] = {"fallback": {"code": "F2", "reason": why, "value": med}}
    report["exact_test_set"] = True
    pred = out_dir / "predictions.jsonl"
    with pred.open("w") as fh:
        for v in ids:
            row = {"schema_version": 1, "method": method, "dataset": ds, "video_id": v, "duration": dur[v],
                   "native_rate": 1.0 / WIN, "score_curve": [float(x) for x in curves[v]], "intervals": [],
                   "error": None, "calls": None, "seed": seed, "code_path": code_path, "extra": extra[v]}
            fh.write(json.dumps(row) + "\n")
    cfg = {"exp_id": EXP_ID, "method": method, "dataset": ds, "seed": seed,
           "date": datetime.date.today().isoformat(), "host": socket.gethostname(), "code": code_path,
           "code_version": code_version(), **config}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    metrics = out_dir / "metrics.json"
    cmd = [sys.executable, "-m", "src.eval.evaluate_four_datasets", "--predictions", str(pred),
           "--gt-dir", str(GT_DIR), "--out", str(metrics), "--datasets", ds]
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(REPO)})
    if p.returncode != 0:
        log(f"FAILED evaluator: {p.stderr[-2000:]}")
        report["evaluator_error"] = p.stderr[-2000:]
    else:
        m = json.loads(metrics.read_text())
        row = [r for r in m["per_dataset"] if r["dataset"] == ds][0]
        report["evaluator_n_videos_overlap"] = row["n_videos_overlap"]
        report["evaluator_n_videos_predicted"] = row["n_videos_predicted"]
        if row["n_videos_overlap"] != COHORT_SIZE[ds] or row["n_videos_predicted"] != COHORT_SIZE[ds]:
            report["exact_test_set"] = False
            log(f"FAILED evaluator saw {row['n_videos_overlap']} / {row['n_videos_predicted']} videos")
        log(f"metrics {ds} seed {seed}: ROC {row['frame_ROC_AUC']:.4f} PR {row['frame_PR_AUC']:.4f} "
            f"within {row['within_video_macro_ROC_AUC']:.4f} (n={row['n_videos_overlap']})")
    (out_dir / "coverage.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
