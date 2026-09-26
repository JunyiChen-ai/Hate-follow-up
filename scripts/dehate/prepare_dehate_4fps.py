#!/usr/bin/env python3
"""DeHate inputs for external validation on the 4 fps protocol (experiments/20260927_dehate_external/README.md).

Stages (CPU, uoa-lab1):
  manifest  data/manifests/DeHate_test.jsonl: official test split, container duration by ffprobe.
  asr       data/asr_whisper_large_v3/DeHate/timestamped_chunks.jsonl: the test rows of the Retrieval-hate DeHate
            Whisper large-v3 transcripts, copied unchanged.
  gt        data/gt_4fps/DeHate.npz (+ report_DeHate.json): the rasterization of data/gt_4fps (frame i at t = i / 4 is
            positive when start <= t < end, floor(duration * 4) frames). Spans are parsed as in the Retrieval-hate
            DeHate amendment: every "(a, b)" pair, or the one row written "[a, b]"; end <= start is dropped. A hateful
            video with no usable span is excluded (rule (b): text-only hate, not on the timeline).
Only the gt stage reads labels; the manifest and transcripts carry none.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = Path.home() / "data/DeHate"
LABELS = DATA / "DeHate_labels.csv"
VIDEO_DIR = DATA / "test"
ASR_SRC = Path.home() / "Retrieval-hate/results/reproduction/asr/dehate_all/timestamped_chunks.jsonl"
MANIFEST = ROOT / "data/manifests/DeHate_test.jsonl"
ASR_OUT = ROOT / "data/asr_whisper_large_v3/DeHate/timestamped_chunks.jsonl"
GT_DIR = ROOT / "data/gt_4fps"
FPS = 4.0

_PAIR = re.compile(r"\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)")
_BARE = re.compile(r"^\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]$")


def test_rows():
    csv.field_size_limit(1 << 30)
    with open(LABELS, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["Split"] == "test"]
    ids = [r["Video ID"].strip() for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate DeHate test ids")
    return rows


def parse_spans(text):
    text = (text or "").strip()
    pairs = [(float(a), float(b)) for a, b in _PAIR.findall(text)]
    if not pairs:
        m = _BARE.match(text)
        if m:
            pairs = [(float(m.group(1)), float(m.group(2)))]
    return pairs


def probe(path):
    """Container duration in seconds (format), else the longest stream; None if unreadable."""
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,duration",
                        "-of", "json", str(path)], capture_output=True, text=True)
    if r.returncode != 0:
        return None, None
    info = json.loads(r.stdout or "{}")
    streams = info.get("streams", [])
    has_video = any(s.get("codec_type") == "video" for s in streams)
    try:
        return float(info["format"]["duration"]), has_video
    except (KeyError, TypeError, ValueError):
        ds = [float(s["duration"]) for s in streams if s.get("duration") not in (None, "N/A")]
        return (max(ds) if ds else None), has_video


def manifest(workers):
    rows = test_rows()

    def one(r):
        vid = r["Video ID"].strip()
        path = VIDEO_DIR / f"{vid}.mp4"
        if not path.is_file():
            return vid, str(path), None, None
        dur, has_video = probe(path)
        return vid, str(path), dur, has_video

    with ThreadPoolExecutor(workers) as ex:
        out = list(ex.map(one, rows))
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    bad = []
    with open(MANIFEST, "w") as fh:
        for vid, path, dur, has_video in out:
            if not dur:
                bad.append(vid)
                continue
            fh.write(json.dumps({"dataset": "DeHate", "video_id": vid, "duration": round(dur, 3),
                                 "video_path": path, "has_video_stream": bool(has_video)}) + "\n")
    print(f"manifest {MANIFEST}: {len(out) - len(bad)} videos; no duration {len(bad)} {bad[:10]}; "
          f"no video stream {sum(1 for x in out if x[2] and not x[3])}")


def asr():
    ids = {r["Video ID"].strip() for r in test_rows()}
    kept = []
    for line in open(ASR_SRC):
        rec = json.loads(line)
        if rec["video_id"] in ids:
            kept.append(line if line.endswith("\n") else line + "\n")
    ASR_OUT.parent.mkdir(parents=True, exist_ok=True)
    ASR_OUT.write_text("".join(kept))
    got = {json.loads(x)["video_id"] for x in kept}
    errors = sum(1 for x in kept if json.loads(x).get("error"))
    print(f"asr {ASR_OUT}: {len(kept)} rows for {len(ids)} test ids; missing {len(ids - got)}; error rows {errors}")


def rasterize(spans, duration):
    n = max(1, int(np.floor(float(duration) * FPS)))
    t = np.arange(n, dtype=float) / FPS
    y = np.zeros(n, dtype=np.int8)
    clean = []
    for start, end in spans:
        start, end = max(0.0, float(start)), min(float(duration), float(end))
        if end > start:
            y[(t >= start) & (t < end)] = 1
            clean.append([start, end])
    clean.sort()
    merged = []
    for start, end in clean:
        if merged and start <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return y, np.asarray(merged, dtype=float).reshape(-1, 2)


def gt():
    media = {json.loads(l)["video_id"]: json.loads(l) for l in open(MANIFEST) if l.strip()}
    video_ids, durations, labels, spans_out = [], [], [], []
    excluded, no_duration, dropped = [], [], 0
    for r in test_rows():
        vid = r["Video ID"].strip()
        if vid not in media:
            no_duration.append(vid)
            continue
        dur = float(media[vid]["duration"])
        spans = []
        if r["Hate"] == "1":
            raw = parse_spans(r["Hate Segment"])
            spans = [(a, b) for a, b in raw if b > a]
            dropped += len(raw) - len(spans)
            if not spans:
                excluded.append(vid)
                continue
        y, merged = rasterize(spans, dur)
        if r["Hate"] == "1" and y.sum() == 0:
            excluded.append(vid)  # every span lies past the end of the media
            continue
        video_ids.append(vid); durations.append(dur); labels.append(y); spans_out.append(merged)
    obj = lambda values: np.asarray(values + [None], dtype=object)[:-1]
    np.savez_compressed(GT_DIR / "DeHate.npz", video_ids=np.asarray(video_ids),
                        split=np.asarray(["test"] * len(video_ids)), duration=np.asarray(durations),
                        y4=obj(labels), spans=obj(spans_out),
                        n_spans=np.asarray([len(x) for x in spans_out], dtype=np.int16))
    n_frames = sum(map(len, labels)); n_pos = sum(int(x.sum()) for x in labels)
    report = {"dataset": "DeHate", "split": "test", "n_videos": len(video_ids),
              "n_hateful": int(sum(1 for y in labels if y.any())), "n_frames": n_frames, "n_positive": n_pos,
              "base_rate": n_pos / max(1, n_frames), "degenerate_spans_dropped": dropped,
              "n_videos_with_both_classes": int(sum(1 for y in labels if 0 < y.sum() < len(y))),
              "excluded_positive_without_span": sorted(excluded), "missing_duration": no_duration}
    (GT_DIR / "report_DeHate.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "excluded_positive_without_span"}, indent=1),
          "excluded", len(excluded))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["manifest", "asr", "gt"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    {"manifest": lambda: manifest(a.workers), "asr": asr, "gt": gt}[a.stage]()


if __name__ == "__main__":
    main()
