#!/usr/bin/env python3
"""Label-free, frozen-train audit of HateClipSeg visual-media decodability."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import decord

from hate_common import data as hdata
from vera_adapter import video_path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ffprobe_version() -> str:
    result = subprocess.run(["ffprobe", "-version"], check=True,
                            capture_output=True, text=True)
    return result.stdout.splitlines()[0]


def probe(path: Path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=index,codec_type,codec_name", "-of", "json", str(path)],
        check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams", [])
    video_streams = [row for row in streams if row.get("codec_type") == "video"]
    if not video_streams:
        return "no_decodable_visual_stream", None, streams
    try:
        reader = decord.VideoReader(str(path), num_threads=1)
        if len(reader) <= 0 or float(reader.get_avg_fps()) <= 0:
            return "no_decodable_visual_stream", None, streams
        duration = len(reader) / float(reader.get_avg_fps())
    except decord.DECORDError:
        return "no_decodable_visual_stream", None, streams
    return "decodable_visual_stream", duration, streams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    train = hdata.load_split("hateclipseg", "train")
    val = set(hdata.load_split("hateclipseg", "val"))
    test = set(hdata.load_split("hateclipseg", "test"))
    if len(train) != len(set(train)) or set(train) & (val | test):
        raise RuntimeError("frozen HateClipSeg train split invalid")
    records = {}
    for position, vid in enumerate(train, 1):
        path = Path(video_path("hateclipseg", vid)).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        status, duration, streams = probe(path)
        records[vid] = {
            "path": str(path), "media_sha256": sha256(path),
            "size_bytes": path.stat().st_size, "status": status,
            "duration": duration, "streams": streams,
        }
        print(f"[{position}/{len(train)}] {vid}: {status}", flush=True)
    payload = {
        "corpus": "hateclipseg", "split": "train",
        "label_access": "none", "frozen_train_ids": train,
        "probe_policy": "ffprobe video stream exists AND decord opens with frames/fps",
        "ffprobe_version": ffprobe_version(), "decord_version": decord.__version__,
        "records": records,
    }
    out = Path(args.out).resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, out)
    included = sum(row["status"] == "decodable_visual_stream"
                   for row in records.values())
    print(f"wrote {out}: {included} included, {len(train)-included} excluded")


if __name__ == "__main__":
    main()
