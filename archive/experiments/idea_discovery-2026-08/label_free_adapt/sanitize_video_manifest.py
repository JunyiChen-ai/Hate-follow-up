#!/usr/bin/env python3
"""Transcode videos unsupported by qwen-vl-utils to a decode-safe cache."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def codec(path: str) -> str:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name", "-of", "default=nw=1:nk=1", path,
    ], text=True).strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True)
    args = ap.parse_args()
    cache = Path(args.cache)
    rows = [json.loads(x) for x in Path(args.manifest).read_text().splitlines() if x.strip()]
    output = []
    for index, row in enumerate(rows, 1):
        if codec(row["video_path"]) in {"av1", "vp9"}:
            dst = cache / row["dataset"] / f"{row['video_id']}.mp4"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                subprocess.run([
                    "ffmpeg", "-v", "error", "-y", "-i", row["video_path"],
                    "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "23", "-c:a", "aac", str(dst),
                ], check=True)
            row = dict(row)
            row["video_path"] = str(dst.resolve())
        output.append(row)
        if index % 25 == 0:
            print(f"checked={index}/{len(rows)}", flush=True)
    Path(args.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
