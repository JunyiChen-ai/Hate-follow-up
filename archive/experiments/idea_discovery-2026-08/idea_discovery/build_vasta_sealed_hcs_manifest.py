#!/usr/bin/env python3
"""Build the label-free input manifest for frozen p11 train+val validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--media-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")

    split = json.loads(args.split.read_text())
    ids = list(split["train"]) + list(split["val"])
    if len(ids) != len(set(ids)) or set(ids) & set(split["test"]):
        raise RuntimeError("p11 train/val IDs are duplicated or overlap test")
    chunks = {row["video_id"]: row
              for row in map(json.loads, args.chunks.open())}
    rows, missing = [], []
    for video_id in ids:
        media = sorted(args.media_dir.glob(video_id + ".*"))
        if len(media) != 1 or video_id not in chunks:
            missing.append({"video_id": video_id, "n_media": len(media),
                            "has_chunks": video_id in chunks})
            continue
        source = chunks[video_id]
        duration = source.get("container_duration") or source.get("wav_duration")
        if not duration or float(duration) <= 0:
            missing.append({"video_id": video_id, "reason": "bad duration"})
            continue
        rows.append({
            "dataset": "HateClipSeg_sealed",
            "duration": float(duration),
            "transcript": str(source.get("text", "")),
            "video_id": video_id,
            "video_path": str(media[0].resolve()),
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"requested": len(ids), "written": len(rows),
                      "missing": missing}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
