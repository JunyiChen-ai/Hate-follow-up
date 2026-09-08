#!/usr/bin/env python3
"""Build the label-free HateMM validation manifest for prospective CCA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--asr", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--developed-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    ids = [line.strip() for line in args.ids.read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate validation IDs")
    asr = {str(row["video_id"]): row for row in map(json.loads, args.asr.open())}
    developed = {str(row["video_id"]) for row in map(json.loads, args.developed_manifest.open())
                 if row["dataset"] == "HateMM"}
    report = {"requested": len(ids), "overlap_developed": [], "missing_asr": [], "missing_media": []}
    rows = []
    for video_id in ids:
        if video_id in developed:
            report["overlap_developed"].append(video_id); continue
        record = asr.get(video_id)
        if not record or not record.get("wav_duration"):
            report["missing_asr"].append(video_id); continue
        media = args.video_dir / f"{video_id}.mp4"
        if not media.exists():
            report["missing_media"].append(video_id); continue
        rows.append({"dataset": "HateMM_val_sealed", "video_id": video_id,
                     "duration": float(record["wav_duration"]),
                     "transcript": str(record.get("text", "")),
                     "video_path": str(media.resolve())})
    if report["overlap_developed"]:
        raise RuntimeError(f"validation/test overlap: {report['overlap_developed'][:3]}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report["written"] = len(rows)
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
