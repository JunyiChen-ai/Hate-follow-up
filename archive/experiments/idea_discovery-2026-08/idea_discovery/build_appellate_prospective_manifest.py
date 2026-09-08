#!/usr/bin/env python3
"""Build a GT-blind prospective manifest for Appellate-PACT.

This builder intentionally accepts only split IDs, ASR records, and media
directories.  It never reads class labels or temporal annotations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_ids(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise RuntimeError(f"duplicate IDs in {path}")
    return values


def read_asr(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    output = {str(row["video_id"]): row for row in rows}
    if len(output) != len(rows):
        raise RuntimeError(f"duplicate ASR IDs in {path}")
    return output


def add_cohort(
    output: list[dict], *, dataset: str, ids_path: Path, asr_path: Path,
    video_dir: Path, limit: int,
) -> dict:
    ids = read_ids(ids_path)
    if limit:
        ids = sorted(ids, key=lambda value: hashlib.sha256(
            f"{dataset}\0{value}".encode()).hexdigest())[:limit]
    asr = read_asr(asr_path)
    report = {"requested": len(ids), "missing_asr": [], "asr_error": [],
              "missing_duration": [], "missing_media": []}
    for video_id in ids:
        row = asr.get(video_id)
        if row is None:
            report["missing_asr"].append(video_id)
            continue
        if row.get("error"):
            report["asr_error"].append(video_id)
            continue
        duration = row.get("wav_duration") or row.get("container_duration")
        if not duration or float(duration) <= 0:
            report["missing_duration"].append(video_id)
            continue
        media = video_dir / f"{video_id}.mp4"
        if not media.is_file():
            report["missing_media"].append(video_id)
            continue
        output.append({
            "dataset": dataset,
            "video_id": video_id,
            "duration": float(duration),
            "transcript": str(row.get("text", "")),
            "video_path": str(media.resolve()),
            "split_source": str(ids_path.resolve()),
            "gt_access": False,
        })
    report["written"] = sum(row["dataset"] == dataset for row in output)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--en-ids", type=Path, required=True)
    ap.add_argument("--en-asr", type=Path, required=True)
    ap.add_argument("--en-video-dir", type=Path, required=True)
    ap.add_argument("--zh-ids", type=Path, required=True)
    ap.add_argument("--zh-asr", type=Path, required=True)
    ap.add_argument("--zh-video-dir", type=Path, required=True)
    ap.add_argument("--limit-per-dataset", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    rows: list[dict] = []
    reports = {
        "MHC_en_train_prospective": add_cohort(
            rows, dataset="MHC_en_train_prospective", ids_path=args.en_ids,
            asr_path=args.en_asr, video_dir=args.en_video_dir,
            limit=args.limit_per_dataset),
        "MHC_zh_train_prospective": add_cohort(
            rows, dataset="MHC_zh_train_prospective", ids_path=args.zh_ids,
            asr_path=args.zh_asr, video_dir=args.zh_video_dir,
            limit=args.limit_per_dataset),
    }
    failures = sum(len(values) for report in reports.values()
                   for key, values in report.items()
                   if key not in {"requested", "written"})
    if failures:
        raise RuntimeError(json.dumps(reports, ensure_ascii=False))
    rows.sort(key=lambda row: (row["dataset"], row["video_id"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"reports": reports, "total": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
