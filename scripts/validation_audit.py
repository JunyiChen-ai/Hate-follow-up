#!/usr/bin/env python3
"""Validation-run audits for split, frame, and JSONL outputs."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path("/data/jehc223/EMNLP3")
sys.path.insert(0, str(PROJECT_ROOT / "src" / "our_method"))

from data_utils import (  # noqa: E402
    DATASET_ROOTS,
    SKIP_VIDEOS,
    generate_clean_splits,
    load_annotations,
    load_clean_split_ids,
)

ALL_DATASETS = ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def frame_count(dataset: str, vid: str) -> int:
    folder = Path(DATASET_ROOTS[dataset]) / "frames_16" / vid
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def prepare_splits(datasets: list[str]) -> int:
    for ds in datasets:
        print(f"[split] {ds}")
        generate_clean_splits(ds, splits=["validation"])
    return 0


def audit_frames(datasets: list[str], split: str, require_complete: bool) -> int:
    exit_code = 0
    for ds in datasets:
        ids = load_clean_split_ids(ds, split)
        ann = load_annotations(ds)
        counts = {v: frame_count(ds, v) for v in ids if v not in SKIP_VIDEOS.get(ds, set())}
        complete = [v for v, n in counts.items() if n >= 16]
        missing = [v for v, n in counts.items() if n == 0]
        partial = [v for v, n in counts.items() if 0 < n < 16]
        print(
            f"[frames] {ds}/{split}: expected={len(ids)} "
            f"ann={sum(1 for v in ids if v in ann)} "
            f"skip={sum(1 for v in ids if v in SKIP_VIDEOS.get(ds, set()))} "
            f"complete16={len(complete)} missing={len(missing)} partial={len(partial)}"
        )
        if missing:
            print(f"  missing_sample={missing[:10]}")
        if partial:
            print(f"  partial_sample={[(v, counts[v]) for v in partial[:10]]}")
        if require_complete and (missing or partial):
            exit_code = 2
    return exit_code


def read_jsonl(path: Path):
    rows = []
    bad = []
    if not path.exists():
        return rows, [(0, "missing file")]
    with path.open() as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                bad.append((i, str(exc)))
    return rows, bad


def audit_jsonl(dataset: str, split: str, path: Path, skip_file: Path | None) -> int:
    expected = load_clean_split_ids(dataset, split)
    expected_set = set(expected)
    skip_set = set()
    if skip_file and skip_file.exists():
        skip_set = {line.strip() for line in skip_file.read_text().splitlines() if line.strip()}
    rows, bad = read_jsonl(path)
    ids = [r.get("video_id") for r in rows if r.get("video_id")]
    dupes = sorted([vid for vid, n in Counter(ids).items() if n > 1])
    outside = sorted([vid for vid in ids if vid not in expected_set])
    pred_dist = Counter(r.get("pred") for r in rows)
    valid_pred = sum(1 for r in rows if r.get("pred") in (0, 1))
    covered = set(ids) | skip_set
    missing = [v for v in expected if v not in covered]
    print(
        f"[jsonl] {dataset}/{split}: path={path} rows={len(rows)} "
        f"valid_pred={valid_pred} bad_json={len(bad)} dupes={len(dupes)} "
        f"outside={len(outside)} missing={len(missing)} crash_skip={len(skip_set)} "
        f"pred_dist={dict(pred_dist)}"
    )
    if bad:
        print(f"  bad_json_sample={bad[:5]}")
    if dupes:
        print(f"  duplicate_sample={dupes[:10]}")
    if outside:
        print(f"  outside_sample={outside[:10]}")
    if missing:
        print(f"  missing_sample={missing[:10]}")
    invalid_pred = len(rows) - valid_pred
    if invalid_pred:
        bad_pred_sample = [
            {"video_id": r.get("video_id"), "pred": r.get("pred")}
            for r in rows
            if r.get("pred") not in (0, 1)
        ][:10]
        print(f"  invalid_pred_sample={bad_pred_sample}")
    return 0 if not bad and not dupes and not outside and not missing and not invalid_pred else 2


def audit_captions(dataset: str, split: str, path: Path) -> int:
    expected = load_clean_split_ids(dataset, split)
    if not path.exists():
        print(f"[captions] {dataset}/{split}: missing {path}")
        return 2
    try:
        with path.open("rb") as f:
            payload = pickle.load(f)
    except Exception as exc:
        print(f"[captions] {dataset}/{split}: unreadable {path}: {exc}")
        return 2
    expected_set = set(expected)
    keys = set(payload)
    missing = [v for v in expected if v not in keys and v not in SKIP_VIDEOS.get(dataset, set())]
    outside = sorted(keys - expected_set)
    empty_records = [
        vid for vid, rec in payload.items()
        if vid in expected_set and (not isinstance(rec, dict) or not any((v or "").strip() for v in rec.values()))
    ]
    print(
        f"[captions] {dataset}/{split}: path={path} records={len(payload)} "
        f"missing={len(missing)} outside={len(outside)} empty={len(empty_records)}"
    )
    if missing:
        print(f"  missing_sample={missing[:10]}")
    if outside:
        print(f"  outside_sample={outside[:10]}")
    if empty_records:
        print(f"  empty_sample={empty_records[:10]}")
    return 0 if not missing and not outside and not empty_records else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare-splits")
    p.add_argument("--dataset", choices=ALL_DATASETS)
    p.add_argument("--all", action="store_true")

    p = sub.add_parser("frames")
    p.add_argument("--dataset", choices=ALL_DATASETS)
    p.add_argument("--all", action="store_true")
    p.add_argument("--split", default="validation")
    p.add_argument("--require-complete", action="store_true")

    p = sub.add_parser("jsonl")
    p.add_argument("--dataset", required=True, choices=ALL_DATASETS)
    p.add_argument("--split", default="validation")
    p.add_argument("--path", required=True)
    p.add_argument("--skip-file")

    p = sub.add_parser("captions")
    p.add_argument("--dataset", required=True, choices=ALL_DATASETS)
    p.add_argument("--split", default="validation")
    p.add_argument("--path", required=True)

    args = parser.parse_args()
    if args.cmd in {"prepare-splits", "frames"}:
        if not args.all and not args.dataset:
            parser.error("Provide --dataset or --all")
        datasets = ALL_DATASETS if args.all else [args.dataset]
        if args.cmd == "prepare-splits":
            return prepare_splits(datasets)
        return audit_frames(datasets, args.split, args.require_complete)
    if args.cmd == "captions":
        return audit_captions(args.dataset, args.split, Path(args.path))
    return audit_jsonl(
        args.dataset,
        args.split,
        Path(args.path),
        Path(args.skip_file) if args.skip_file else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
