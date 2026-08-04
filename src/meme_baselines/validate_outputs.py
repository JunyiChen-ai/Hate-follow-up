from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import DATASETS, DATA_ROOT, RESULT_ROOT, dataset_iter, load_split, read_jsonl


def fail(msg: str) -> None:
    raise SystemExit(f"[invalid] {msg}")


def check_processed(dataset: str) -> dict:
    out = {}
    for split in ("train", "test"):
        rows = load_split(dataset, split)
        if not rows:
            fail(f"{dataset}/{split}: empty processed file")
        ids = [r.get("id") for r in rows]
        if len(ids) != len(set(ids)):
            fail(f"{dataset}/{split}: duplicate ids")
        missing = [r["id"] for r in rows if not Path(r.get("image_path", "")).is_file()]
        if missing:
            fail(f"{dataset}/{split}: missing images {len(missing)}, first={missing[:3]}")
        bad_labels = [r["id"] for r in rows if r.get("label") not in (0, 1)]
        if bad_labels:
            fail(f"{dataset}/{split}: bad labels first={bad_labels[:3]}")
        out[split] = {"n": len(rows), "n_pos": sum(int(r["label"]) for r in rows)}
    return out


def check_result(dataset: str, baseline: str, filename: str, allow_incomplete: bool = False) -> dict:
    expected = len(load_split(dataset, "test"))
    path = RESULT_ROOT / baseline / dataset / filename
    rows = read_jsonl(path)
    ids = [r.get("id") for r in rows]
    if len(ids) != len(set(ids)):
        fail(f"{path}: duplicate ids")
    bad_labels = [r.get("id") for r in rows if r.get("label") not in (0, 1)]
    if bad_labels:
        fail(f"{path}: bad labels first={bad_labels[:3]}")
    bad_preds = [r.get("id") for r in rows if r.get("pred") not in (0, 1, -1, None)]
    if bad_preds:
        fail(f"{path}: illegal pred first={bad_preds[:3]}")
    if not allow_incomplete and len(rows) != expected:
        fail(f"{path}: row count {len(rows)} != expected {expected}")
    valid = sum(1 for r in rows if r.get("pred") in (0, 1))
    return {"path": str(path), "expected": expected, "rows": len(rows), "valid_pred": valid, "missing": max(expected - len(rows), 0)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", choices=("all", *DATASETS))
    parser.add_argument("--stage", default="processed", choices=("processed", "result"))
    parser.add_argument("--baseline")
    parser.add_argument("--filename")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    report = {}
    for ds in dataset_iter(args.dataset):
        if args.stage == "processed":
            report[ds] = check_processed(ds)
        else:
            if not args.baseline or not args.filename:
                parser.error("--baseline and --filename are required for --stage result")
            report[ds] = check_result(ds, args.baseline, args.filename, args.allow_incomplete)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

