from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .common import DATASETS, DEFAULT_ORDER, RESULT_ROOT, parse_yes_no, read_jsonl, required_judge_ids
    from .data_utils import PROCESSED_ROOT
except ImportError:
    from common import DATASETS, DEFAULT_ORDER, RESULT_ROOT, parse_yes_no, read_jsonl, required_judge_ids
    from data_utils import PROCESSED_ROOT


def fail(msg: str) -> None:
    raise SystemExit(f"[invalid] {msg}")


def check_processed(dataset: str) -> dict:
    out = {}
    for split in ("train", "test"):
        rows = read_jsonl(PROCESSED_ROOT / dataset / f"{split}.jsonl")
        if not rows:
            fail(f"{dataset}/{split}: processed jsonl is empty")
        ids = [r.get("id") for r in rows]
        if len(ids) != len(set(ids)):
            fail(f"{dataset}/{split}: duplicate ids")
        bad_labels = [r.get("id") for r in rows if r.get("label") not in (0, 1)]
        if bad_labels:
            fail(f"{dataset}/{split}: bad labels, first={bad_labels[:3]}")
        missing = [r.get("id") for r in rows if not Path(r.get("image_path", "")).is_file()]
        if missing:
            fail(f"{dataset}/{split}: missing images {len(missing)}, first={missing[:3]}")
        out[split] = {"n": len(rows), "n_pos": sum(int(r["label"]) for r in rows)}
    return out


def check_stage1(dataset: str, slug: str, allow_incomplete: bool = False) -> dict:
    out = {}
    for split in ("train", "test"):
        expected = len(read_jsonl(PROCESSED_ROOT / dataset / f"{split}.jsonl"))
        rows = read_jsonl(RESULT_ROOT / f"holistic_{slug}" / dataset / f"{split}_binary.jsonl")
        ids = [r.get("id") for r in rows]
        scored = [r for r in rows if r.get("score") is not None]
        bad_scores = [r.get("id") for r in scored if not (0.0 <= float(r["score"]) <= 1.0)]
        if len(ids) != len(set(ids)):
            fail(f"{dataset}/{split}: duplicate Stage-1 ids")
        if not allow_incomplete and len(rows) != expected:
            fail(f"{dataset}/{split}: Stage-1 row count {len(rows)} != expected {expected}")
        if not allow_incomplete and len(scored) != expected:
            fail(f"{dataset}/{split}: Stage-1 scored count {len(scored)} != expected {expected}")
        if bad_scores:
            fail(f"{dataset}/{split}: Stage-1 bad scores, first={bad_scores[:3]}")
        out[split] = {"expected": expected, "rows": len(rows), "scored": len(scored), "missing": max(expected - len(rows), 0)}
    return out


def check_judges(dataset: str, tags: tuple[str, ...], allow_incomplete: bool = False) -> dict:
    expected_full = len(read_jsonl(PROCESSED_ROOT / dataset / "test.jsonl"))
    band_rows = read_jsonl(RESULT_ROOT / "boundary" / dataset / "candidates_entropy_band.jsonl")
    out = {}
    for tag in tags:
        required_ids = required_judge_ids(dataset, tag)
        if not required_ids:
            required_ids = {r.get("id") for r in band_rows if r.get("in_band") is True}
        expected_required = len(required_ids) if required_ids else expected_full
        rows = read_jsonl(RESULT_ROOT / "judges" / dataset / f"test_{tag}.jsonl")
        ids = [r.get("id") for r in rows]
        if len(ids) != len(set(ids)):
            fail(f"{dataset}/{tag}: duplicate judge ids")
        effective_preds = [r.get("pred") if r.get("pred") in (0, 1) else parse_yes_no(r.get("raw_response")) for r in rows]
        bad = [r.get("id") for r, pred in zip(rows, effective_preds) if pred not in (0, 1, None)]
        if bad:
            fail(f"{dataset}/{tag}: illegal judge pred, first={bad[:3]}")
        invalid = [r.get("id") for r, pred in zip(rows, effective_preds) if pred is None]
        if invalid and not allow_incomplete:
            fail(f"{dataset}/{tag}: unparsed judge predictions {len(invalid)}, first={invalid[:3]}")
        covered_required = required_ids & set(ids)
        if not allow_incomplete and len(covered_required) != expected_required:
            missing = sorted(required_ids - set(ids))[:3]
            fail(f"{dataset}/{tag}: judge required coverage {len(covered_required)} != expected {expected_required}, first_missing={missing}")
        out[tag] = {
            "expected_full": expected_full,
            "expected_required": expected_required,
            "rows": len(rows),
            "valid_pred": sum(1 for pred in effective_preds if pred in (0, 1)),
            "required_covered": len(covered_required),
            "missing_required": max(expected_required - len(covered_required), 0),
        }
    return out


def check_boundary(dataset: str) -> dict:
    base = read_jsonl(RESULT_ROOT / "boundary" / dataset / "baseline_preds.jsonl")
    band = read_jsonl(RESULT_ROOT / "boundary" / dataset / "candidates_entropy_band.jsonl")
    if not base or not band:
        fail(f"{dataset}: boundary files missing or empty")
    if len(base) != len(band):
        fail(f"{dataset}: boundary base/band length mismatch {len(base)} vs {len(band)}")
    bad = [r.get("id") for r in base if r.get("pred_baseline") not in (0, 1)]
    if bad:
        fail(f"{dataset}: illegal baseline pred, first={bad[:3]}")
    return {"rows": len(base), "n_in_band": sum(1 for r in band if r.get("in_band") is True)}


def check_final(dataset: str) -> dict:
    rows = read_jsonl(RESULT_ROOT / "final" / dataset / "test_sequential.jsonl")
    expected = len(read_jsonl(RESULT_ROOT / "boundary" / dataset / "baseline_preds.jsonl"))
    if not rows:
        fail(f"{dataset}: final output missing or empty")
    if expected and len(rows) != expected:
        fail(f"{dataset}: final length mismatch {len(rows)} vs {expected}")
    bad = [r.get("id") for r in rows if r.get("pred_final") not in (0, 1)]
    if bad:
        fail(f"{dataset}: illegal final pred, first={bad[:3]}")
    return {"rows": len(rows), "n_in_band": sum(1 for r in rows if r.get("in_band") is True)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate meme variant intermediate outputs")
    parser.add_argument("--dataset", default="all", choices=("all", *DATASETS))
    parser.add_argument("--stage", default="all", choices=("all", "processed", "stage1", "judges", "boundary", "final"))
    parser.add_argument("--model-slug", default="2b")
    parser.add_argument("--judge-tags", default=",".join(DEFAULT_ORDER))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    tags = tuple(x.strip() for x in args.judge_tags.split(",") if x.strip())
    report = {}
    for ds in datasets:
        report[ds] = {}
        if args.stage in ("all", "processed"):
            report[ds]["processed"] = check_processed(ds)
        if args.stage in ("all", "stage1"):
            report[ds]["stage1"] = check_stage1(ds, args.model_slug, allow_incomplete=args.allow_incomplete)
        if args.stage in ("all", "judges"):
            report[ds]["judges"] = check_judges(ds, tags, allow_incomplete=args.allow_incomplete)
        if args.stage in ("all", "boundary"):
            report[ds]["boundary"] = check_boundary(ds)
        if args.stage in ("all", "final"):
            report[ds]["final"] = check_final(ds)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
