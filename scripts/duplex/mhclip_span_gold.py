#!/usr/bin/env python3
"""Build MultiHateClip temporal-span gold files for the localization probe.

Upstream MultiHateClip (https://github.com/Social-AI-Studio/MultiHateClip) ships a
``Duration`` column in ``{English,Chinese}_data/annotation/{train,valid,test}.tsv``.
The column holds a Python-literal list of ``(start_sec, end_sec)`` integer tuples
marking the segments annotators flagged as carrying the hateful/offensive content,
and is ``[]`` when annotators marked none. Our local mirror
(``annotation(new).json``) dropped that column, so this script re-joins it.

Inputs
    <data-root>/Multihateclip/upstream_spans/{en,zh}_{train,valid,test}.tsv
    <data-root>/Multihateclip/{English,Chinese}/annotation(new).json

Output
    results/mhclip_localization/span_gold_{en,zh}.json

No transcript text is copied into the output: only IDs, labels, and span numbers.

Video duration is not present in the local mirror (no duration field, no audio or
video files, frames are a fixed 16-frame uniform sample), so span *coverage*
(fraction of the video flagged) cannot be computed and is reported as
unavailable rather than guessed.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

CORPORA = {
    "en": {"prefix": "en", "local_dir": "English"},
    "zh": {"prefix": "zh", "local_dir": "Chinese"},
}
SPLITS = ("train", "valid", "test")
POSITIVE_LABELS = ("Hateful", "Offensive")


def parse_spans(raw: str) -> list[list[float]]:
    """Parse the upstream ``Duration`` literal into a list of [start, end] floats."""
    raw = (raw or "").strip()
    if not raw or raw == "[]":
        return []
    value = ast.literal_eval(raw)
    spans = []
    for item in value:
        start, end = float(item[0]), float(item[1])
        if end < start:
            start, end = end, start
        spans.append([start, end])
    return spans


def read_upstream(span_dir: Path, prefix: str) -> dict[str, dict]:
    """Read the three upstream TSVs for one corpus, keyed by Video_ID.

    Upstream English lists one video (k9OtaMbK0Ac) in both train and test with an
    identical label and identical spans. Such benign repeats are merged and the
    extra split recorded; a repeat that disagrees on label or spans is an error.
    """
    records: dict[str, dict] = {}
    for split in SPLITS:
        path = span_dir / f"{prefix}_{split}.tsv"
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                vid = row["Video_ID"].strip()
                entry = {
                    "split": split,
                    "majority_label": row["Majority_Voting"].strip(),
                    "spans": parse_spans(row["Duration"]),
                }
                prior = records.get(vid)
                if prior is None:
                    records[vid] = entry
                    continue
                same = (
                    prior["majority_label"] == entry["majority_label"]
                    and prior["spans"] == entry["spans"]
                )
                if not same:
                    raise ValueError(
                        f"conflicting duplicate Video_ID {vid} in {prefix}: "
                        f"{prior} vs {entry}"
                    )
                prior.setdefault("extra_splits", []).append(split)
    return records


def read_local(local_json: Path) -> dict[str, str]:
    """Read the local mirror, returning Video_ID -> local Label."""
    data = json.loads(local_json.read_text(encoding="utf-8"))
    return {rec["Video_ID"]: rec.get("Label") for rec in data}


def describe(values: list[float]) -> dict | None:
    if not values:
        return None
    return {
        "n": len(values),
        "min": round(min(values), 3),
        "median": round(statistics.median(values), 3),
        "mean": round(statistics.fmean(values), 3),
        "max": round(max(values), 3),
    }


def build_corpus(code: str, data_root: Path, out_dir: Path) -> dict:
    cfg = CORPORA[code]
    upstream = read_upstream(data_root / "Multihateclip" / "upstream_spans", cfg["prefix"])
    local = read_local(data_root / "Multihateclip" / cfg["local_dir"] / "annotation(new).json")

    missing_upstream = sorted(v for v in local if v not in upstream)
    entries = []
    for vid, local_label in local.items():
        if vid not in upstream:
            continue
        up = upstream[vid]
        spans = up["spans"]
        majority = up["majority_label"]
        entries.append(
            {
                "video_id": vid,
                "split": up["split"],
                "extra_splits": up.get("extra_splits", []),
                "majority_label": majority,
                "local_label": local_label,
                "label_agrees_with_local": majority == local_label,
                "spans": spans,
                "n_spans": len(spans),
                # Upstream contains a few (0, 0) marks; surfaced, not repaired.
                "n_degenerate_spans": sum(1 for s, e in spans if e - s <= 0),
                "normal_with_span": majority == "Normal" and bool(spans),
            }
        )
    entries.sort(key=lambda e: e["video_id"])

    by_label = Counter(e["majority_label"] for e in entries)
    with_span = Counter(e["majority_label"] for e in entries if e["n_spans"] > 0)
    span_lengths = [end - start for e in entries for start, end in e["spans"]]
    per_video_span_counts = [e["n_spans"] for e in entries if e["n_spans"] > 0]
    per_video_flagged = [
        sum(end - start for start, end in e["spans"]) for e in entries if e["n_spans"] > 0
    ]
    max_span_end = [max(end for _, end in e["spans"]) for e in entries if e["n_spans"] > 0]

    positives = [e for e in entries if e["majority_label"] in POSITIVE_LABELS]
    summary = {
        "corpus": code,
        "local_videos": len(local),
        "upstream_videos": len(upstream),
        "matched_videos": len(entries),
        "local_ids_absent_upstream": missing_upstream,
        "videos_by_majority_label": dict(by_label),
        "videos_with_spans_by_label": {k: with_span.get(k, 0) for k in by_label},
        "videos_without_spans_by_label": {k: by_label[k] - with_span.get(k, 0) for k in by_label},
        "positive_videos": len(positives),
        "positive_with_spans": sum(1 for e in positives if e["n_spans"] > 0),
        "positive_without_spans": sum(1 for e in positives if e["n_spans"] == 0),
        "normal_with_span": sum(1 for e in entries if e["normal_with_span"]),
        "videos_with_degenerate_span": sum(1 for e in entries if e["n_degenerate_spans"]),
        "degenerate_spans": sum(e["n_degenerate_spans"] for e in entries),
        "label_mismatch_with_local": sum(1 for e in entries if not e["label_agrees_with_local"]),
        "span_duration_sec": describe(span_lengths),
        "spans_per_annotated_video": describe([float(n) for n in per_video_span_counts]),
        "flagged_seconds_per_annotated_video": describe(per_video_flagged),
        "max_span_end_sec": describe(max_span_end),
        "coverage_fraction": None,
        "coverage_note": (
            "unavailable: the local mirror carries no video duration "
            "(no duration field, no audio/video files, frames are a fixed 16-frame "
            "uniform sample), so flagged-seconds / video-length cannot be computed. "
            "max_span_end_sec is a lower bound on duration, not a coverage figure."
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"span_gold_{code}.json"
    out_path.write_text(
        json.dumps({"summary": summary, "videos": entries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary["_out_path"] = str(out_path)
    return summary


def print_summary(summary: dict) -> None:
    code = summary["corpus"]
    print(f"\n=== MultiHateClip {code.upper()} ===")
    print(f"local videos          : {summary['local_videos']}")
    print(f"upstream videos       : {summary['upstream_videos']}")
    print(f"matched videos        : {summary['matched_videos']}")
    if summary["local_ids_absent_upstream"]:
        print(f"local IDs not upstream: {summary['local_ids_absent_upstream']}")
    print(f"label mismatch vs local mirror: {summary['label_mismatch_with_local']}")
    print("\nby majority label     : videos / with spans / without spans")
    for label, total in sorted(summary["videos_by_majority_label"].items()):
        got = summary["videos_with_spans_by_label"][label]
        print(f"  {label:<10} {total:>5} / {got:>5} / {total - got:>5}")
    print(
        f"  {'H+O':<10} {summary['positive_videos']:>5} / "
        f"{summary['positive_with_spans']:>5} / {summary['positive_without_spans']:>5}"
    )
    print(f"\nNormal-majority videos carrying spans: {summary['normal_with_span']} "
          f"(flagged as normal_with_span=true; handling left to the caller)")
    print(f"zero-length upstream spans: {summary['degenerate_spans']} "
          f"in {summary['videos_with_degenerate_span']} videos (surfaced, not repaired)")
    for key in (
        "span_duration_sec",
        "spans_per_annotated_video",
        "flagged_seconds_per_annotated_video",
        "max_span_end_sec",
    ):
        print(f"{key}: {summary[key]}")
    print(f"coverage_fraction: unavailable -- {summary['coverage_note']}")
    print(f"written: {summary['_out_path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/home/jehc223/data"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results" / "mhclip_localization",
    )
    args = parser.parse_args()
    for code in CORPORA:
        print_summary(build_corpus(code, args.data_root, args.out_dir))


if __name__ == "__main__":
    main()
