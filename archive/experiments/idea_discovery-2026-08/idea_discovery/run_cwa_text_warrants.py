#!/usr/bin/env python3
"""Score native, timestamp-matched transcript views for CWA.

This is a development-cohort pilot.  It reads only inference fields from the
ASR files (video id, span, and text); annotation fields present in legacy files
are never copied into the in-memory records or output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.idea_discovery.run_melt import MLLM
from scripts.idea_discovery.run_transcript_existence import make_prompt, score

ROOT = Path(__file__).resolve().parents[2]
CHUNK_SOURCES = {
    "HateMM": ROOT / "results/hatemm_localization/per_chunk.jsonl",
    "HateClipSeg": ROOT / "results/reproduction/ours/hateclipseg/per_chunk.jsonl",
    "MHC": ROOT / "results/reproduction/ours/mhclip_en/per_chunk.jsonl",
    "MHC_zh": ROOT / "results/reproduction/ours/mhclip_zh/per_chunk.jsonl",
}
STAMP_SOURCES = {
    "HateClipSeg": ROOT / "results/interleaved_timeline/hateclipseg/timestamped_chunks.jsonl",
    "MHC": ROOT / "results/interleaved_timeline/mhclip_en/timestamped_chunks.jsonl",
    "MHC_zh": ROOT / "results/interleaved_timeline/mhclip_zh/timestamped_chunks.jsonl",
}
POLICY = (
    "Hateful content is an asserted or endorsed attack, dehumanization, threat, "
    "exclusion, or hateful stereotype directed at a person or protected/social "
    "group. Quotation, neutral reporting, counterspeech, satire, and condemnation "
    "are not endorsement."
)
QUESTION = "Does this transcript contain at least one complete asserted or endorsed hateful event?"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_map(path: Path) -> dict[tuple[str, str], dict]:
    return {(row["dataset"], row["video_id"]): row for row in load_jsonl(path)}


def inference_chunks() -> dict[tuple[str, str], list[dict]]:
    stamped: dict[tuple[str, str], list[dict]] = {}
    for dataset, path in STAMP_SOURCES.items():
        for row in load_jsonl(path):
            stamped[(dataset, str(row["video_id"]))] = [
                {"span": [float(c["start"]), float(c["end"])], "text": str(c.get("text", ""))}
                for c in row.get("chunks", [])
                if c.get("start") is not None and c.get("end") is not None
            ]

    out: dict[tuple[str, str], list[dict]] = {}
    for dataset, path in CHUNK_SOURCES.items():
        for source in load_jsonl(path):
            video_id = str(source["video_id"])
            span = source.get("span")
            if not span or len(span) != 2:
                continue
            text = source.get("text")
            if text is None:
                index = int(source.get("chunk_index", -1))
                candidates = stamped.get((dataset, video_id), [])
                text = candidates[index]["text"] if 0 <= index < len(candidates) else ""
            # Explicit allowlist: do not retain labels, overlap, gold, or model scores.
            out.setdefault((dataset, video_id), []).append({
                "span": [float(span[0]), float(span[1])], "text": str(text or "")
            })
    for chunks in out.values():
        chunks.sort(key=lambda item: (item["span"][0], item["span"][1]))
    return out


def text_in_window(chunks: list[dict], start: float, end: float) -> str:
    return " ".join(
        item["text"].strip() for item in chunks
        if item["span"][1] > start and item["span"][0] < end and item["text"].strip()
    )


def matched_windows(start: float, end: float, duration: float) -> tuple[list[float] | None, list[float] | None]:
    width = max(0.0, end - start)
    left = [start - width, start] if width > 0 and start >= width else None
    right = [end, end + width] if width > 0 and end + width <= duration else None
    return left, right


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--extent", type=Path, required=True)
    parser.add_argument("--a08", type=Path, required=True)
    parser.add_argument("--a12", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--char-cap", type=int, default=6000)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")

    manifest, extent = load_map(args.manifest), load_map(args.extent)
    a08, a12 = load_map(args.a08), load_map(args.a12)
    chunks = inference_chunks()
    keys = sorted(set(manifest) & set(extent) & set(a08) & set(a12))
    keys = [key for key in keys if bool(a08[key].get("intervals")) != bool(a12[key].get("intervals"))]

    examples = []
    for key in keys:
        intervals = extent[key].get("intervals", [])
        if not intervals:
            continue
        start = min(float(item[0]) for item in intervals)
        end = max(float(item[1]) for item in intervals)
        duration = float(manifest[key]["duration"])
        left, right = matched_windows(start, end, duration)
        local_chunks = chunks.get(key, [])
        examples.append({
            "key": key,
            "proposal": [start, end],
            "inside": text_in_window(local_chunks, start, end),
            "left_window": left,
            "left": text_in_window(local_chunks, *left) if left else None,
            "right_window": right,
            "right": text_in_window(local_chunks, *right) if right else None,
        })

    model = MLLM(args.model)
    null_score = score(model, [make_prompt("", POLICY, QUESTION, args.char_cap)], 1)[0]
    config = {
        "method": "cwa_text_native_matched_views_v1",
        "model": args.model,
        "char_cap": args.char_cap,
        "batch_size": 1,
        "null_log_odds": null_score,
        "null_prompt": "explicit_empty_transcript",
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "padding_side": model.processor.tokenizer.padding_side,
        "gt_access": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        for index, example in enumerate(examples, 1):
            view_text = [example["inside"]]
            view_names = ["inside"]
            for name in ("left", "right"):
                if example[name] is not None:
                    view_names.append(name)
                    view_text.append(example[name])
            values = score(model, [make_prompt(t, POLICY, QUESTION, args.char_cap) for t in view_text], 1)
            scores = dict(zip(view_names, values))
            controls = [scores[name] for name in ("left", "right") if name in scores]
            inside = scores["inside"]
            support = len(controls) == 2 and inside > null_score and all(inside > value for value in controls)
            oppose = len(controls) == 2 and inside < null_score and all(inside < value for value in controls)
            state = "support" if support else "oppose" if oppose else "abstain"
            row = {
                "dataset": example["key"][0], "video_id": example["key"][1],
                "proposal": example["proposal"],
                "windows": {"left": example["left_window"], "right": example["right_window"]},
                "scores": scores, "text_lengths": {name: len(text) for name, text in zip(view_names, view_text)},
                "warrant": state, "config": config,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps({"i": index, "n": len(examples), "key": example["key"], "warrant": state}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
