#!/usr/bin/env python3
"""Score natural timestamped ASR chunks with one frozen stance scorer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.idea_discovery.run_cwa_text_warrants import POLICY, QUESTION
from scripts.idea_discovery.run_melt import MLLM
from scripts.idea_discovery.run_transcript_existence import make_prompt, score

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "HateMM": ROOT / "results/hatemm_localization/timestamped_chunks.jsonl",
    "HateClipSeg": ROOT / "results/interleaved_timeline/hateclipseg/timestamped_chunks.jsonl",
    "MHC": ROOT / "results/interleaved_timeline/mhclip_en/timestamped_chunks.jsonl",
    "MHC_zh": ROOT / "results/interleaved_timeline/mhclip_zh/timestamped_chunks.jsonl",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--char-cap", type=int, default=6000)
    parser.add_argument("--source", action="append", default=[],
                        help="Additional/override timestamp source as DATASET=PATH")
    parser.add_argument("--dataset-alias", action="append", default=[],
                        help="Proposal dataset alias as TARGET=SOURCE")
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    keys = {(row["dataset"], str(row["video_id"]))
            for row in map(json.loads, args.proposals.open())}
    examples = []
    source_hashes = {}
    source_items = list(SOURCES.items())
    aliases = dict(value.split("=", 1) for value in args.dataset_alias)
    for value in args.source:
        dataset, raw_path = value.split("=", 1)
        source_items.append((dataset, Path(raw_path)))
    records = {}
    for dataset, path in source_items:
        source_hashes[f"{dataset}:{path}"] = hashlib.sha256(path.read_bytes()).hexdigest()
        for row in map(json.loads, path.open()):
            target_dataset = next((target for target, source in aliases.items()
                                   if source == dataset and (target, str(row["video_id"])) in keys),
                                  dataset)
            key = (target_dataset, str(row["video_id"]))
            if key not in keys:
                continue
            chunks = row.get("chunks", row.get("segments", []))
            # Later sources override only when they contain usable segments;
            # this lets fresh WAV ASR supplement rather than erase old coverage.
            if chunks or key not in records:
                records[key] = chunks
    for (dataset, video_id), chunks in sorted(records.items()):
            for index, chunk in enumerate(chunks):
                start, end = chunk.get("start"), chunk.get("end")
                if start is None or end is None or float(end) <= float(start):
                    continue
                examples.append({"dataset": dataset, "video_id": video_id,
                                 "chunk_index": index, "start": float(start),
                                 "end": float(end), "text": str(chunk.get("text", ""))})
    model = MLLM(args.model)
    null = score(model, [make_prompt("", POLICY, QUESTION, args.char_cap)], 1)[0]
    prompts = [make_prompt(row["text"], POLICY, QUESTION, args.char_cap) for row in examples]
    values = score(model, prompts, args.batch_size)
    config = {"model": args.model, "prompt": "stance_complete_asserted_event",
              "null_log_odds": null, "null_prompt": "explicit_empty_transcript",
              "batch_size": args.batch_size,
              "padding_side": model.processor.tokenizer.padding_side,
              "source_sha256": source_hashes,
              "proposals_sha256": hashlib.sha256(args.proposals.read_bytes()).hexdigest(),
              "gt_access": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        for row, value in zip(examples, values):
            # Text is intentionally omitted from the artifact; only native
            # timestamp and inference output are retained.
            handle.write(json.dumps({k: v for k, v in row.items() if k != "text"} |
                                    {"log_odds": value, "config": config},
                                    ensure_ascii=False) + "\n")
    print(json.dumps({"videos": len({(x['dataset'], x['video_id']) for x in examples}),
                      "chunks": len(examples), "null_log_odds": null,
                      "calls": model.calls, "positive_chunks": sum(v > null for v in values)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
