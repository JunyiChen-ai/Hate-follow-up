#!/usr/bin/env python3
"""Score transcript evidence aligned to the frozen visual proposal core."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.idea_discovery.run_cwa_text_warrants import (
    POLICY, QUESTION, inference_chunks, load_map, text_in_window,
)
from scripts.idea_discovery.run_melt import MLLM
from scripts.idea_discovery.run_transcript_existence import make_prompt, score


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--a08", type=Path, required=True)
    parser.add_argument("--a12", type=Path, required=True)
    parser.add_argument("--whole-text", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--char-cap", type=int, default=6000)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    manifest, core = load_map(args.manifest), load_map(args.core)
    a08, a12, whole = load_map(args.a08), load_map(args.a12), load_map(args.whole_text)
    chunks = inference_chunks()
    # The sealed cohort uses a provenance-preserving dataset alias but shares
    # the authoritative HateClipSeg timestamped-ASR source.
    for (dataset, video_id), value in list(chunks.items()):
        if dataset == "HateClipSeg":
            chunks[("HateClipSeg_sealed", video_id)] = value
    keys = sorted(set(manifest) & set(core) & set(a08) & set(a12) & set(whole))
    anchors = {float(whole[key]["config"]["null_log_odds"]) for key in keys}
    if len(anchors) != 1:
        raise RuntimeError(f"expected one whole-text null, got {anchors}")
    whole_anchor = anchors.pop()
    selected = [key for key in keys
                if bool(a08[key].get("intervals")) != bool(a12[key].get("intervals"))
                and float(whole[key]["log_odds"]) < whole_anchor and core[key].get("intervals")]
    examples = []
    for key in selected:
        start = min(float(value[0]) for value in core[key]["intervals"])
        end = max(float(value[1]) for value in core[key]["intervals"])
        local = text_in_window(chunks.get(key, []), start, end)
        examples.append((key, start, end, local))

    model = MLLM(args.model)
    null_score = score(model, [make_prompt("", POLICY, QUESTION, args.char_cap)], 1)[0]
    config = {
        "method": "proposal_scoped_text_v1", "model": args.model,
        "char_cap": args.char_cap, "batch_size": 1,
        "null_log_odds": null_score, "null_prompt": "explicit_empty_transcript",
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "padding_side": model.processor.tokenizer.padding_side, "gt_access": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        for index, (key, start, end, local) in enumerate(examples, 1):
            value = score(model, [make_prompt(local, POLICY, QUESTION, args.char_cap)], 1)[0]
            row = {"dataset": key[0], "video_id": key[1], "proposal": [start, end],
                   "local_log_odds": value, "local_text_chars": len(local),
                   "local_veto": value < null_score, "config": config}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps({"i": index, "n": len(examples), "key": key,
                              "z": value, "chars": len(local), "veto": value < null_score}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
