#!/usr/bin/env python3
"""Export inference-only ASR records with a strict field allowlist."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "HateMM": ROOT / "results/hatemm_localization/per_chunk.jsonl",
    "HateClipSeg": ROOT / "results/reproduction/ours/hateclipseg/per_chunk.jsonl",
    "MHC": ROOT / "results/reproduction/ours/mhclip_en/per_chunk.jsonl",
    "MHC_zh": ROOT / "results/reproduction/ours/mhclip_zh/per_chunk.jsonl",
}
STAMP = {
    "HateClipSeg": ROOT / "results/interleaved_timeline/hateclipseg/timestamped_chunks.jsonl",
    "MHC": ROOT / "results/interleaved_timeline/mhclip_en/timestamped_chunks.jsonl",
    "MHC_zh": ROOT / "results/interleaved_timeline/mhclip_zh/timestamped_chunks.jsonl",
}
OUT = ROOT / "results/idea_discovery/paradigm_adapt/asr_inference_only"
ALLOW = ("video_id", "chunk_index", "span", "text", "z_masked", "z_isolated")


def load(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stamped = {}
    for dataset, path in STAMP.items():
        for row in load(path): stamped[(dataset, row["video_id"])] = row.get("chunks", [])
    for dataset, path in SOURCES.items():
        clean = []
        for source in load(path):
            row = {key: source[key] for key in ALLOW if key in source}
            if not row.get("text"):
                chunks = stamped.get((dataset, row["video_id"]), [])
                index = int(row.get("chunk_index", -1))
                row["text"] = chunks[index].get("text", "") if 0 <= index < len(chunks) else ""
            forbidden = set(row) - set(ALLOW)
            if forbidden: raise RuntimeError(f"forbidden exported fields: {forbidden}")
            clean.append(row)
        target = OUT / f"{dataset}.jsonl"
        target.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in clean))
        print(json.dumps({"dataset": dataset, "rows": len(clean), "out": str(target)}))


if __name__ == "__main__": main()
