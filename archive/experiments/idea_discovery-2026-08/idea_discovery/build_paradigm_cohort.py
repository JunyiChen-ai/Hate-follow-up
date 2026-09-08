#!/usr/bin/env python3
"""Build deterministic paradigm cohorts without opening any ground-truth file."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "results/label_free_adapt/manifests/all_test.jsonl"
FEATURES = Path("/home/jehc223/Retrieval-hate/data/CLIP_Embedding")
ASR = {
    dataset: ROOT / f"results/idea_discovery/paradigm_adapt/asr_inference_only/{dataset}.jsonl"
    for dataset in ("HateMM", "HateClipSeg", "MHC", "MHC_zh")
}


def load(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def load_asr(path):
    output = load(path)
    allowed = {"video_id", "chunk_index", "span", "text", "z_masked", "z_isolated"}
    for row in output:
        if set(row) - allowed: raise RuntimeError(f"label/GT-bearing ASR fields rejected: {set(row)-allowed}")
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-dataset", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--salt", default="paradigm-stage-a-20260826")
    args = ap.parse_args()
    available_asr = {dataset: {x["video_id"] for x in load_asr(path)}
                     for dataset, path in ASR.items()}
    grouped = {dataset: [] for dataset in ASR}
    for row in load(MANIFEST):
        dataset = row["dataset"]; video_id = row["video_id"]
        feature = FEATURES / dataset / "coca_vitL14_4fps" / f"{video_id}.npy"
        if dataset in grouped and video_id in available_asr[dataset] and feature.exists():
            key = hashlib.sha256(f"{args.salt}/{dataset}/{video_id}".encode()).hexdigest()
            grouped[dataset].append((key, row))
    selected = []
    for dataset in ASR:
        chosen = [row for _, row in sorted(grouped[dataset])[:args.per_dataset]]
        if len(chosen) != args.per_dataset:
            raise RuntimeError(f"{dataset}: only {len(chosen)} eligible videos")
        selected.extend(chosen)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in selected))
    print(json.dumps({"out": str(args.out), "n": len(selected),
                      "per_dataset": args.per_dataset, "salt": args.salt}))


if __name__ == "__main__":
    main()
