#!/usr/bin/env python3
"""Rebuild the frozen Retrieval-hate 4 FPS GT from authoritative span sources."""
from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

import numpy as np

FPS = 4.0
ROOT = Path("/home/jehc223/Hate-follow-up")
DATA = Path("/home/jehc223/data")
RETRIEVAL = Path("/home/jehc223/Retrieval-hate")


def rasterize(spans, duration):
    n = max(1, int(np.floor(float(duration) * FPS)))
    t = np.arange(n, dtype=float) / FPS
    y = np.zeros(n, dtype=np.int8)
    clean = []
    for start, end in spans:
        start, end = max(0.0, float(start)), min(float(duration), float(end))
        if end > start:
            y[(t >= start) & (t < end)] = 1
            clean.append([start, end])
    clean.sort()
    merged = []
    for start, end in clean:
        if merged and start <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return y, np.asarray(merged, dtype=float).reshape(-1, 2)


def manifests(path):
    return {x["video_id"]: x for x in
            (json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())}


def mhc_rows(code):
    out = {}
    # Retrieval-hate uses a frozen temporal split independent of the upstream
    # train/valid/test partition, so collect gold spans from all three TSVs.
    for split in ("train", "valid", "test"):
        path = DATA / "Multihateclip/upstream_spans" / f"{code}_{split}.tsv"
        with path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                raw = ast.literal_eval(row["Duration"] or "[]")
                spans = []
                for value in raw:
                    if isinstance(value, (list, tuple)) and len(value) >= 2:
                        spans.append([float(value[0]), float(value[1])])
                out.setdefault(row["Video_ID"], spans)
    return out


def dataset_spans(dataset, ids):
    if dataset == "HateMM":
        source = json.loads((ROOT / "results/hatemm_localization/span_gold.json").read_text())["spans"]
        return {x: source.get(x, []) for x in ids}, {}
    if dataset == "MHC":
        source = mhc_rows("en")
        return {x: source.get(x, []) for x in ids}, {}
    if dataset == "MHC_zh":
        source = mhc_rows("zh")
        return {x: source.get(x, []) for x in ids}, {}
    source = json.loads((RETRIEVAL / "data/gt/HateClipSeg/gold_segments.json").read_text())
    spans, durations = {}, {}
    for video_id in ids:
        entry = source[video_id]; durations[video_id] = float(entry["duration"])
        spans[video_id] = [[a, b] for a, b, dims in entry["segments"] if any(dims[1:])]
    return spans, durations


def build(dataset, manifest_dir, out_dir):
    split_sources = {
        "HateMM": DATA / "HateMM/splits/test_clean.csv",
        "MHC": RETRIEVAL / "data/gt/MHC_temporal/test.jsonl",
        "MHC_zh": RETRIEVAL / "data/gt/MHC_zh_temporal/test.jsonl",
        "HateClipSeg": RETRIEVAL / "data/gt/HateClipSeg/p11_split.json",
    }
    if dataset == "HateClipSeg":
        ids = [str(x) for x in json.loads(split_sources[dataset].read_text())["test"]]
    elif split_sources[dataset].suffix == ".jsonl":
        ids = [str(json.loads(x)["id"]) for x in split_sources[dataset].read_text().splitlines() if x.strip()]
    else:
        ids = [x.strip() for x in split_sources[dataset].read_text().splitlines() if x.strip()]
    media = manifests(Path(manifest_dir) / f"{dataset}_test.jsonl")
    span_map, duration_fallback = dataset_spans(dataset, ids)
    video_ids, durations, labels, spans_out = [], [], [], []
    missing_duration = []
    for video_id in ids:
        duration = media.get(video_id, {}).get("duration", duration_fallback.get(video_id))
        if not duration:
            missing_duration.append(video_id); continue
        y, spans = rasterize(span_map.get(video_id, []), duration)
        video_ids.append(video_id); durations.append(duration); labels.append(y); spans_out.append(spans)
    obj = lambda values: np.asarray(values + [None], dtype=object)[:-1]
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{dataset}.npz",
                        video_ids=np.asarray(video_ids), split=np.asarray(["test"] * len(video_ids)),
                        duration=np.asarray(durations), y4=obj(labels), spans=obj(spans_out),
                        n_spans=np.asarray([len(x) for x in spans_out], dtype=np.int16))
    return {"dataset": dataset, "n_videos": len(video_ids),
            "n_frames": sum(map(len, labels)), "n_positive": sum(int(x.sum()) for x in labels),
            "base_rate": sum(int(x.sum()) for x in labels) / max(1, sum(map(len, labels))),
            "missing_duration": missing_duration}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", default="results/label_free_adapt/manifests")
    ap.add_argument("--out-dir", default="results/label_free_adapt/gt_4fps")
    args = ap.parse_args()
    report = [build(x, args.manifest_dir, args.out_dir)
              for x in ("HateMM", "MHC", "MHC_zh", "HateClipSeg")]
    path = Path(args.out_dir) / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
