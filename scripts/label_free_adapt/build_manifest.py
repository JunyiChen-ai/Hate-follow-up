#!/usr/bin/env python3
"""Build label-free media manifests and duration-balanced pilot cohorts.

The coordinator may read frozen split membership, but emits only id/path/duration.
No annotation, class, span, or positive-rate field is copied to inference files.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

DATASETS = {
    "HateMM": {
        "videos": Path("/home/jehc223/data/HateMM/video"),
        "split": Path("/home/jehc223/data/HateMM/splits/test_clean.csv"),
        "eligible": Path("/home/jehc223/Retrieval-hate/data/gt/HateMM/hate_spans.json"),
        "transcript": Path("/home/jehc223/Hate-follow-up/results/testruns/hatemm/fresh_transcripts.jsonl"),
    },
    "MHC": {
        "videos": Path("/home/jehc223/data/Multihateclip/English/video_mp4"),
        "split": Path("/home/jehc223/Retrieval-hate/data/gt/MHC_temporal/test.jsonl"),
        "transcript": Path("/home/jehc223/Hate-follow-up/results/testruns/mhclip_en/fresh_transcripts.jsonl"),
    },
    "MHC_zh": {
        "videos": Path("/home/jehc223/data/Multihateclip/Chinese/video"),
        "split": Path("/home/jehc223/Retrieval-hate/data/gt/MHC_zh_temporal/test.jsonl"),
        "transcript": Path("/home/jehc223/Hate-follow-up/results/testruns/mhclip_zh/fresh_transcripts.jsonl"),
    },
    "HateClipSeg": {
        "videos": Path("/home/jehc223/data/HateClipSeg/video"),
        "split": Path("/home/jehc223/Retrieval-hate/data/gt/HateClipSeg/p11_split.json"),
        "transcript": Path("/home/jehc223/Hate-follow-up/results/hateclipseg/fresh_transcripts.jsonl"),
    },
}
EXTENSIONS = (".mp4", ".webm", ".mkv", ".avi")


def split_ids(dataset: str, path: Path) -> list[str]:
    if dataset == "HateClipSeg":
        return [str(x) for x in json.loads(path.read_text())["test"]]
    if path.suffix == ".jsonl":
        return [str(json.loads(x)["id"]) for x in path.read_text().splitlines() if x.strip()]
    return [x.strip().split(",")[0] for x in path.read_text().splitlines() if x.strip()]


def find_video(root: Path, video_id: str) -> Path | None:
    for ext in EXTENSIONS:
        path = root / f"{video_id}{ext}"
        if path.exists():
            return path.resolve()
    return None


def duration(path: Path) -> float | None:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, timeout=60, check=False,
    )
    try:
        value = float(result.stdout.strip())
        return value if value > 0 else None
    except ValueError:
        return None


def duration_balanced(rows: list[dict], n: int) -> list[dict]:
    """Deterministic, label-blind sampling over the duration order statistics."""
    if n >= len(rows):
        return list(rows)
    ordered = sorted(rows, key=lambda x: (x["duration"], x["video_id"]))
    indices = [round(i * (len(ordered) - 1) / (n - 1)) for i in range(n)] if n > 1 else [0]
    return [ordered[i] for i in indices]


def load_transcripts(path: Path | None) -> dict[str, str]:
    result = {}
    if path is None or not path.exists():
        return result
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        video_id = str(row.get("video_id", row.get("id", "")))
        text = str(row.get("fresh_text", row.get("text", ""))).strip()
        if video_id:
            result[video_id] = text
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/label_free_adapt/manifests")
    ap.add_argument("--pilot-per-dataset", type=int, default=8)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    aggregate = {"test": [], "pilot": []}
    for dataset in args.datasets.split(","):
        spec = DATASETS[dataset]
        rows, missing, bad_duration = [], [], []
        transcripts = load_transcripts(spec.get("transcript"))
        ids = split_ids(dataset, spec["split"])
        if spec.get("eligible"):
            # Only membership is retained. Annotation values never enter the manifest.
            eligible = set(json.loads(spec["eligible"].read_text()))
            ids = [x for x in ids if x in eligible]
        for video_id in ids:
            path = find_video(spec["videos"], video_id)
            if path is None:
                missing.append(video_id); continue
            seconds = duration(path)
            if seconds is None:
                bad_duration.append(video_id); continue
            rows.append({"dataset": dataset, "video_id": video_id,
                         "video_path": str(path), "duration": seconds,
                         "transcript": transcripts.get(video_id, "")})
        for name, values in (("test", rows),
                             ("pilot", duration_balanced(rows, args.pilot_per_dataset))):
            path = out_dir / f"{dataset}_{name}.jsonl"
            path.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in values))
            aggregate[name].extend(values)
        report[dataset] = {"n_split_ids": len(ids),
                           "n_media": len(rows), "missing_media": missing,
                           "bad_duration": bad_duration,
                           "n_pilot": min(args.pilot_per_dataset, len(rows))}
    for name, values in aggregate.items():
        (out_dir / f"all_{name}.jsonl").write_text(
            "".join(json.dumps(x, sort_keys=True) + "\n" for x in values))
    (out_dir / "manifest_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
