#!/usr/bin/env python3
"""Build label-free timestamped ASR records from sanitized manifests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="openai/whisper-large-v3")
    parser.add_argument("--audio-root", default="",
                        help="Optional directory of <video_id>.wav files")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines()
            if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    destination = Path(args.out)
    latest = {}
    if destination.exists():
        for line in destination.read_text().splitlines():
            row = json.loads(line)
            latest[(row["dataset"], row["video_id"])] = row

    from transformers import pipeline
    recognizer = pipeline(
        "automatic-speech-recognition", model=args.model,
        torch_dtype=torch.float16, device=0,
        model_kwargs={"local_files_only": True},
        chunk_length_s=30, batch_size=16,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        for row in rows:
            key = (row["dataset"], row["video_id"])
            if key in latest and not latest[key].get("error"):
                continue
            record = {"dataset": key[0], "video_id": key[1], "segments": [],
                      "model": args.model, "error": None}
            try:
                source = (str(Path(args.audio_root) / f"{row['video_id']}.wav")
                          if args.audio_root else row["video_path"])
                output = recognizer(source, return_timestamps=True,
                                    generate_kwargs={"task": "transcribe"})
                record["segments"] = [
                    {"start": float(chunk["timestamp"][0] or 0.0),
                     "end": float(chunk["timestamp"][1] or row["duration"]),
                     "text": chunk["text"].strip()}
                    for chunk in output.get("chunks", []) if chunk.get("text", "").strip()
                ]
                record["text"] = output.get("text", "").strip()
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
