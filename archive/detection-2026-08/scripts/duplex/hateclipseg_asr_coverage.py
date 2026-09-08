"""Measure timestamped-ASR coverage over the HateClipSeg split, and list the gap.

Every trained baseline in the reproduction study reads a text branch, so the
train split needs Whisper chunks just as the test split does. The frozen
manifest at ``results/interleaved_timeline/hateclipseg/timestamped_chunks.jsonl``
was produced before this split existed, so what it covers has to be measured
rather than assumed.

This script writes, under ``results/reproduction/asr/hateclipseg_missing/``:

    ids.txt        the split videos with no usable record in the frozen
                   manifest, one id per line, sorted -- the input to
                   ``interleaved_timeline_asr.py --corpus hateclipseg_missing``
    coverage.json  the counts and the reasons, so the gap is auditable
    STATUS, DONE   the run markers this repo uses for detached work

A record counts as usable only if it has chunks, no error, and a positive
``wav_duration``: a present-but-empty record is a gap, not coverage.

CPU only, no model calls. Nothing here transcribes; it decides what to
transcribe.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

_THIS = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))

RESULTS = os.path.join(PROJECT_ROOT, "results")
SPLIT_DIR = os.path.join(RESULTS, "reproduction", "splits")
FROZEN_MANIFEST = os.path.join(
    RESULTS, "interleaved_timeline", "hateclipseg", "timestamped_chunks.jsonl")
OUT_DIR = os.path.join(RESULTS, "reproduction", "asr", "hateclipseg_missing")
WAV_DIRS = [
    os.path.join("/home/jehc223/data", "HateClipSeg", "wav"),
    os.path.join(RESULTS, "hateclipseg", "wav"),
]


def read_ids(path):
    with open(path, encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def load_manifest(path):
    """video id -> record, for records that are actually usable."""
    usable, unusable = {}, {}
    if not os.path.exists(path):
        return usable, unusable
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            vid = rec["video_id"]
            if rec.get("error") or not rec.get("chunks") \
                    or not rec.get("wav_duration"):
                unusable[vid] = {
                    "error": rec.get("error"),
                    "n_chunks": rec.get("n_chunks"),
                    "wav_duration": rec.get("wav_duration"),
                }
                continue
            usable[vid] = rec
    return usable, unusable


def find_wav(vid):
    for wav_dir in WAV_DIRS:
        cand = os.path.join(wav_dir, vid + ".wav")
        if os.path.isfile(cand):
            return cand
    return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    status_path = os.path.join(OUT_DIR, "STATUS")
    with open(status_path, "w", encoding="utf-8") as fh:
        fh.write("running\n")

    splits = {name: read_ids(os.path.join(SPLIT_DIR, "hateclipseg_%s.txt"
                                          % name))
              for name in ("train", "test")}
    usable, unusable = load_manifest(FROZEN_MANIFEST)

    missing = []
    per_split = {}
    for name, ids in splits.items():
        gap = sorted(v for v in ids if v not in usable)
        per_split[name] = {
            "videos": len(ids),
            "covered": len(ids) - len(gap),
            "missing": len(gap),
            "missing_ids": gap,
        }
        missing.extend(gap)
    missing = sorted(set(missing))

    without_wav = sorted(v for v in missing if find_wav(v) is None)
    coverage = {
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "frozen_manifest": os.path.relpath(FROZEN_MANIFEST, PROJECT_ROOT),
        "frozen_manifest_records_usable": len(usable),
        "frozen_manifest_records_unusable": unusable,
        "per_split": per_split,
        "missing_total": len(missing),
        "missing_without_a_wav": without_wav,
        "next_step": (
            "python scripts/duplex/interleaved_timeline_asr.py "
            "--corpus hateclipseg_missing --out-root results/reproduction/asr"
            if missing else "nothing to transcribe"),
    }

    with open(os.path.join(OUT_DIR, "ids.txt"), "w", encoding="utf-8") as fh:
        for vid in missing:
            fh.write(vid + "\n")
    with open(os.path.join(OUT_DIR, "coverage.json"), "w",
              encoding="utf-8") as fh:
        json.dump(coverage, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    for name in ("train", "test"):
        row = per_split[name]
        print("hateclipseg_%-6s %4d videos, %4d covered, %4d missing"
              % (name, row["videos"], row["covered"], row["missing"]))
    print("frozen manifest: %d usable records, %d unusable"
          % (len(usable), len(unusable)))
    print("missing total  : %d  (%d of them have no wav on disk yet)"
          % (len(missing), len(without_wav)))
    print("next           : %s" % coverage["next_step"])

    with open(status_path, "w", encoding="utf-8") as fh:
        fh.write("missing=%d\n" % len(missing))
    with open(os.path.join(OUT_DIR, "DONE"), "w", encoding="utf-8") as fh:
        fh.write("")
    return 0


if __name__ == "__main__":
    sys.exit(main())
