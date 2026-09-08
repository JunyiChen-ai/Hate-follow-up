#!/usr/bin/env python3
"""Fail unless two VERA per-video raw artifacts are exactly equivalent."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    args = parser.parse_args()
    ref_path, candidate_path = map(Path, (args.reference, args.candidate))
    reference = json.loads(ref_path.read_text())
    candidate = json.loads(candidate_path.read_text())

    for key in ("video_id", "duration"):
        if reference[key] != candidate[key]:
            raise SystemExit(f"mismatch {key}: {reference[key]!r} != {candidate[key]!r}")
    if len(reference["segments"]) != len(candidate["segments"]):
        raise SystemExit("segment count mismatch")
    keys = ("start", "end", "score", "response")
    for index, (left, right) in enumerate(zip(reference["segments"],
                                               candidate["segments"])):
        for key in keys:
            if left[key] != right[key]:
                raise SystemExit(
                    f"segment {index} mismatch {key}: {left[key]!r} != {right[key]!r}")
    print(json.dumps({
        "exact": True,
        "video_id": reference["video_id"],
        "segments": len(reference["segments"]),
        "reference_sha256": digest(ref_path),
        "candidate_sha256": digest(candidate_path),
    }, indent=2))


if __name__ == "__main__":
    main()
