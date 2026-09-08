#!/bin/bash
set -euo pipefail

cd /data/jehc223/EMNLP3

LOCK=/data/jehc223/EMNLP3/results/validation_runs/validation_frames.lock
mkdir -p "$(dirname "$LOCK")"

(
  flock -x 9
  python scripts/validation_audit.py prepare-splits --all
  python src/match_repro/extract_frames.py --all --split validation --workers 8
  python scripts/validation_audit.py frames --all --split validation --require-complete
) 9>"$LOCK"
