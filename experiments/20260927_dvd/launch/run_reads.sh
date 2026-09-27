#!/usr/bin/env bash
# DVD reads (README §3) on HateMM + HateClipSeg, then DeHate. HateVLM env; needs data/frames_k20 and
# data/asr_whisper_large_v3 for the three corpora and data/manifests/DeHate_test.jsonl.
# Usage (lab): cd ~/Hate-follow-up && setsid nohup bash experiments/20260927_dvd/launch/run_reads.sh > runs/20260927_dvd/launch_reads.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../../.."
PY="${READS_PYTHON:-$HOME/miniconda3/envs/HateVLM/bin/python}"
echo "host $(hostname) date $(date -Is) commit $(git rev-parse --short HEAD)"
"$PY" experiments/20260927_dvd/dvd_measure.py --run-name reads_main "$@" || { echo "RUN_FAILED reads_main"; exit 1; }
"$PY" experiments/20260927_dvd/dvd_measure.py --run-name reads_dehate --datasets DeHate \
  --manifest data/manifests/DeHate_test.jsonl "$@" || { echo "RUN_FAILED reads_dehate"; exit 1; }
echo "RUN_DONE $(date -Is)"
