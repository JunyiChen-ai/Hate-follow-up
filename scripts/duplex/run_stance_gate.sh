#!/usr/bin/env bash
# Stance-axis gate test: cohort build, probe scoring, analysis.
#
# Launch detached:
#   setsid nohup bash scripts/duplex/run_stance_gate.sh \
#     > results/stance_gate/logs/run.log 2>&1 < /dev/null &
set -uo pipefail

cd /home/jehc223/Hate-follow-up
export HVD_DATA_ROOT=/home/jehc223/data
export TOKENIZERS_PARALLELISM=false
source ~/venvs/SafetyContradiction/bin/activate

OUT=results/stance_gate
GPULOCK=results/testruns/gpu.lock
mkdir -p "$OUT/logs"
[ -e "$GPULOCK" ] || : > "$GPULOCK"
STATUS="$OUT/STATUS"

say()   { echo "[$(date '+%F %T')] $*"; }
stage() { echo "$1" > "$STATUS"; say "=== STAGE: $1 ==="; }
fail()  { echo "FAILED: $1" > "$STATUS"; say "!!! FAILED at $1"; exit 1; }

stage cohorts
python -u scripts/duplex/stance_gate_cohorts.py \
  --out "$OUT/cohorts.json" > "$OUT/logs/cohorts.log" 2>&1 || fail cohorts
tail -n 12 "$OUT/logs/cohorts.log"

stage probe
say "waiting for the GPU lock (probe)"
flock "$GPULOCK" python -u scripts/duplex/stance_gate_probe.py \
  --cohorts "$OUT/cohorts.json" \
  --out "$OUT/probe_scores.jsonl" > "$OUT/logs/probe.log" 2>&1 || fail probe
say "released the GPU lock (probe)"
tail -n 5 "$OUT/logs/probe.log"

stage analyze
python -u scripts/duplex/stance_gate_analyze.py \
  --cohorts "$OUT/cohorts.json" \
  --scores "$OUT/probe_scores.jsonl" \
  --out docs/duplex/reports/stance_gate_diag.json > "$OUT/logs/analyze.log" 2>&1 || fail analyze
tail -n 40 "$OUT/logs/analyze.log"

stage note
python -u scripts/duplex/testrun_note.py docs/duplex/TEST_RUNS_NOTE.md \
  > "$OUT/logs/note.log" 2>&1 || fail note

stage DONE
say "all stages complete"
