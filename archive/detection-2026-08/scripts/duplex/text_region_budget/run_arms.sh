#!/bin/bash
# Detached driver for the three visual arms of the text-region pixel-budget
# probe. One GPU process at a time, resume-safe, survives every disconnect.
#
#   setsid nohup bash scripts/duplex/text_region_budget/run_arms.sh \
#     > results/text_region_budget/logs/arms.log 2>&1 < /dev/null &
#
# Progress: results/text_region_budget/STATUS  (final value DONE or FAILED: <arm>)
# Marker:   results/text_region_budget/DONE

set -u
set -o pipefail

ROOT=/home/jehc223/Hate-follow-up
OUT=$ROOT/results/text_region_budget
LOGS=$OUT/logs
STATUS=$OUT/STATUS
mkdir -p "$LOGS"
rm -f "$OUT/DONE"

cd "$ROOT" || exit 1
export HVD_DATA_ROOT=/home/jehc223/data
export TOKENIZERS_PARALLELISM=false
# shellcheck disable=SC1091
source ~/venvs/SafetyContradiction/bin/activate

say() { echo "[$(date '+%F %T')] $*"; }
stage() { echo "$1" > "$STATUS"; say "=== $1 ==="; }

say "run_arms.sh pid=$$ sid=$(ps -o sid= -p $$ | tr -d ' ')"
say "gpu: $(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader | head -1)"

for ARM in text rand anti; do
  stage "judge_8b_$ARM"
  for attempt in 1 2 3; do
    python -u scripts/duplex/text_region_budget/score_arm.py \
      --manifest "$OUT/manifest_$ARM.json" \
      --out-dir "$OUT/judge_8b_$ARM" \
      --transcript-override-json "$ROOT/results/testruns/mhclip_en/c2_overrides.json" \
      2>&1 | tee -a "$LOGS/judge_8b_$ARM.log"
    rc=${PIPESTATUS[0]}
    [ "$rc" = 0 ] && break
    say "arm $ARM attempt $attempt returned $rc; retrying in 20s"
    sleep 20
  done
  [ "$rc" = 0 ] || { stage "FAILED: judge_8b_$ARM"; exit 1; }
  say "arm $ARM: $(wc -l < "$OUT/judge_8b_$ARM/scores.jsonl") scored"
done

stage "analysis"
python -u scripts/duplex/text_region_budget/analyze.py \
  2>&1 | tee "$LOGS/analyze.log" || { stage "FAILED: analysis"; exit 1; }

stage "DONE"
date -Is > "$OUT/DONE"
say "=== DONE ==="
