#!/bin/bash
# Detached driver for the entropy-adaptation kill test.
#
#   setsid nohup bash scripts/duplex/run_entropy_tta.sh \
#     > results/entropy_tta/logs/run.log 2>&1 < /dev/null &
#
# Strictly sequential, one GPU job at a time: the real arm trains and rescores
# to completion before the placebo arm starts, and the descriptive lr-1e-6
# sensitivity arm runs last. Every stage writes its own STATUS and DONE marker,
# so an interrupted run is visible and nothing needs retraining.

set -u
set -o pipefail

ROOT=/home/jehc223/Hate-follow-up
OUT=$ROOT/results/entropy_tta
LOGS=$OUT/logs
PY=/home/jehc223/venvs/SafetyContradiction/bin/python
mkdir -p "$LOGS"
cd "$ROOT" || exit 1

export HVD_DATA_ROOT=/home/jehc223/data
export TOKENIZERS_PARALLELISM=false

say() { echo "[$(date '+%F %T')] $*"; }
stage() { echo "$1" > "$OUT/STATUS"; say "=== STAGE: $1 ==="; }
fail() { stage "FAILED: $1"; say "ABORTING at $1"; exit 1; }

say "run_entropy_tta.sh pid=$$"
say "gpu: $(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader | head -1)"

run_arm() {  # arm_name  train_flags...
  local arm="$1"; shift
  local dir=$OUT/$arm

  stage "train_$arm"
  $PY -u scripts/duplex/entropy_tta.py train --out-dir "$dir" "$@" \
    2>&1 | tee "$LOGS/train_$arm.log" || fail "train_$arm"
  [ -f "$dir/DONE" ] || fail "train_$arm (no DONE marker)"

  stage "score_$arm"
  $PY -u scripts/duplex/entropy_tta.py score --out-dir "$dir" \
    --model-state "$dir/norm_gains.pt" \
    2>&1 | tee "$LOGS/score_$arm.log" || fail "score_$arm"
  [ -f "$dir/DONE" ] || fail "score_$arm (no DONE marker)"
  say "$arm: $(wc -l < "$dir/scores.jsonl") videos rescored"
}

# The real arm first, complete, before anything else touches the GPU.
run_arm real --arm real --lr 1e-5
run_arm placebo --arm placebo --lr 1e-5

# Descriptive only: the pre-registration gives this arm no confirmatory weight.
run_arm real_lr1e6 --arm real --lr 1e-6

stage "analysis"
$PY -u scripts/duplex/entropy_tta_analyze.py \
  --extra-arm "real_lr1e-6_descriptive=$OUT/real_lr1e6/scores.jsonl" \
  --out "$OUT/results.json" 2>&1 | tee "$LOGS/analyze.log" || fail "analysis"

stage "DONE"
touch "$OUT/DONE"
say "=== DONE ==="
