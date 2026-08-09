#!/bin/bash
# Detached driver for the spec-displacement pilot's GPU stage.
#
#   setsid nohup bash scripts/duplex/run_spec_displacement.sh \
#     > results/spec_displacement/logs/run.log 2>&1 < /dev/null &
#
# Waits for the GPU to be free before loading anything: another pilot holds it
# for hours at a time, and the rule is to queue behind that pilot rather than
# to interrupt it. The wait polls every five minutes and never signals another
# process.
#
# Two processes can still reach for the card in the same gap between polls, so
# the scorer is attempted several times and every attempt re-waits for a free
# GPU first. The scorer itself is resume-safe: a video counts as done only when
# its finite z and its hidden-state array are both on disk, so an interrupted
# attempt costs only the videos it had not finished.
#
# Progress: results/spec_displacement/logs/run.log
# Stage:    results/spec_displacement/STATUS

set -u
set -o pipefail

ROOT=/home/jehc223/Hate-follow-up
OUT=$ROOT/results/spec_displacement
LOGDIR=$OUT/logs
mkdir -p "$OUT" "$LOGDIR"

cd "$ROOT" || exit 1
export HVD_DATA_ROOT=/home/jehc223/data
export TOKENIZERS_PARALLELISM=false
# shellcheck disable=SC1091
source ~/venvs/SafetyContradiction/bin/activate

say() { echo "[$(date '+%F %T')] [specdisp] $*"; }

# Free means: no other process holds meaningful device memory. Xorg's few MiB
# are ignored; an 8B judge or a Whisper pass is not. Two consecutive free polls
# are required, which rejects the brief gap between another job's stages.
FREE_MIB=${FREE_MIB:-3000}
POLL=${POLL:-300}
CONFIRM=${CONFIRM:-20}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-6}

wait_for_gpu() {
  local n=0 used
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ -n "$used" ] && [ "$used" -lt "$FREE_MIB" ]; then
      sleep "$CONFIRM"
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
      if [ -n "$used" ] && [ "$used" -lt "$FREE_MIB" ]; then
        say "GPU free (${used} MiB used, confirmed after $n polls)"
        return 0
      fi
      say "GPU freed then refilled (${used} MiB); keeping the queue"
    fi
    if [ $((n % 6)) = 0 ]; then
      echo "waiting for GPU (${used} MiB used, poll $n)" > "$OUT/STATUS"
      say "GPU busy (${used} MiB used); waiting"
    fi
    n=$((n + 1))
    sleep "$POLL"
  done
}

say "starting; pid=$$ threshold ${FREE_MIB} MiB, poll ${POLL}s, up to ${MAX_ATTEMPTS} attempts"

RC=1
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  wait_for_gpu
  say "=== scoring attempt $attempt ==="
  python -u scripts/duplex/spec_displacement_score.py 2>&1 | tee -a "$LOGDIR/score.log"
  RC=${PIPESTATUS[0]}
  say "scoring attempt $attempt returned rc=$RC"
  [ "$RC" = 0 ] && break
  echo "retrying after attempt $attempt (rc=$RC)" > "$OUT/STATUS"
  sleep 60
done

if [ "$RC" != 0 ]; then
  echo "FAILED: scoring rc=$RC" > "$OUT/STATUS"
  say "giving up after $MAX_ATTEMPTS attempts"
  exit "$RC"
fi

say "=== SCORING_DONE ==="
