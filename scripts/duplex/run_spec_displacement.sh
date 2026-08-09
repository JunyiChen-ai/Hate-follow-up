#!/bin/bash
# Detached driver for the spec-displacement pilot's GPU stage.
#
#   setsid nohup bash scripts/duplex/run_spec_displacement.sh \
#     > results/spec_displacement/logs/run.log 2>&1 < /dev/null &
#
# Waits for the GPU to be free before loading anything: another pilot may be
# holding it, and the rule is to queue behind it rather than to interrupt it.
# The wait polls every five minutes and never signals another process.
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
# are ignored; an 8B judge or a Whisper pass is not.
FREE_MIB=${FREE_MIB:-2000}
POLL=${POLL:-300}

say "starting; pid=$$ waiting for the GPU (threshold ${FREE_MIB} MiB, poll ${POLL}s)"
n=0
while :; do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ -n "$USED" ] && [ "$USED" -lt "$FREE_MIB" ]; then
    say "GPU free (${USED} MiB used after $n polls)"
    break
  fi
  if [ $((n % 6)) = 0 ]; then
    echo "waiting for GPU (${USED} MiB used, poll $n)" > "$OUT/STATUS"
    say "GPU busy (${USED} MiB used); waiting"
  fi
  n=$((n + 1))
  sleep "$POLL"
done

say "=== scoring ==="
python -u scripts/duplex/spec_displacement_score.py 2>&1 | tee -a "$LOGDIR/score.log"
RC=${PIPESTATUS[0]}
say "scoring returned rc=$RC"
if [ "$RC" != 0 ]; then
  echo "FAILED: scoring rc=$RC" > "$OUT/STATUS"
  exit "$RC"
fi

say "=== SCORING_DONE ==="
